"""The MCP interface: one session's tools served to an MCP client.

The server runs in process against a fake kernel (no Julia), through the SDK's
own client, so these exercise the protocol path a real client takes: list and
call tools, live output as progress, cancellation, and links into the viewer.
"""

from __future__ import annotations

import asyncio
import base64
import time
from pathlib import Path
from typing import Any

import anyio
import httpx
import mcp_types as types
import pytest
from mcp import Client

from jutul_agent.juliakernel import EvalResult
from jutul_agent.lab.fakes import FakeJulia, make_fake_adapter

EXPECTED_TOOLS = [
    "run_julia",
    "reset_julia",
    "plot_julia",
    "recapture_plot",
    "close_plots",
    "record_attempt",
    "write_report",
]


@pytest.fixture
def fake_kernel(monkeypatch) -> FakeJulia:
    """Patch out Julia, env preparation and the warm-up; return the session's kernel."""
    import jutul_agent.juliakernel as juliakernel_mod
    from jutul_agent.agent import builder as builder_mod
    from jutul_agent.julia import requirements as requirements_mod
    from jutul_agent.simulators import warmup as warmup_mod

    kernel = FakeJulia()
    monkeypatch.setattr(requirements_mod, "require_julia", lambda *a, **kw: None)
    monkeypatch.setattr(juliakernel_mod, "JuliaKernel", lambda config: kernel)
    monkeypatch.setattr(builder_mod, "resolve_package_sources", lambda project: [])
    monkeypatch.setattr(warmup_mod, "start_warmup", lambda *a, **kw: None)
    return kernel


def _launch(tmp_path: Path, **overrides: Any):
    from jutul_agent.interfaces.mcp.runtime import McpLaunch

    workspace = tmp_path / "ws"
    workspace.mkdir(exist_ok=True)
    options: dict[str, Any] = {
        "simulator": make_fake_adapter(tmp_path),
        "workspace": workspace,
        "prepare_env": False,
        "virtual_display": False,
        "viewer": False,
    }
    options.update(overrides)
    return McpLaunch(**options)


def _text(result: types.CallToolResult) -> str:
    return "\n".join(b.text for b in result.content if isinstance(b, types.TextContent))


@pytest.mark.parametrize("mode", ["auto", "legacy"])
async def test_lists_the_session_tools_with_their_specs(fake_kernel, tmp_path, mode) -> None:
    from jutul_agent.interfaces.mcp.runtime import McpRuntime
    from jutul_agent.interfaces.mcp.server import build_server

    async with Client(build_server(McpRuntime(_launch(tmp_path))), mode=mode) as client:
        listed = (await client.list_tools()).tools
        assert client.instructions and "FakeSim" in client.instructions

    assert [tool.name for tool in listed] == EXPECTED_TOOLS
    by_name = {tool.name: tool for tool in listed}
    assert by_name["run_julia"].title == "Julia"
    assert "code" in by_name["run_julia"].input_schema["properties"]
    # The id a plot is filed under is the framework's to inject, not the caller's.
    assert "tool_call_id" not in by_name["plot_julia"].input_schema.get("properties", {})


async def test_a_call_runs_in_the_session_and_is_traced(fake_kernel, tmp_path) -> None:
    from jutul_agent.interfaces.mcp.runtime import McpRuntime
    from jutul_agent.interfaces.mcp.server import build_server
    from jutul_agent.trace.schema import TOOL_CALL, TOOL_RESULT

    fake_kernel._answers["1 + 1"] = "2"
    runtime = McpRuntime(_launch(tmp_path))
    async with Client(build_server(runtime)) as client:
        result = await client.call_tool("run_julia", {"code": "1 + 1"})
        trace = runtime.core.session.trace
        events = [(e.kind, e.payload) for e in trace.events_after(0)]

    assert not result.is_error
    assert "2" in _text(result)
    assert "1 + 1" in fake_kernel.calls
    calls = [p for kind, p in events if kind == TOOL_CALL]
    results = [p for kind, p in events if kind == TOOL_RESULT]
    assert calls[-1]["name"] == "run_julia" and calls[-1]["args"] == {"code": "1 + 1"}
    assert results[-1]["tool_call_id"] == calls[-1]["id"]
    assert results[-1]["status"] == "success"


async def test_failures_come_back_as_error_results(fake_kernel, tmp_path) -> None:
    from jutul_agent.interfaces.mcp.runtime import McpRuntime
    from jutul_agent.interfaces.mcp.server import build_server

    async with Client(build_server(McpRuntime(_launch(tmp_path)))) as client:
        unknown = await client.call_tool("no_such_tool", {})
        missing_argument = await client.call_tool("run_julia", {})

    assert unknown.is_error and "no_such_tool" in _text(unknown)
    assert missing_argument.is_error and "run_julia" in _text(missing_argument)


async def test_a_startup_failure_is_reported_to_the_client(monkeypatch, tmp_path) -> None:
    from jutul_agent.interfaces.mcp.runtime import McpRuntime
    from jutul_agent.interfaces.mcp.server import build_server
    from jutul_agent.julia import requirements as requirements_mod

    def _no_julia(*_a, **_kw):
        raise RuntimeError("Julia is not installed")

    monkeypatch.setattr(requirements_mod, "require_julia", _no_julia)
    async with Client(build_server(McpRuntime(_launch(tmp_path)))) as client:
        result = await client.call_tool("run_julia", {"code": "1"})

    assert result.is_error
    assert "could not start" in _text(result) and "Julia is not installed" in _text(result)


@pytest.mark.parametrize("mode", ["auto", "legacy"])
async def test_live_output_reaches_the_client_as_progress(fake_kernel, tmp_path, mode) -> None:
    from jutul_agent.interfaces.mcp.runtime import McpRuntime
    from jutul_agent.interfaces.mcp.server import build_server

    async def _slow(code: str) -> EvalResult:
        await asyncio.sleep(0.3)  # long enough for the relay to forward a line
        return EvalResult(output="done")

    fake_kernel._stream_chunks = ["Step 1\n", "\x1b[32mStep 2 of 3\x1b[0m\rStep 3 of 3\n"]
    fake_kernel._eval_handler = _slow
    seen: list[tuple[float, str | None]] = []

    async def _on_progress(progress: float, total: float | None, message: str | None) -> None:
        seen.append((progress, message))

    async with Client(build_server(McpRuntime(_launch(tmp_path))), mode=mode) as client:
        result = await client.call_tool(
            "run_julia", {"code": "solve"}, progress_callback=_on_progress
        )

    assert not result.is_error
    # The line a terminal would show: the last one, overwrites applied, colour removed.
    assert seen and seen[0] == (1, "Step 3 of 3")


async def test_a_cancelled_call_lets_julia_recover_first(fake_kernel, tmp_path) -> None:
    """One cancellation reaches the eval, and its recovery runs to the end.

    The kernel answers a cancelled eval by interrupting Julia and waiting for it,
    shielded. An anyio cancel scope (how an MCP cancel arrives) re-delivers the
    cancellation, which would abandon that wait if the eval ran in the scope.
    """
    from jutul_agent.interfaces.mcp.runtime import McpRuntime
    from jutul_agent.trace.schema import TOOL_RESULT

    state: dict[str, Any] = {"cancelled": 0}
    started = asyncio.Event()

    async def _recover() -> None:
        await asyncio.sleep(0.2)
        state["recovered"] = True

    async def _eval(code: str) -> EvalResult:
        started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            state["cancelled"] += 1
            await asyncio.shield(_recover())
            state["waited"] = True
            raise
        return EvalResult(output="finished")

    fake_kernel._eval_handler = _eval
    runtime = McpRuntime(_launch(tmp_path))
    await runtime.start()
    try:
        await runtime.ready()
        async with anyio.create_task_group() as group:
            group.start_soon(runtime.call, "run_julia", {"code": "sleep(30)"})
            await started.wait()
            group.cancel_scope.cancel()
        assert state == {"cancelled": 1, "recovered": True, "waited": True}
        last = list(runtime.core.session.trace.events_after(0))[-1]
        assert last.kind == TOOL_RESULT and last.payload["status"] == "error"
    finally:
        await runtime.aclose()


async def test_the_next_result_says_julia_restarted_and_which_plots_died(
    fake_kernel, tmp_path
) -> None:
    """When a cancelled eval ignores the interrupt, Julia restarts. The client never
    sees the cancelled call's result, so the next one carries the news, once: when,
    why, and which live plots no longer update."""
    from jutul_agent.interfaces.mcp.runtime import McpRuntime
    from jutul_agent.interfaces.mcp.server import call_result
    from jutul_agent.juliakernel.kernel import KernelRestart
    from jutul_agent.trace.schema import ARTIFACT, TOOL_RESULT, artifact_payload

    runtime = McpRuntime(_launch(tmp_path))
    started = asyncio.Event()

    async def _eval(code: str) -> EvalResult:
        if code == "plot":
            runtime.core.session.trace.append(
                ARTIFACT,
                artifact_payload(
                    path="artifacts/p.png",
                    mime="image/png",
                    caption="Pressure",
                    kind="plot",
                    slot="pressure",
                    live_url="/live/s/viz/pressure",
                ),
            )
        if code == "stuck":
            started.set()
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                # What the kernel does when the interrupt goes unheeded.
                fake_kernel.generation += 1
                fake_kernel.last_restart = KernelRestart(time.time(), "the eval did not stop")
                raise
        return EvalResult(output="42")

    fake_kernel._eval_handler = _eval
    fake_kernel.generation = 1
    await runtime.start()
    try:
        await runtime.ready()
        assert (await runtime.call("run_julia", {"code": "plot"})).notice is None
        async with anyio.create_task_group() as group:
            group.start_soon(runtime.call, "run_julia", {"code": "stuck"})
            await started.wait()
            group.cancel_scope.cancel()
        cancelled = list(runtime.core.session.trace.events_after(0))[-1]
        after = call_result(await runtime.call("run_julia", {"code": "x"}), [])
        again = call_result(await runtime.call("run_julia", {"code": "x"}), [])
    finally:
        await runtime.aclose()

    assert cancelled.kind == TOOL_RESULT
    restarted = "Julia restarted because the eval did not stop."
    assert cancelled.payload["content"] == f"Cancelled. {restarted}"
    assert not after.is_error
    notice = after.content[0].text
    assert notice.startswith(
        "Note: Julia restarted under a minute ago because the eval did not stop. "
        "Variables and `using` imports from before are gone."
    )
    assert 'no longer update (plot them again): "Pressure"' in notice
    assert "42" in after.content[1].text
    assert "restarted" not in _text(again)


async def test_a_reset_says_so_itself(fake_kernel, tmp_path) -> None:
    """``reset_julia`` restarts Julia because the client asked; its own result says
    so, and no notice repeats it."""
    from jutul_agent.interfaces.mcp.runtime import McpRuntime
    from jutul_agent.juliakernel.kernel import KernelRestart

    async def _reset() -> EvalResult:
        fake_kernel.generation += 1
        fake_kernel.last_restart = KernelRestart(time.time(), "it was reset")
        return EvalResult(output="")

    fake_kernel.generation = 1
    fake_kernel.reset = _reset
    runtime = McpRuntime(_launch(tmp_path))
    await runtime.start()
    try:
        reset = await runtime.call("reset_julia", {})
        after = await runtime.call("run_julia", {"code": "x"})
    finally:
        await runtime.aclose()

    assert fake_kernel.generation == 2
    assert "restarted" in str(reset.content)
    assert reset.notice is None and after.notice is None


def test_long_text_reaches_the_client_as_its_start_and_end(tmp_path) -> None:
    from jutul_agent.interfaces.mcp.runtime import TEXT_LIMIT, shorten_long_text

    text = "\n".join(f"line {i:06d}" for i in range(10_000))  # 120 000 characters
    saved = tmp_path / "tool-results" / "call.txt"
    short = shorten_long_text(text, saved)

    assert len(short) < TEXT_LIMIT
    assert short.startswith("line 000000\n") and short.endswith("\nline 009999")
    assert saved.read_text(encoding="utf-8") == text
    head, rest = short.split("\n[... ", 1)
    note, tail = rest.split("]\n\n", 1)
    assert f"characters omitted. The full text is saved in {saved}" in note
    # Cut at line ends: every line kept is whole.
    assert {len(line) for line in head.splitlines() + tail.splitlines()} == {11}
    # Short text and images pass through; each long text is saved on its own.
    image = {"type": "image", "base64": "AAAA"}
    blocks = shorten_long_text([{"type": "text", "text": "ok"}, image, text, text], saved)
    assert blocks[:2] == [{"type": "text", "text": "ok"}, image]
    assert str(saved) in blocks[2] and str(saved.with_name("call-2.txt")) in blocks[3]
    assert shorten_long_text("short", saved) == "short"


async def test_the_trace_keeps_a_long_result_whole(fake_kernel, tmp_path) -> None:
    from jutul_agent.interfaces.mcp.runtime import TEXT_LIMIT, McpRuntime
    from jutul_agent.trace.schema import TOOL_RESULT

    fake_kernel._answers["big"] = "x" * (3 * TEXT_LIMIT)
    runtime = McpRuntime(_launch(tmp_path))
    await runtime.start()
    try:
        outcome = await runtime.call("run_julia", {"code": "big"})
        session = runtime.core.session
        results = [e for e in session.trace.events_after(0) if e.kind == TOOL_RESULT]
        saved = list((session.output_dir / "tool-results").glob("*.txt"))
    finally:
        await runtime.aclose()

    assert len(outcome.content) < TEXT_LIMIT
    assert "x" * (3 * TEXT_LIMIT) in results[-1].payload["content"]
    assert len(saved) == 1 and "x" * (3 * TEXT_LIMIT) in saved[0].read_text(encoding="utf-8")


def test_artifact_links_point_into_the_viewer(tmp_path) -> None:
    from jutul_agent.interfaces.mcp.runtime import artifact_links

    payloads = [
        {
            "path": "artifacts/plots/pressure.png",
            "kind": "plot",
            "caption": "Pressure",
            "live_url": "/live/s1/viz/pressure",
            "size_px": [900, 600],
        },
        {"path": "artifacts/report.html", "kind": "report", "mime": "text/html"},
        {"path": "artifacts/plots/snap.png", "kind": "snapshot", "mime": "image/png"},
    ]
    lines = artifact_links(
        payloads, session_id="s1", base_url="http://127.0.0.1:5555", output_dir=tmp_path
    )
    text = "\n".join(lines)
    popout = "http://127.0.0.1:5555/popout/s1?route=pressure&w=900&h=600"
    assert f"Pressure (interactive): {popout}" in text
    assert "http://127.0.0.1:5555/sessions/s1/artifacts/report.html" in text
    assert "http://127.0.0.1:5555/sessions/s1/artifacts/plots/snap.png" in text
    assert str(tmp_path / "artifacts" / "plots" / "pressure.png") in text

    # Without a viewer, the saved files are all there is to point at.
    bare = artifact_links(payloads, session_id="s1", base_url=None, output_dir=tmp_path)
    assert all("http" not in line for line in bare)
    assert str(tmp_path / "artifacts" / "report.html") in "\n".join(bare)


async def test_the_viewer_serves_this_session_to_its_own_origin_only(fake_kernel, tmp_path) -> None:
    from jutul_agent.interfaces.mcp.runtime import McpRuntime

    runtime = McpRuntime(_launch(tmp_path, viewer=True))
    await runtime.start()
    try:
        core = await runtime.ready()
        plots = core.session.output_dir / "artifacts" / "plots"
        plots.mkdir(parents=True, exist_ok=True)
        (plots / "p.png").write_bytes(b"png")
        base = runtime.viewer.base_url
        path = f"/sessions/{runtime.session_id}/artifacts/plots/p.png"
        async with httpx.AsyncClient() as http:
            served = await http.get(base + path)
            rebound = await http.get(base + path, headers={"Host": "attacker.example"})
            guessed = await http.get(base + "/sessions/2026-01-01-0000/artifacts/plots/p.png")
        assert served.status_code == 200 and served.content == b"png"
        assert rebound.status_code == 403
        assert guessed.status_code == 404
    finally:
        await runtime.aclose()
    assert runtime.viewer is None


def test_the_session_id_is_dated_and_unguessable() -> None:
    from jutul_agent.interfaces.mcp.runtime import capability_session_id
    from jutul_agent.paths import is_dated_session_id, session_output_dir
    from jutul_agent.session import default_session_id

    first, second = capability_session_id(), capability_session_id()
    assert first != second
    assert first[:10] == default_session_id()[:10]  # same date stamp as any session
    assert len(first) - len(default_session_id()) == 24  # 96 random bits
    # Laid out like any dated session: its output folder is named by the id.
    assert is_dated_session_id(first)
    assert session_output_dir(first).name == first


def test_status_line_is_the_last_line_that_says_something() -> None:
    from jutul_agent.interfaces.mcp.runtime import split_finished, status_line

    assert status_line("a\nb\n") == "b"
    assert status_line("10%\r50%\r\x1b[1m90%\x1b[0m\r") == "90%"
    assert status_line("Stats: 4 iterations\n╰────┴────╯\n") == "Stats: 4 iterations"
    assert status_line("\x1b]0;title\x07done\x1b[2A\n") == "done"
    assert status_line("\n \n") == ""
    assert status_line("x" * 300, limit=10) == "x" * 9 + "…"

    assert split_finished("a\nb") == ("a\n", "b")
    assert split_finished("10%\r") == ("10%\r", "")
    assert split_finished("x\x1b[") == ("", "x\x1b[")


async def test_progress_shows_only_finished_lines() -> None:
    """Output arrives in chunks that can end mid-line or mid-escape; the relay
    holds those back rather than show a cut-off line."""
    from jutul_agent.interfaces.mcp.server import ProgressRelay

    sent: list[str | None] = []

    async def _report(progress: float, total: float | None, message: str | None) -> None:
        sent.append(message)

    relay = ProgressRelay(_report, interval=0.01)
    try:
        for chunk in [
            "Stats: 8 iterations in 10.",
            "59 s\nnext \x1b",
            "[32mstep\x1b[0m",
            "\r╰──╯\n",
        ]:
            relay.write(chunk)
            await asyncio.sleep(0.05)
    finally:
        relay.close()
    assert sent == ["Stats: 8 iterations in 10.59 s", "next step"]


def test_image_blocks_become_mcp_images() -> None:
    from jutul_agent.interfaces.mcp.server import content_blocks

    png = base64.b64encode(b"png").decode()
    blocks = content_blocks(
        [
            {"type": "text", "text": "Saved plot"},
            {"type": "image", "base64": png, "mime_type": "image/png"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{png}"}},
        ]
    )
    assert isinstance(blocks[0], types.TextContent) and blocks[0].text == "Saved plot"
    assert isinstance(blocks[1], types.ImageContent) and blocks[1].data == png
    assert isinstance(blocks[2], types.ImageContent) and blocks[2].mime_type == "image/jpeg"


def test_instructions_point_at_the_simulator_skills(tmp_path) -> None:
    from jutul_agent.interfaces.mcp.runtime import McpLaunch
    from jutul_agent.interfaces.mcp.server import server_instructions
    from jutul_agent.simulators import registry

    launch = McpLaunch(simulator=registry.get("jutuldarcy"), workspace=tmp_path)
    text = server_instructions(launch)
    assert "JutulDarcy" in text and str(tmp_path) in text
    guides = [line for line in text.splitlines() if line.endswith("SKILL.md)")]
    assert guides, text
    for line in guides:
        assert Path(line.rsplit("(", 1)[1].rstrip(")")).is_file()


def test_the_mcp_command_drops_provider_keys(monkeypatch) -> None:
    import os

    from jutul_agent.interfaces.cli.mcp import _drop_provider_keys

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("UNRELATED_SETTING", "kept")
    _drop_provider_keys()
    assert "OPENAI_API_KEY" not in os.environ and "ANTHROPIC_API_KEY" not in os.environ
    assert os.environ["UNRELATED_SETTING"] == "kept"


def test_the_mcp_command_prepares_only_the_env_it_owns(tmp_path) -> None:
    from jutul_agent.interfaces.cli.mcp import build_launch, build_parser
    from jutul_agent.paths import set_workspace_root

    set_workspace_root(tmp_path)
    own = build_launch(build_parser().parse_args(["--sim", "jutuldarcy"]))
    assert own is not None and own.prepare_env and own.viewer
    assert own.simulator.name == "jutuldarcy" and own.workspace == tmp_path

    # A project passed in is prepared already, as for the other interfaces.
    given = build_launch(
        build_parser().parse_args(["--julia-project", str(tmp_path / "env"), "--no-viewer"])
    )
    assert given is not None and not given.prepare_env and not given.viewer
    assert given.julia_project == tmp_path / "env"


async def test_the_command_serves_mcp_on_stdio(tmp_path) -> None:
    """The real command, launched the way a client launches it.

    The handshake must succeed before Julia is up (stdout carries only the
    protocol), and a session that cannot start is an error the client's model can
    read, not a server that dies. Julia is kept off ``PATH`` so the start fails.
    """
    import os
    import sys

    from mcp import StdioServerParameters

    workspace = tmp_path / "ws"
    workspace.mkdir()
    keep = ("HOME", "LANG", "SYSTEMROOT", "TEMP", "TMP", "USERPROFILE")
    env = {key: value for key, value in os.environ.items() if key in keep}
    env["PATH"] = str(Path(sys.executable).parent)
    env["JUTUL_AGENT_NO_UPDATE_CHECK"] = "1"
    entry = "import sys; from jutul_agent.interfaces.cli.main import main; sys.exit(main())"
    args = ["-c", entry, "mcp", "--sim", "jutuldarcy", "--workspace", str(workspace)]
    args += ["--state-home", str(tmp_path / "state"), "--no-viewer"]
    params = StdioServerParameters(command=sys.executable, args=args, env=env)

    with anyio.fail_after(60):
        async with Client(params) as client:
            assert "JutulDarcy" in (client.instructions or "")
            result = await client.call_tool("run_julia", {"code": "1 + 1"})

    assert result.is_error
    assert "could not start" in _text(result) and "Julia" in _text(result)
