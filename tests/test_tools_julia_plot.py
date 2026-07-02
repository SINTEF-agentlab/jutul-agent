"""Tests for the plot_julia tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jutul_agent.agent.plot_julia import make_plot_julia_tool, make_recapture_tool
from jutul_agent.julia.session import EvalResult
from jutul_agent.lab.fakes import FakeJulia, make_fake_adapter
from jutul_agent.session import Session
from jutul_agent.trace import TraceLog


def _contains_key(obj: Any, key: str) -> bool:
    """Recursively check whether ``key`` appears anywhere in a JSON schema."""
    if isinstance(obj, dict):
        return key in obj or any(_contains_key(v, key) for v in obj.values())
    if isinstance(obj, list):
        return any(_contains_key(v, key) for v in obj)
    return False


def test_plot_tool_schemas_are_provider_safe(tmp_path: Path) -> None:
    """Array params must use ``items``, not ``prefixItems``.

    A ``tuple[int, int]`` annotation emits JSON-Schema ``prefixItems`` with no
    ``items`` field, which Gemini (and strict OpenAI schema validation) reject
    with "properties[size].items: missing field". ``list[int]`` emits ``items``.
    """
    julia = FakeJulia()
    session = _session(tmp_path, julia)
    for tool in (make_plot_julia_tool(session), make_recapture_tool(session)):
        schema = tool.args_schema.model_json_schema()
        assert not _contains_key(schema, "prefixItems"), (
            f"{tool.name} schema uses prefixItems (Gemini-incompatible): {schema}"
        )


async def _invoke(tool, args: dict, *, tool_call_id: str) -> Any:
    msg = await tool.ainvoke(
        {
            "type": "tool_call",
            "name": "plot_julia",
            "id": tool_call_id,
            "args": args,
        }
    )
    return getattr(msg, "content", msg)


async def _plot_call(tool, args: dict, *, tool_call_id: str) -> str:
    return str(await _invoke(tool, args, tool_call_id=tool_call_id))


def _write_artifact(path: str) -> None:
    """Write a real (tiny but valid) PNG so the view/downscale path works."""
    from PIL import Image

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (10, 20, 30)).save(p, format="PNG")


def _extract_path(code: str) -> str | None:
    for line in code.splitlines():
        if "path = raw" in line:
            start = line.index('raw"') + 4
            end = line.rindex('"')
            return line[start:end]
    return None


def _make_plot_eval_handler(
    written: list[str],
    *,
    accept_figure: bool = True,
    gl_ok: bool = True,
    seen: list[str] | None = None,
):
    async def fake_eval(code: str) -> EvalResult:
        if seen is not None:
            seen.append(code)
        stripped = code.strip()
        if stripped == "using GLMakie":  # the only backend the tool loads
            if gl_ok:
                return EvalResult(output="")
            return EvalResult(output="", error="libGL error: failed to load driver")
        if stripped == "using JutulAgent":  # loads the precompiled capture helpers
            return EvalResult(output="")
        if "JutulAgentPlots.capture" in code:
            if not accept_figure:
                return EvalResult(
                    output="",
                    error="plot_julia: the code did not produce a Makie figure. Return a Figure.",
                )
            path = _extract_path(code)
            if path:
                written.append(path)
                _write_artifact(path)
            return EvalResult(output="")
        if "JutulAgentPlots.recapture" in code:
            path = _extract_path(code)
            if path:
                written.append(path)
                _write_artifact(path)
            return EvalResult(output="")
        return EvalResult(output="")

    return fake_eval


def _session(tmp_path: Path, julia: FakeJulia, *, open_windows: bool = False) -> Session:
    return Session.create(
        julia=julia,
        state_root=tmp_path,
        simulator=make_fake_adapter(tmp_path),
        open_windows=open_windows,
    )


async def test_plot_julia_records_artifact_and_writes_file(tmp_path: Path) -> None:
    written: list[str] = []
    julia = FakeJulia(eval_handler=_make_plot_eval_handler(written))
    plot_session = _session(tmp_path, julia)

    tool = make_plot_julia_tool(plot_session)
    result = await _plot_call(
        tool,
        {"code": "plot_reservoir(model)", "caption": "reservoir"},
        tool_call_id="call_plot_default",
    )

    assert "saved plot to" in result
    assert "/artifacts/plot-" in result
    assert written
    assert Path(written[0]).exists()

    log = TraceLog(plot_session.state_dir / "trace.sqlite")
    try:
        artifact = next(ev for ev in log.iter_events() if ev.kind == "artifact")
        assert artifact.payload["mime"] == "image/png"
        assert artifact.payload["caption"] == "reservoir"
        assert artifact.payload["path"].startswith("artifacts/plot-")
        assert artifact.payload["source_code"] == "plot_reservoir(model)"
        assert artifact.payload["format"] == "png"
    finally:
        log.close()


async def test_plot_julia_slot_overwrites_path(tmp_path: Path) -> None:
    written: list[str] = []
    julia = FakeJulia(eval_handler=_make_plot_eval_handler(written))
    plot_session = _session(tmp_path, julia)

    tool = make_plot_julia_tool(plot_session)
    result = await _plot_call(
        tool,
        {"code": "lines(1:2)", "slot": "comparison"},
        tool_call_id="call_plot_slot",
    )

    assert "/artifacts/comparison.png" in result
    assert "slot=comparison" in result

    log = TraceLog(plot_session.state_dir / "trace.sqlite")
    try:
        artifact = next(ev for ev in log.iter_events() if ev.kind == "artifact")
        assert artifact.payload["slot"] == "comparison"
        assert artifact.payload["path"] == "artifacts/comparison.png"
    finally:
        log.close()


async def test_plot_julia_records_tool_call_id(tmp_path: Path) -> None:
    written: list[str] = []
    julia = FakeJulia(eval_handler=_make_plot_eval_handler(written))
    plot_session = _session(tmp_path, julia)

    tool = make_plot_julia_tool(plot_session)
    await _plot_call(
        tool,
        {"code": "lines(1:2)", "caption": "rates"},
        tool_call_id="call_plot_99",
    )

    log = TraceLog(plot_session.state_dir / "trace.sqlite")
    try:
        artifact = next(ev for ev in log.iter_events() if ev.kind == "artifact")
        assert artifact.payload["tool_call_id"] == "call_plot_99"
    finally:
        log.close()


async def test_plot_julia_size_px_in_artifact(tmp_path: Path) -> None:
    written: list[str] = []
    julia = FakeJulia(eval_handler=_make_plot_eval_handler(written))
    plot_session = _session(tmp_path, julia)

    tool = make_plot_julia_tool(plot_session)
    await _plot_call(
        tool,
        {"code": "lines(1:2)", "size": (640, 480)},
        tool_call_id="call_plot_size",
    )

    log = TraceLog(plot_session.state_dir / "trace.sqlite")
    try:
        artifact = next(ev for ev in log.iter_events() if ev.kind == "artifact")
        assert artifact.payload["size_px"] == [640, 480]
    finally:
        log.close()


async def test_plot_julia_not_a_figure_error(tmp_path: Path) -> None:
    julia = FakeJulia(eval_handler=_make_plot_eval_handler([], accept_figure=False))
    plot_session = _session(tmp_path, julia)

    tool = make_plot_julia_tool(plot_session)
    result = await _plot_call(tool, {"code": "42"}, tool_call_id="call_plot_bad")

    assert result.startswith("ERROR:")
    assert "Figure" in result


async def test_plot_julia_glmakie_load_failure_is_actionable(tmp_path: Path) -> None:
    # GLMakie is the only backend; if it can't load, the tool says so clearly.
    julia = FakeJulia(eval_handler=_make_plot_eval_handler([], gl_ok=False))
    plot_session = _session(tmp_path, julia)

    tool = make_plot_julia_tool(plot_session)
    result = await _plot_call(tool, {"code": "lines(1:10)"}, tool_call_id="call_plot_no_gl")

    assert result.startswith("ERROR:")
    assert "GLMakie" in result
    assert "xvfb" in result  # points at the headless-Linux fix


async def test_plot_julia_view_returns_image_block(tmp_path: Path) -> None:
    written: list[str] = []
    julia = FakeJulia(eval_handler=_make_plot_eval_handler(written))
    plot_session = _session(tmp_path, julia)

    tool = make_plot_julia_tool(plot_session)
    content = await _invoke(
        tool,
        {"code": "plot_well_results(wd)", "view": True},
        tool_call_id="call_plot_view",
    )

    # view returns standardized content blocks: a text summary + an image.
    assert isinstance(content, list)
    types = [b.get("type") for b in content]
    assert "text" in types
    assert "image" in types
    image = next(b for b in content if b["type"] == "image")
    assert image["mime_type"] == "image/png"
    assert image["base64"]  # non-empty base64 payload


async def test_plot_julia_opens_window_when_session_can(tmp_path: Path) -> None:
    written: list[str] = []
    seen: list[str] = []
    julia = FakeJulia(eval_handler=_make_plot_eval_handler(written, seen=seen))
    plot_session = _session(tmp_path, julia, open_windows=True)

    tool = make_plot_julia_tool(plot_session)
    result = await _plot_call(
        tool,
        {"code": "plot_reservoir(model)", "slot": "reservoir"},
        tool_call_id="call_plot_window",
    )

    assert "opened window" in result
    render = next(c for c in seen if "JutulAgentPlots.capture" in c)
    assert "GLMakie.activate!(visible = true)" in render
    assert "open_window = true" in render
    assert 'window_key = raw"reservoir"' in render


async def test_plot_julia_window_false_suppresses(tmp_path: Path) -> None:
    written: list[str] = []
    seen: list[str] = []
    julia = FakeJulia(eval_handler=_make_plot_eval_handler(written, seen=seen))
    plot_session = _session(tmp_path, julia, open_windows=True)

    tool = make_plot_julia_tool(plot_session)
    result = await _plot_call(
        tool, {"code": "plot_reservoir(model)", "window": False}, tool_call_id="call_no_win"
    )
    assert "opened window" not in result
    render = next(c for c in seen if "JutulAgentPlots.capture" in c)
    assert "open_window = false" in render


async def test_plot_julia_no_window_when_headless(tmp_path: Path) -> None:
    written: list[str] = []
    seen: list[str] = []
    julia = FakeJulia(eval_handler=_make_plot_eval_handler(written, seen=seen))
    # Headless session (open_windows defaults False): window=True must not open one.
    plot_session = _session(tmp_path, julia)

    tool = make_plot_julia_tool(plot_session)
    result = await _plot_call(tool, {"code": "plot_reservoir(model)"}, tool_call_id="call_headless")
    assert "opened window" not in result
    render = next(c for c in seen if "JutulAgentPlots.capture" in c)
    assert "GLMakie.activate!(visible = false)" in render


async def test_recapture_returns_image_by_default(tmp_path: Path) -> None:
    written: list[str] = []
    seen: list[str] = []
    julia = FakeJulia(eval_handler=_make_plot_eval_handler(written, seen=seen))
    plot_session = _session(tmp_path, julia, open_windows=True)

    from jutul_agent.agent.plot_julia import make_recapture_tool

    tool = make_recapture_tool(plot_session)
    msg = await tool.ainvoke(
        {"type": "tool_call", "name": "recapture_plot", "id": "c_recap", "args": {}}
    )
    content = getattr(msg, "content", msg)
    # view defaults True -> text + image content blocks of the current window view.
    assert isinstance(content, list)
    assert "image" in [b.get("type") for b in content]
    assert written and "recapture-" in written[0] and written[0].endswith(".png")
    # It re-renders a stored window figure, not arbitrary code.
    assert any("JutulAgentPlots.recapture" in c for c in seen)


async def test_recapture_targets_window_slot(tmp_path: Path) -> None:
    seen: list[str] = []
    julia = FakeJulia(eval_handler=_make_plot_eval_handler([], seen=seen))
    plot_session = _session(tmp_path, julia, open_windows=True)
    from jutul_agent.agent.plot_julia import make_recapture_tool

    tool = make_recapture_tool(plot_session)
    await tool.ainvoke(
        {
            "type": "tool_call",
            "name": "recapture_plot",
            "id": "c_recap_slot",
            "args": {"slot": "reservoir", "view": False},
        }
    )
    call = next(c for c in seen if "JutulAgentPlots.recapture" in c)
    assert 'key = raw"reservoir"' in call


async def test_close_plots(tmp_path: Path) -> None:
    seen: list[str] = []
    julia = FakeJulia(eval_handler=_make_plot_eval_handler([], seen=seen))
    plot_session = _session(tmp_path, julia, open_windows=True)
    from jutul_agent.agent.plot_julia import make_close_plots_tool

    tool = make_close_plots_tool(plot_session)
    one = await tool.ainvoke(
        {"type": "tool_call", "name": "close_plots", "id": "cl1", "args": {"slot": "reservoir"}}
    )
    assert "reservoir" in str(getattr(one, "content", one))
    assert any('close_windows(raw"reservoir")' in c for c in seen)
    allw = await tool.ainvoke({"type": "tool_call", "name": "close_plots", "id": "cl2", "args": {}})
    assert "all" in str(getattr(allw, "content", allw))


async def test_recapture_no_window_errors(tmp_path: Path) -> None:
    async def fake_eval(code: str) -> EvalResult:
        if "JutulAgentPlots.recapture" in code:
            return EvalResult(output="", error="recapture: no interactive window is open.")
        return EvalResult(output="")

    julia = FakeJulia(eval_handler=fake_eval)
    plot_session = _session(tmp_path, julia)
    from jutul_agent.agent.plot_julia import make_recapture_tool

    tool = make_recapture_tool(plot_session)
    msg = await tool.ainvoke(
        {"type": "tool_call", "name": "recapture_plot", "id": "c_recap2", "args": {"view": False}}
    )
    result = str(getattr(msg, "content", msg))
    assert result.startswith("ERROR")
    assert "no interactive window" in result


def test_all_envs_include_glmakie_and_graphmakie() -> None:
    from jutul_agent.simulators.battmo import BATTMO
    from jutul_agent.simulators.jutuldarcy import JUTULDARCY

    for adapter in (BATTMO, JUTULDARCY):
        text = (adapter.julia_env_template_path / "Project.toml").read_text(encoding="utf-8")
        assert "GLMakie" in text
        assert "CairoMakie" in text
        assert "GraphMakie" in text
