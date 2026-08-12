"""End-to-end tests for the server: REST lifecycle and the turn WebSocket.

The agent and Julia kernel are fakes (see ``jutul_agent.lab.fakes``), so a turn runs through the
real ``TurnRunner`` and wire protocol without a provider API or a Julia process.
A test ``SessionManager`` is injected with a host factory that wraps those fakes.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from jutul_agent.agent.capabilities import Capability
from jutul_agent.interfaces.server.app import artifact_wire_events, create_app
from jutul_agent.interfaces.server.manager import SessionBusyError, SessionManager
from jutul_agent.lab.fakes import (
    FakeJulia,
    echo_agent,
    interrupt_agent,
    make_fake_adapter,
    streaming_agent,
)
from jutul_agent.session import Session, default_session_id
from jutul_agent.session_host import SessionHost


def _manager(
    agent_factory: Callable[[], Any], tmp_path: Path, *, max_live: int = 16
) -> SessionManager:
    """A manager whose sessions wrap a fresh fake agent and a real (fake-kernel) Session."""

    async def host_factory(
        *, sim, model, approval_mode, workspace, resume, session_id, extensions=()
    ) -> SessionHost:
        adapter = make_fake_adapter(tmp_path)
        sid = session_id or default_session_id()
        session = Session.create(
            julia=FakeJulia(), simulator=adapter, session_id=sid, state_root=tmp_path
        )
        return SessionHost(session=session, agent=agent_factory())

    return SessionManager(host_factory=host_factory, max_live=max_live)


def _client(agent_factory: Callable[[], Any], tmp_path: Path) -> TestClient:
    return TestClient(create_app(_manager(agent_factory, tmp_path)))


def _drain_turn(ws: Any) -> list[dict]:
    """Read events until the turn ends or pauses for approval."""
    events: list[dict] = []
    while True:
        event = ws.receive_json()
        events.append(event)
        if event["type"] in {"turn_end", "interrupt"}:
            return events


@pytest.fixture(autouse=True)
def _provider_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Placeholder keys so the create-session credential guard doesn't depend on the
    host environment. These tests drive fake agents, never a real provider; a session
    creates with the default model (openai), so without a key the guard would 400
    here on CI (no keys) but pass on a dev box. Tests of the missing-key path clear
    the relevant key themselves.
    """
    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.setenv(var, "test-key")


def test_models_endpoint(tmp_path: Path) -> None:
    from jutul_agent.interfaces.server import protocol

    with _client(echo_agent, tmp_path) as client:
        body = client.get("/models").json()
    assert "default" in body
    assert isinstance(body["providers"], list)
    # The version a third-party front end negotiates against is announced here.
    assert body["protocol"] == protocol.PROTOCOL_VERSION


def test_models_endpoint_reports_the_launch_default_model(tmp_path: Path) -> None:
    # /models reports the server's actual default so the UI seeds the right model: the
    # launch --model when set, else the catalog default. Otherwise the UI would show
    # and resume onto the catalog default even when the server runs a different model.
    from jutul_agent.models import DEFAULT_MODEL

    app = create_app(_manager(echo_agent, tmp_path), default_model="provider:custom")
    with TestClient(app) as c:
        assert c.get("/models").json()["default"] == "provider:custom"
    with TestClient(create_app(_manager(echo_agent, tmp_path))) as c:
        assert c.get("/models").json()["default"] == DEFAULT_MODEL


def test_credentials_endpoint_lists_providers(tmp_path: Path) -> None:
    with _client(echo_agent, tmp_path) as client:
        body = client.get("/credentials").json()
    assert "path" in body
    providers = {p["provider"]: p for p in body["providers"]}
    assert {"openai", "anthropic", "google_genai"} <= set(providers)
    # The placeholder keys read as set; only masked previews cross the wire.
    assert providers["openai"]["is_set"] and providers["openai"]["masked"]


def test_post_credentials_saves_and_is_reflected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with _client(echo_agent, tmp_path) as client:
        before = {p["provider"]: p for p in client.get("/credentials").json()["providers"]}
        assert not before["anthropic"]["is_set"]
        ok = client.post("/credentials", json={"provider": "anthropic", "value": "sk-newkey-1234"})
        assert ok.status_code == 200 and ok.json()["env_var"] == "ANTHROPIC_API_KEY"
        after = {p["provider"]: p for p in client.get("/credentials").json()["providers"]}
        assert after["anthropic"]["is_set"] and after["anthropic"]["source"] == "file"
        # Unknown providers are rejected, not written.
        assert (
            client.post("/credentials", json={"provider": "bogus", "value": "x"}).status_code == 400
        )


def test_create_session_requires_credential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A new session on a model whose key is missing is refused with a structured error
    # (the UI shows a key prompt on it), before any kernel is stood up.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app = create_app(
        _manager(echo_agent, tmp_path), default_sim="demo", default_model="openai:gpt-5.4"
    )
    with TestClient(app) as client:
        resp = client.post("/sessions", json={})
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        assert detail["error"] == "credential_required" and detail["env_var"] == "OPENAI_API_KEY"
        # A keyless local model still creates fine.
        assert client.post("/sessions", json={"model": "ollama:qwen3"}).status_code == 200


def test_set_model_prompts_for_missing_key_over_ws(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with _client(echo_agent, tmp_path) as client:
        sid = client.post("/sessions", json={"sim": "demo", "model": "ollama:qwen3"}).json()[
            "session_id"
        ]
        with client.websocket_connect(f"/sessions/{sid}/stream") as ws:
            ws.send_json({"type": "command", "command": "set_model", "arg": "openai:gpt-5.4"})
            msg = ws.receive_json()
    assert msg["type"] == "credential_required" and msg["env_var"] == "OPENAI_API_KEY"


def test_simulators_endpoint(tmp_path: Path) -> None:
    with _client(echo_agent, tmp_path) as client:
        body = client.get("/simulators").json()
    assert "jutuldarcy" in body["simulators"]
    # Each simulator carries its display name and starter prompts for a welcome screen.
    detail = body["details"]["jutuldarcy"]
    assert detail["display_name"] == "JutulDarcy"
    assert detail["examples"] and all(isinstance(e, str) for e in detail["examples"])


def test_bound_simulator_uses_one_and_rejects_mismatch(tmp_path: Path) -> None:
    # A server bound to a simulator (the `web` case) uses it for every session
    # and refuses a request for a different one — one folder, one simulator, no
    # in-place switching. Without a bound simulator the caller's choice is honoured.
    manager = _manager(echo_agent, tmp_path)
    with TestClient(create_app(manager, default_sim="jutuldarcy")) as client:
        assert client.get("/simulators").json()["default"] == "jutuldarcy"
        assert client.post("/sessions", json={"sim": "jutuldarcy"}).status_code == 200
        assert client.post("/sessions", json={}).status_code == 200  # omitted → the bound one
        mismatch = client.post("/sessions", json={"sim": "battmo"})
        assert mismatch.status_code == 409 and "bound" in mismatch.json()["detail"]


def test_default_approval_mode_applies_when_request_omits_it(tmp_path: Path) -> None:
    # `jutul-agent web --approval-mode auto` sets the default policy for new sessions;
    # a per-request approval_mode still wins (and the UI can change it live).
    seen: list[str | None] = []

    async def host_factory(
        *, sim, model, approval_mode, workspace, resume, session_id, extensions=()
    ) -> SessionHost:
        seen.append(approval_mode)
        adapter = make_fake_adapter(tmp_path)
        sid = session_id or default_session_id()
        session = Session.create(
            julia=FakeJulia(), simulator=adapter, session_id=sid, state_root=tmp_path
        )
        return SessionHost(session=session, agent=echo_agent())

    manager = SessionManager(host_factory=host_factory, max_live=16)
    with TestClient(create_app(manager, default_approval_mode="auto")) as client:
        assert client.post("/sessions", json={"sim": "demo"}).status_code == 200
        assert (
            client.post("/sessions", json={"sim": "demo", "approval_mode": "ask"}).status_code
            == 200
        )
    assert seen == ["auto", "ask"]  # omitted → the launch default; explicit → the request


def test_default_model_applies_when_request_omits_it(tmp_path: Path) -> None:
    # `jutul-agent web --model <m>` sets the default model for new sessions; a
    # per-request model (the UI's picker) still wins.
    seen: list[str | None] = []

    async def host_factory(
        *, sim, model, approval_mode, workspace, resume, session_id, extensions=()
    ) -> SessionHost:
        seen.append(model)
        adapter = make_fake_adapter(tmp_path)
        sid = session_id or default_session_id()
        session = Session.create(
            julia=FakeJulia(), simulator=adapter, session_id=sid, state_root=tmp_path
        )
        return SessionHost(session=session, agent=echo_agent())

    manager = SessionManager(host_factory=host_factory, max_live=16)
    with TestClient(create_app(manager, default_model="prov:base")) as client:
        assert client.post("/sessions", json={"sim": "demo"}).status_code == 200
        assert (
            client.post("/sessions", json={"sim": "demo", "model": "prov:override"}).status_code
            == 200
        )
    assert seen == ["prov:base", "prov:override"]


async def test_launch_defaults_reach_session_host_start(monkeypatch, tmp_path: Path) -> None:
    # The folder-fixed launch knobs (--threads/--add-dir/--ephemeral-memory/
    # --julia-project) ride in the default host factory's closure and are handed
    # to SessionHost.start for every session, so the server honours them.
    from jutul_agent.interfaces.server import manager as manager_mod
    from jutul_agent.interfaces.server.manager import SessionLaunchDefaults, make_host_factory

    captured: dict[str, Any] = {}

    async def fake_start(**kwargs: Any) -> str:
        captured.update(kwargs)
        return "host"

    monkeypatch.setattr(manager_mod.SessionHost, "start", staticmethod(fake_start))
    monkeypatch.setattr("jutul_agent.simulators.registry.get", lambda name: f"adapter:{name}")

    factory = make_host_factory(
        SessionLaunchDefaults(
            julia_project=tmp_path / "proj",
            threads="3",
            add_dirs=(tmp_path / "extra",),
            ephemeral_memory=True,
        )
    )
    await factory(
        sim="demo",
        model=None,
        approval_mode=None,
        workspace=None,
        resume=False,
        session_id=None,
        extensions=(),
    )
    assert captured["threads"] == "3"
    assert captured["ephemeral_memory"] is True
    assert captured["add_dirs"] == (tmp_path / "extra",)
    assert captured["julia_project"] == tmp_path / "proj"


async def test_start_wires_surface_and_capability_dependencies(monkeypatch, tmp_path: Path) -> None:
    """``SessionHost.start`` threads ``surface`` into the ``Session`` it creates,
    and merges discovered + passed-in capabilities' dependency paths into the
    ``prepare_workspace_env`` call — both added in this branch alongside the
    capability-dependencies feature.
    """
    import jutul_agent.juliakernel as juliakernel_mod
    from jutul_agent.agent import builder as builder_mod
    from jutul_agent.agent import capabilities as capabilities_mod
    from jutul_agent.julia import requirements as requirements_mod
    from jutul_agent.simulators import env_setup as env_setup_mod
    from jutul_agent.simulators import warmup as warmup_mod

    workspace = tmp_path / "ws"
    workspace.mkdir()
    adapter = make_fake_adapter(tmp_path)

    monkeypatch.setattr(requirements_mod, "require_julia", lambda *a, **kw: None)
    monkeypatch.setattr(juliakernel_mod, "JuliaKernel", lambda config: FakeJulia())
    monkeypatch.setattr(warmup_mod, "start_warmup", lambda *a, **kw: None)
    monkeypatch.setattr(builder_mod, "resolve_package_sources", lambda project: [])

    discovered_dep = tmp_path / "DiscoveredCap" / "Project.toml"
    passed_dep = tmp_path / "PassedCap" / "Project.toml"
    discovered_cap = Capability(name="discovered", dependencies=(discovered_dep,))
    passed_cap = Capability(name="passed", dependencies=(passed_dep,))
    monkeypatch.setattr(capabilities_mod, "discover_extensions", lambda: [discovered_cap])

    prepare_calls: dict[str, Any] = {}

    def _fake_prepare(simulator, *, workspace, julia_project, sim_name, dependencies):
        prepare_calls["dependencies"] = dependencies

    monkeypatch.setattr(env_setup_mod, "prepare_workspace_env", _fake_prepare)

    build_calls: dict[str, Any] = {}

    def _fake_build_agent(session, **kwargs):
        build_calls["surface"] = kwargs.get("surface")
        build_calls["extensions"] = kwargs.get("extensions")
        return object(), object()

    monkeypatch.setattr(builder_mod, "build_agent", _fake_build_agent)

    host = await SessionHost.start(
        simulator=adapter,
        workspace=workspace,
        state_root=tmp_path,
        surface="tui",
        extensions=[passed_cap],
    )
    try:
        assert prepare_calls["dependencies"] == [discovered_dep, passed_dep]
        assert host.session.surface == "tui"
        assert build_calls["surface"] == "tui"
        assert discovered_cap in build_calls["extensions"]
        assert passed_cap in build_calls["extensions"]
    finally:
        await host.aclose()


def test_unbound_server_requires_a_simulator(tmp_path: Path) -> None:
    # No bound simulator (tests / a future multi-folder launcher): the caller must
    # name one, and an omitted simulator is a clear 400 rather than a crash.
    with _client(echo_agent, tmp_path) as client:
        assert client.post("/sessions", json={}).status_code == 400


def test_manager_caps_live_sessions(tmp_path: Path) -> None:
    # Each live session pins a Julia kernel, so the manager keeps only the most
    # recent ``max_live`` and closes the rest (they stay resumable on disk).
    manager = _manager(echo_agent, tmp_path, max_live=2)
    with TestClient(create_app(manager)) as client:
        ids = [
            client.post("/sessions", json={"sim": "demo"}).json()["session_id"] for _ in range(3)
        ]
        live = client.get("/sessions").json()["sessions"]
    assert set(live) == {ids[1], ids[2]}  # the oldest was evicted


async def test_eviction_skips_attached_sessions(tmp_path: Path) -> None:
    # A session a client is connected to must not be torn down mid-turn: eviction
    # skips attached hosts and takes the oldest idle one instead, even when the
    # attached host is the oldest.
    manager = _manager(echo_agent, tmp_path, max_live=2)
    a = await manager.create(sim="demo")
    a.attach()  # a live connection now holds the oldest session
    b = await manager.create(sim="demo")
    c = await manager.create(sim="demo")  # over cap → evict the oldest *idle* host (b)
    live = set(manager.list_ids())
    assert a.session_id in live  # attached, so kept despite being oldest
    assert c.session_id in live
    assert b.session_id not in live


def test_second_connection_to_a_session_is_refused(tmp_path: Path) -> None:
    # Two live sockets on one session would run turns on one kernel concurrently;
    # the second is refused, and once the first closes a new one can attach.
    with _client(echo_agent, tmp_path) as client:
        sid = client.post("/sessions", json={"sim": "demo"}).json()["session_id"]
        with (
            client.websocket_connect(f"/sessions/{sid}/stream"),  # first holds the session
            client.websocket_connect(f"/sessions/{sid}/stream") as ws2,
        ):
            refused = ws2.receive_json()
        assert refused["type"] == "error" and "another window" in refused["message"]
        # The first socket has closed, so a fresh connection attaches and runs.
        with client.websocket_connect(f"/sessions/{sid}/stream") as ws3:
            ws3.send_json({"type": "prompt", "text": "hi"})
            assert _drain_turn(ws3)[-1]["type"] == "turn_end"


def test_web_ui_is_served(tmp_path: Path) -> None:
    with _client(echo_agent, tmp_path) as client:
        root = client.get("/")
    assert root.status_code == 200
    assert "jutul-agent" in root.text


def test_create_list_delete(tmp_path: Path) -> None:
    with _client(echo_agent, tmp_path) as client:
        sid = client.post("/sessions", json={"sim": "demo"}).json()["session_id"]
        assert sid in client.get("/sessions").json()["sessions"]
        assert client.delete(f"/sessions/{sid}").json() == {"ok": True}
        assert client.get("/sessions").json()["sessions"] == []
        assert client.delete(f"/sessions/{sid}").status_code == 404


async def test_manager_aclose_isolates_a_failing_teardown(tmp_path: Path) -> None:
    # Server shutdown closes every live session; one session whose teardown raises
    # (e.g. a kernel already gone) must not abort the loop and orphan the rest.
    class _Host:
        def __init__(self, sid: str, boom: bool) -> None:
            self.session_id = sid
            self._boom = boom
            self.closed = False

        @property
        def attached(self) -> bool:
            return False

        async def aclose(self) -> None:
            self.closed = True
            if self._boom:
                raise RuntimeError("kernel already gone")

    manager = SessionManager()
    first, second = _Host("a", boom=True), _Host("b", boom=False)
    manager._hosts["a"] = first  # type: ignore[assignment]
    manager._hosts["b"] = second  # type: ignore[assignment]

    await manager.aclose()  # must not raise despite first's teardown error

    assert first.closed and second.closed  # both were torn down
    assert manager.list_ids() == []  # and both removed from the registry


def test_reattach_leaves_the_live_host_as_is(tmp_path: Path) -> None:
    # Reattaching to a live idle session must NOT reconfigure it from the resume
    # request. The live host is authoritative: an in-session model/approval change
    # already updated it through the set_model/set_approval command (see
    # test_command_reconfigures_session). The UI always reports the default model by
    # name and never sends the approval mode, so honouring the request here would
    # revert the user's in-session choice on every reconnect.
    async def host_factory(
        *, sim, model, approval_mode, workspace, resume, session_id, extensions=()
    ) -> SessionHost:
        adapter = make_fake_adapter(tmp_path)
        sid = session_id or default_session_id()
        session = Session.create(
            julia=FakeJulia(), simulator=adapter, session_id=sid, state_root=tmp_path
        )
        return SessionHost(session=session, agent=echo_agent(), model=model)

    manager = SessionManager(host_factory=host_factory, max_live=16)
    with TestClient(create_app(manager)) as client:
        sid = client.post("/sessions", json={"sim": "demo", "model": "prov:b"}).json()["session_id"]
        host = manager.get(sid)
        calls: list[dict] = []
        host.reconfigure = lambda **kw: calls.append(kw)  # type: ignore[method-assign]

        # Even a resume request naming a different model leaves the live host untouched.
        resp = client.post(f"/sessions/{sid}/resume", json={"sim": "demo", "model": "prov:c"})
        assert resp.json()["kernel_restarted"] is False
        assert calls == []


def test_reconfigure_keeps_state_consistent_when_build_fails(tmp_path: Path, monkeypatch) -> None:
    # If build_agent rejects a value (e.g. an unknown approval mode), reconfigure
    # must leave the host reporting its previous, still-running model/approval —
    # not the rejected value, which a same-value reattach would later read as
    # "unchanged" and silently skip, stranding the desync.
    import pytest

    adapter = make_fake_adapter(tmp_path)
    session = Session.create(
        julia=FakeJulia(), simulator=adapter, session_id=default_session_id(), state_root=tmp_path
    )
    host = SessionHost(
        session=session, agent="AGENT-0", backend=None, model="prov:a", approval_mode="ask"
    )

    def boom(*args, **kwargs):
        raise ValueError("unknown approval mode 'bogus'")

    monkeypatch.setattr("jutul_agent.agent.builder.build_agent", boom)
    with pytest.raises(ValueError):
        host.reconfigure(approval_mode="bogus")

    assert host.approval_mode == "ask"  # not poisoned to the rejected value
    assert host.model == "prov:a"
    assert host.agent == "AGENT-0"  # the previous agent still runs


def test_ws_streaming_prompt(tmp_path: Path) -> None:
    with _client(streaming_agent, tmp_path) as client:
        sid = client.post("/sessions", json={"sim": "demo"}).json()["session_id"]
        with client.websocket_connect(f"/sessions/{sid}/stream") as ws:
            ws.send_json({"type": "prompt", "text": "hi"})
            events = _drain_turn(ws)
    texts = [e["text"] for e in events if e["type"] == "text"]
    assert "".join(texts) == "Hello world"
    assert events[-1]["type"] == "turn_end"


class _FakeWS:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send_json(self, message: dict) -> None:
        self.sent.append(message)


async def test_stream_delta_renders_terminal_output_and_trailing_flushes() -> None:
    # Streamed tool output is rendered the way the TUI renders it: the accumulated
    # raw stream is replayed through the terminal emulator (a progress bar's carriage
    # returns collapse to one line, not a stack) and the client replaces the card.
    # Throttling is leading+trailing, so a delta within the interval still gets shown
    # by a trailing flush rather than lingering until the next event.
    from jutul_agent.interfaces.server.app import _StreamState

    ws = _FakeWS()
    st = _StreamState(ws, None)  # type: ignore[arg-type]
    cid = "c1"

    await st._on_tool_delta(
        cid, {"type": "tool", "event": "delta", "tool_call_id": cid, "content": "step 1\rstep 2"}
    )
    assert ws.sent[-1]["replace"] is True
    assert ws.sent[-1]["content"] == "step 2"  # the carriage return overwrote, not stacked

    sent_so_far = len(ws.sent)
    await st._on_tool_delta(
        cid, {"type": "tool", "event": "delta", "tool_call_id": cid, "content": "\nstep 3"}
    )
    assert len(ws.sent) == sent_so_far  # within the throttle: not sent yet...
    assert cid in st._tool_flush  # ...but a trailing flush is scheduled
    await st._tool_flush[cid]  # which sends the combined, rendered state
    assert "step 3" in ws.sent[-1]["content"]

    st._end_tool_stream(cid)  # the final result ends the stream and clears state
    assert cid not in st._tool_streams and cid not in st._tool_flush


def test_ws_echo_prompt(tmp_path: Path) -> None:
    with _client(echo_agent, tmp_path) as client:
        sid = client.post("/sessions", json={"sim": "demo"}).json()["session_id"]
        with client.websocket_connect(f"/sessions/{sid}/stream") as ws:
            ws.send_json({"type": "prompt", "text": "hi"})
            events = _drain_turn(ws)
    assert any(e["type"] == "text" and "Echo:" in e["text"] for e in events)
    assert events[-1]["type"] == "turn_end"


def test_ws_interrupt_then_approve(tmp_path: Path) -> None:
    with _client(interrupt_agent, tmp_path) as client:
        sid = client.post("/sessions", json={"sim": "demo"}).json()["session_id"]
        with client.websocket_connect(f"/sessions/{sid}/stream") as ws:
            ws.send_json({"type": "prompt", "text": "please run"})
            paused = _drain_turn(ws)
            interrupt = paused[-1]
            assert interrupt["type"] == "interrupt"
            assert interrupt["actions"][0]["name"] == "execute"
            assert set(interrupt["allowed_decisions"]) == {"approve", "reject", "respond"}

            ws.send_json({"type": "decision", "decision": "approve"})
            resumed = _drain_turn(ws)
    assert any(e["type"] == "text" and "approval handled" in e["text"] for e in resumed)
    assert resumed[-1]["type"] == "turn_end"


def test_ws_reconnect_resurfaces_a_pending_approval(tmp_path: Path) -> None:
    # If the connection drops while an approval is pending, a fresh connection to the
    # same live session re-surfaces the interrupt (read from the persisted graph state)
    # so the user can still answer it, instead of the paused turn being orphaned.
    with _client(interrupt_agent, tmp_path) as client:
        sid = client.post("/sessions", json={"sim": "demo"}).json()["session_id"]
        with client.websocket_connect(f"/sessions/{sid}/stream") as ws:
            ws.send_json({"type": "prompt", "text": "please run"})
            assert _drain_turn(ws)[-1]["type"] == "interrupt"
        # ws closed without deciding (a dropped connection); the session stays live.
        with client.websocket_connect(f"/sessions/{sid}/stream") as ws2:
            resurfaced = ws2.receive_json()  # re-sent on attach, before any prompt
            assert resurfaced["type"] == "interrupt"
            assert resurfaced["actions"][0]["name"] == "execute"
            ws2.send_json({"type": "decision", "decision": "approve"})
            resumed = _drain_turn(ws2)
    assert any(e["type"] == "text" and "approval handled" in e["text"] for e in resumed)
    assert resumed[-1]["type"] == "turn_end"


def test_ws_always_allow_auto_approves_future_interrupts(tmp_path: Path) -> None:
    # "Always allow" approves now and remembers the category, so a later interrupt
    # of the same kind auto-approves without asking the user again (like the TUI).
    def agent() -> Any:
        return interrupt_agent(tool_name="write_file", allowed_decisions=["approve", "reject"])

    with _client(agent, tmp_path) as client:
        sid = client.post("/sessions", json={"sim": "demo"}).json()["session_id"]
        with client.websocket_connect(f"/sessions/{sid}/stream") as ws:
            ws.send_json({"type": "prompt", "text": "edit a file"})
            interrupt = _drain_turn(ws)[-1]
            assert interrupt["type"] == "interrupt"
            assert interrupt["allowlist"] == ["file_edits"]  # offered to the front end

            ws.send_json({"type": "decision", "decision": "always_allow"})
            assert _drain_turn(ws)[-1]["type"] == "turn_end"

            # A second file-edit interrupt is now resolved automatically: the client
            # sees the turn complete and is never asked to approve again.
            ws.send_json({"type": "prompt", "text": "edit another file"})
            second = _drain_turn(ws)
            assert second[-1]["type"] == "turn_end"
            assert not any(e["type"] == "interrupt" for e in second)


def test_ws_unknown_session(tmp_path: Path) -> None:
    with (
        _client(echo_agent, tmp_path) as client,
        client.websocket_connect("/sessions/nope/stream") as ws,
    ):
        assert ws.receive_json() == {"type": "error", "message": "no such session"}


def test_ws_decision_without_pending_is_error(tmp_path: Path) -> None:
    with _client(echo_agent, tmp_path) as client:
        sid = client.post("/sessions", json={"sim": "demo"}).json()["session_id"]
        with client.websocket_connect(f"/sessions/{sid}/stream") as ws:
            ws.send_json({"type": "decision", "decision": "approve"})
            event = ws.receive_json()
    assert event["type"] == "error"
    assert "no approval" in event["message"]


async def test_side_outputs_forward_artifacts_and_ui_commands(tmp_path: Path) -> None:
    """Both directions of the user-interface channel meet at the trace.

    A capability's tool drives the front end by appending a ``ui`` event; the
    flush turns it into a ``ui`` wire message. The kinds are read back from the
    trace, so a writer and this reader agreeing is the whole contract.
    """
    from jutul_agent.interfaces.server.app import _StreamState
    from jutul_agent.trace import schema

    session = Session.create(
        julia=FakeJulia(), state_root=tmp_path, simulator=make_fake_adapter(tmp_path)
    )
    host = SessionHost(session=session, agent=None)
    ws = _FakeWS()
    st = _StreamState(ws, host)  # type: ignore[arg-type]

    session.trace.append(schema.UI_COMMAND, {"action": "set_param", "payload": {"p": 2}})
    session.trace.append(
        schema.ARTIFACT,
        schema.artifact_payload(path="artifacts/p.png", mime="image/png", caption="fig"),
    )
    await st._flush_side_outputs()
    assert [m["type"] for m in ws.sent] == ["ui", "artifact"]
    assert ws.sent[0] == {"type": "ui", "action": "set_param", "payload": {"p": 2}}

    # The high-water mark means a second flush re-sends nothing.
    await st._flush_side_outputs()
    assert len(ws.sent) == 2


def test_artifact_wire_events_png_and_html() -> None:
    payloads = [
        {"path": "artifacts/plot.png", "mime": "image/png", "caption": "fig"},
        {
            "path": "artifacts/scene.html",
            "mime": "text/html",
            "caption": "interactive",
            "kind": "plot",
            "poster": "artifacts/scene.png",
            "slot": "scene",
        },
        {
            "path": "artifacts/report.html",
            "mime": "text/html",
            "caption": "Run report",
            "kind": "report",
            "slot": "report",
        },
    ]
    events = artifact_wire_events(payloads, "sid")
    assert events[0] == {
        "type": "artifact",
        "url": "/sessions/sid/artifacts/plot.png",
        "mime": "image/png",
        "caption": "fig",
        "slot": None,
        "format": None,
    }
    # An interactive plot becomes a viz carrying its kind, slot, and poster URL.
    assert events[1] == {
        "type": "viz",
        "url": "/sessions/sid/artifacts/scene.html",
        "title": "interactive",
        "kind": "plot",
        "poster": "/sessions/sid/artifacts/scene.png",
        "slot": "scene",
        "live": False,
        "record": None,
        "width": None,
        "height": None,
    }
    # A written report is a viz too, of kind "report" and with no poster.
    assert events[2] == {
        "type": "viz",
        "url": "/sessions/sid/artifacts/report.html",
        "title": "Run report",
        "kind": "report",
        "poster": None,
        "slot": "report",
        "live": False,
        "record": None,
        "width": None,
        "height": None,
    }


def test_artifact_wire_events_live_plot_uses_live_url() -> None:
    # A live-served plot carries a live_url (the session's Bonito server); the viz
    # points there instead of the static export, but the poster is still served
    # as a session artifact.
    # A live plot's durable record is the PNG (mime image/png); the live_url is
    # where the figure is actually served, so the viz points there, not at the PNG.
    payloads = [
        {
            "path": "artifacts/reservoir.png",
            "mime": "image/png",
            "caption": "Reservoir",
            "kind": "plot",
            "poster": "artifacts/reservoir.png",
            "slot": "reservoir",
            "live_url": "http://127.0.0.1:9123/viz/reservoir",
            "source_code": "plot_reservoir(model, states)",
            "size_px": [1600, 900],
        },
    ]
    (event,) = artifact_wire_events(payloads, "sid")
    assert event == {
        "type": "viz",
        "url": "http://127.0.0.1:9123/viz/reservoir",
        "title": "Reservoir",
        "kind": "plot",
        "poster": "/sessions/sid/artifacts/reservoir.png",
        "slot": "reservoir",
        "live": True,
        # Recorded code makes the plot replayable; the record names its artifact.
        "record": "artifacts/reservoir.png",
        "width": 1600,
        "height": 900,
    }


def test_artifact_wire_events_replay_falls_back_to_poster() -> None:
    # On resume the Julia process (and its Bonito server) is gone, so the recorded
    # live_url is dead. Replaying with live=False must point the viz at the still-on
    # -disk PNG poster, not the dead URL, so the plot is still viewable (static).
    payloads = [
        {
            "path": "artifacts/reservoir.png",
            "mime": "image/png",
            "caption": "Reservoir",
            "kind": "plot",
            "poster": "artifacts/reservoir.png",
            "slot": "reservoir",
            "live_url": "http://127.0.0.1:9123/viz/reservoir",
            "source_code": "plot_reservoir(model, states)",
            "size_px": [1600, 900],
        },
    ]
    (event,) = artifact_wire_events(payloads, "sid", live=False)
    assert event == {
        "type": "viz",
        "url": "/sessions/sid/artifacts/reservoir.png",  # the poster, not the dead live_url
        "title": "Reservoir",
        "kind": "plot",
        "poster": "/sessions/sid/artifacts/reservoir.png",
        "slot": "reservoir",
        "live": False,
        # The record survives the downgrade: it is what the regenerate button
        # sends back to revive the view in a fresh kernel.
        "record": "artifacts/reservoir.png",
        "width": 1600,
        "height": 900,
    }


def _plot_eval_handler(code: str):
    """Answers for the Julia the replot path evaluates, keyed on its markers."""
    from jutul_agent.agent import plot_julia_src as jl
    from jutul_agent.julia.session import EvalResult

    if "Bonito.Server" in code and "__JUTUL_WEB_PORT__" in code:
        return EvalResult(output="__JUTUL_WEB_PORT__=9123")
    if "__JUTUL_WEB_FIGS__[" in code:
        return EvalResult(output=f"{jl.FIG_SIZE_MARKER}=1600x900")
    if jl.SCREEN_PREFERENCE_MARKER in code:
        return EvalResult(output=f"{jl.SCREEN_PREFERENCE_MARKER}=ok")
    if jl.PICK_GUARD_MARKER in code:
        return EvalResult(output=f"{jl.PICK_GUARD_MARKER}=ok")
    return EvalResult(output="")


def _plot_state(tmp_path: Path):
    """A stream state over a session whose trace holds one replayable plot."""
    from jutul_agent.interfaces.server.app import _StreamState
    from jutul_agent.trace import schema

    session = Session.create(
        julia=FakeJulia(eval_handler=_plot_eval_handler),
        state_root=tmp_path,
        simulator=make_fake_adapter(tmp_path),
    )
    session.trace.append(
        schema.ARTIFACT,
        schema.artifact_payload(
            path="artifacts/res.png",
            mime="image/png",
            caption="Reservoir",
            format="png",
            kind="plot",
            poster="artifacts/res.png",
            slot="res",
            live_url="/live/x/viz/res",
            source_code="plot_reservoir(model, states)",
        ),
    )
    # The poster the replayed CairoMakie.save would write (a fake kernel writes
    # nothing); without it the finalize honestly records the HTML fallback.
    poster = session.output_dir / "artifacts" / "res.png"
    poster.parent.mkdir(parents=True, exist_ok=True)
    poster.write_bytes(b"png")
    host = SessionHost(session=session, agent=None)
    ws = _FakeWS()
    return _StreamState(ws, host), ws, session  # type: ignore[arg-type]


async def test_replot_revive_reserves_route_and_revives_view(tmp_path: Path) -> None:
    # The regenerate button: the recorded code re-runs on the plot's own route,
    # the re-finalized artifact flushes as a fresh live viz, and the turn ends.
    st, ws, session = _plot_state(tmp_path)
    await st._replot("artifacts/res.png", "revive")
    viz = next(m for m in ws.sent if m["type"] == "viz")
    assert viz["live"] is True
    assert viz["url"] == f"/live/{session.session_id}/viz/res"
    assert viz["slot"] == "res"
    assert viz["record"] == "artifacts/res.png"
    assert (viz["width"], viz["height"]) == (1600, 900)
    assert ws.sent[-1]["type"] == "turn_end"
    # The replay served the figure and saved a fresh poster (the poster block runs).
    assert any("__JUTUL_WEB_FIGS__[" in c and "CairoMakie.save" in c for c in session.julia.calls)


async def test_replot_unknown_record_is_an_error(tmp_path: Path) -> None:
    st, ws, _session = _plot_state(tmp_path)
    await st._replot("artifacts/nope.png", "revive")
    assert ws.sent[0]["type"] == "error"
    assert "no recorded code" in ws.sent[0]["message"]


async def test_replot_popout_serves_independent_route_and_close_releases_it(
    tmp_path: Path,
) -> None:
    # A popout replays the code into its own route (an independent figure), skips
    # the poster (the record already exists), and the close releases that route.
    st, ws, session = _plot_state(tmp_path)
    await st._replot("artifacts/res.png", "popout")
    (ready,) = [m for m in ws.sent if m["type"] == "popout_ready"]
    assert ready["error"] is None
    # The popup gets the scale-to-fit wrapper, sized from the figure's echo
    # (1600x900 from the fake kernel), hosting the independent live route.
    assert ready["url"] == f"/popout/{session.session_id}?route=res--pop1&w=1600&h=900"
    assert not any(m["type"] == "viz" for m in ws.sent)  # no artifact re-recorded
    serve = next(c for c in session.julia.calls if "__JUTUL_WEB_FIGS__[" in c)
    assert "res--pop1" in serve
    assert "CairoMakie.save" not in serve  # poster skipped for the ephemeral view

    await st._close_popout(ready["url"])
    close = session.julia.calls[-1]
    assert "delete_route!" in close and "res--pop1" in close
    # A second close for the same URL is a no-op (the route is no longer ours).
    calls = len(session.julia.calls)
    await st._close_popout(ready["url"])
    assert len(session.julia.calls) == calls


async def test_replot_with_target_size_refits_the_figure(tmp_path: Path) -> None:
    # A regenerate that carries the stage's measured size re-fits the replayed
    # figure to it; junk sizes are dropped rather than resized to.
    st, ws, session = _plot_state(tmp_path)
    await st._start_replot(
        {"type": "replot", "record": "artifacts/res.png", "width": 900, "height": 850}
    )
    await st._turn
    serve = next(c for c in session.julia.calls if "__JUTUL_WEB_FIGS__[" in c)
    assert "resize!(_fig, 900, 850)" in serve
    # The viz reports the echoed size, not the request: the figure's own layout
    # has the last word on what a resize actually produced.
    viz = next(m for m in ws.sent if m["type"] == "viz")
    assert (viz["width"], viz["height"]) == (1600, 900)


async def test_replot_with_junk_size_replays_unresized(tmp_path: Path) -> None:
    st, _ws, session = _plot_state(tmp_path)
    await st._start_replot(
        {"type": "replot", "record": "artifacts/res.png", "width": "huge", "height": -3}
    )
    await st._turn
    serve = next(c for c in session.julia.calls if "__JUTUL_WEB_FIGS__[" in c)
    assert "resize!(_fig" not in serve


async def test_replot_without_size_fits_the_session_panel_hint(tmp_path: Path) -> None:
    # A plain regenerate re-fits to the panel the same way a fresh plot does:
    # the client refreshes the hint just before sending the replot.
    st, _ws, session = _plot_state(tmp_path)
    session.web_canvas_hint = (900, 1100)
    await st._start_replot({"type": "replot", "record": "artifacts/res.png"})
    await st._turn
    serve = next(c for c in session.julia.calls if "__JUTUL_WEB_FIGS__[" in c)
    assert "local _pw, _ph = 900, 1100" in serve


async def test_refit_resizes_the_live_figure_in_place(tmp_path: Path) -> None:
    # The canvas's grown-stage message: the kernel resize!-es the routed figure
    # (no code re-run), and junk or busy states drop the request silently — the
    # client's scaled presentation is the always-correct fallback.
    st, ws, session = _plot_state(tmp_path)
    await st.handle(
        {"type": "refit", "record": "artifacts/res.png", "width": 1200, "height": 1000}
    )
    call = session.julia.calls[-1]
    assert "resize!(_fig, 1200, 1000)" in call
    assert 'raw"/viz/res"' in call
    assert not any(m["type"] == "viz" for m in ws.sent)  # in place: no re-pin

    calls = len(session.julia.calls)
    await st.handle({"type": "refit", "record": "artifacts/res.png", "width": 9, "height": 9})
    await st.handle({"type": "refit", "record": "artifacts/nope.png", "width": 900, "height": 900})
    assert len(session.julia.calls) == calls  # junk size and unknown record: dropped


async def test_canvas_size_hint_lands_on_the_session(tmp_path: Path) -> None:
    # The ui_event hint doubles as live session state for plot_julia; junk
    # measurements never overwrite a good hint.
    st, _ws, session = _plot_state(tmp_path)
    await st.handle(
        {"type": "ui_event", "payload": {"kind": "canvas_size", "width": 810, "height": 930}}
    )
    assert session.web_canvas_hint == (810, 930)
    await st.handle({"type": "ui_event", "payload": {"kind": "canvas_size", "width": 5}})
    assert session.web_canvas_hint == (810, 930)
    await st.handle({"type": "ui_event", "payload": "not a dict"})
    assert session.web_canvas_hint == (810, 930)


async def test_replot_refused_while_busy(tmp_path: Path) -> None:
    st, ws, _session = _plot_state(tmp_path)

    async def _hang() -> None:
        await asyncio.sleep(30)

    st._turn = asyncio.create_task(_hang())
    try:
        await st._start_replot({"type": "replot", "record": "artifacts/res.png"})
        assert ws.sent[-1]["type"] == "error"
        await st._start_replot(
            {"type": "replot", "record": "artifacts/res.png", "target": "popout"}
        )
        assert ws.sent[-1]["type"] == "popout_ready"
        assert ws.sent[-1]["url"] is None
        assert "finish the current turn" in ws.sent[-1]["error"]
    finally:
        st._turn.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await st._turn


def test_command_reconfigures_session(tmp_path: Path) -> None:
    # A `command` message rebuilds the agent in place (model / approval policy).
    # reconfigure is stubbed here (the real one rebuilds a provider-backed agent);
    # a following unknown command, whose error we read, proves the first was
    # processed in order.
    manager = _manager(echo_agent, tmp_path)
    with TestClient(create_app(manager)) as client:
        sid = client.post("/sessions", json={"sim": "jutuldarcy"}).json()["session_id"]
        calls: list[dict] = []
        manager.get(sid).reconfigure = lambda **kw: calls.append(kw)  # type: ignore[method-assign]
        with client.websocket_connect(f"/sessions/{sid}/stream") as ws:
            ws.send_json({"type": "command", "command": "set_model", "arg": "anthropic:c"})
            ws.send_json({"type": "command", "command": "set_approval", "arg": "auto"})
            ws.send_json({"type": "command", "command": "bogus"})
            err = ws.receive_json()
    assert calls == [{"model": "anthropic:c"}, {"approval_mode": "auto"}]
    assert err["type"] == "error" and "bogus" in err["message"]


def test_command_compact_and_add_dir(tmp_path: Path) -> None:
    # /compact and /add-dir reply with a `notice`; the host methods are stubbed
    # (the real ones summarize via a model / mutate the backend).
    manager = _manager(echo_agent, tmp_path)
    with TestClient(create_app(manager)) as client:
        sid = client.post("/sessions", json={"sim": "jutuldarcy"}).json()["session_id"]
        host = manager.get(sid)
        host.add_dir = lambda arg: f"added:{arg}"  # type: ignore[method-assign]

        async def fake_compact() -> tuple[str, None]:
            return "compacted:ok", None

        host.compact = fake_compact  # type: ignore[method-assign]
        with client.websocket_connect(f"/sessions/{sid}/stream") as ws:
            ws.send_json({"type": "command", "command": "add_dir", "arg": "/data"})
            n1 = ws.receive_json()
            ws.send_json({"type": "command", "command": "compact"})
            n2 = ws.receive_json()
    assert n1 == {"type": "notice", "text": "added:/data"}
    assert n2 == {"type": "notice", "text": "compacted:ok"}


def test_transcript_and_memory_endpoints(tmp_path: Path) -> None:
    with _client(echo_agent, tmp_path) as client:
        sid = client.post("/sessions", json={"sim": "jutuldarcy"}).json()["session_id"]
        html = client.get(f"/sessions/{sid}/transcript")
        assert html.status_code == 200 and "text/html" in html.headers["content-type"]
        md = client.get(f"/sessions/{sid}/transcript", params={"format": "md"})
        assert md.status_code == 200 and "markdown" in md.headers["content-type"]
        mem = client.get(f"/sessions/{sid}/memory")
        assert mem.status_code == 200 and "Memory" in mem.text
        assert client.get("/sessions/nope/transcript").status_code == 404


def test_context_endpoint_renders_panel(tmp_path: Path) -> None:
    # /context renders the same panel as the TUI. It reads the trace from a fresh
    # connection because the endpoint runs in a threadpool and the session's own
    # SQLite connection is bound to the thread it was created on — a regression
    # guard for the cross-thread error that returned a 500.
    from jutul_agent.trace import TraceLog

    manager = _manager(echo_agent, tmp_path)
    with TestClient(create_app(manager)) as client:
        sid = client.post("/sessions", json={"sim": "jutuldarcy"}).json()["session_id"]
        state_dir = manager.get(sid).session.state_dir  # type: ignore[union-attr]
        with TraceLog(state_dir / "trace.sqlite") as log:  # own connection (test thread)
            log.append("model_usage", {"input_tokens": 1200, "output_tokens": 80})
        resp = client.get(f"/sessions/{sid}/context")
        assert resp.status_code == 200
        assert resp.json()["markdown"].strip()
        assert client.get("/sessions/nope/context").status_code == 404


def test_history_endpoint_shape(tmp_path: Path) -> None:
    with _client(echo_agent, tmp_path) as client:
        body = client.get("/sessions/history").json()
    assert isinstance(body.get("sessions"), list)
    for s in body["sessions"]:
        assert {"id", "title", "started", "sim"} <= set(s)


def test_history_derives_title_when_none_stored(tmp_path: Path) -> None:
    # A real conversation whose title file never landed (its titling never persisted)
    # must still appear in history, derived from its first prompt, instead of
    # vanishing. An abandoned new-chat (no prompt) is hidden.
    from jutul_agent.session import sessions_root
    from jutul_agent.trace import TraceLog

    root = sessions_root()  # workspace_state_dir()/sessions under the autouse fixture
    root.mkdir(parents=True, exist_ok=True)

    convo = root / "2026-06-20-1835-aaaa"
    convo.mkdir()
    with TraceLog(convo / "trace.sqlite") as log:
        log.append("session_start", {"session_id": convo.name, "simulator": "battmo"})
        log.append("message_user", {"content": "Discharge the chen cell and plot the voltage"})

    empty = root / "2026-06-20-1840-bbbb"
    empty.mkdir()
    with TraceLog(empty / "trace.sqlite") as log:
        log.append("session_start", {"session_id": empty.name, "simulator": "battmo"})

    with _client(echo_agent, tmp_path) as client:
        sessions = client.get("/sessions/history").json()["sessions"]

    by_id = {s["id"]: s for s in sessions}
    assert "2026-06-20-1835-aaaa" in by_id  # the real conversation shows...
    shown = by_id["2026-06-20-1835-aaaa"]
    assert shown["title"].startswith("Discharge the chen cell")  # ...with a derived title
    assert shown["sim"] == "battmo"
    assert "2026-06-20-1840-bbbb" not in by_id  # the abandoned new-chat stays hidden


def test_history_caps_by_last_use_not_creation(tmp_path: Path) -> None:
    # The `limit` cap is applied AFTER sorting by last use, not before: an old-created
    # but recently-used session must survive the cut, and a newest-created but
    # least-recently-used one is dropped.
    from jutul_agent.session import sessions_root
    from jutul_agent.trace import TraceLog

    root = sessions_root()
    root.mkdir(parents=True, exist_ok=True)
    # Appended oldest-id last, so id order (newest first) is A,B,C while activity
    # order (latest last) is A,B,C — i.e. the oldest id, C, was used most recently.
    for sid in ("2026-06-22-2300-aaaa", "2026-06-22-1200-bbbb", "2026-06-22-0100-cccc"):
        d = root / sid
        d.mkdir()
        with TraceLog(d / "trace.sqlite") as log:
            log.append("session_start", {"session_id": sid, "simulator": "battmo"})
            log.append("message_user", {"content": f"work {sid[-4:]}"})

    with _client(echo_agent, tmp_path) as client:
        ids = [s["id"] for s in client.get("/sessions/history?limit=2").json()["sessions"]]

    assert ids[0] == "2026-06-22-0100-cccc"  # oldest id, used most recently → top
    assert "2026-06-22-2300-aaaa" not in ids  # newest id, used first → cut by the limit
    assert len(ids) == 2


def test_delete_refuses_a_session_open_in_a_connection(tmp_path: Path) -> None:
    # Deleting a session a connection is driving would tear its kernel down under a
    # running turn; refuse with 409. Once the socket closes (detaches), delete works.
    manager = _manager(echo_agent, tmp_path)
    with TestClient(create_app(manager)) as client:
        sid = client.post("/sessions", json={"sim": "demo"}).json()["session_id"]
        with client.websocket_connect(f"/sessions/{sid}/stream"):
            assert client.delete(f"/sessions/{sid}").status_code == 409
        assert client.delete(f"/sessions/{sid}").status_code == 200


async def test_end_all_tool_streams_clears_state_and_cancels_flushes() -> None:
    # A cancelled/errored turn leaves tool-stream state behind; _end_all_tool_streams
    # must drop every per-call dict and cancel any pending trailing-flush task.
    import asyncio
    import contextlib

    from jutul_agent.interfaces.server.app import _StreamState

    st = _StreamState(_FakeWS(), None)  # type: ignore[arg-type]
    st._tool_streams["c1"] = "partial output"
    st._tool_render_at["c1"] = 1.0
    st._tool_delta_wire["c1"] = {"type": "tool", "event": "delta", "tool_call_id": "c1"}
    task = asyncio.create_task(asyncio.sleep(100))
    st._tool_flush["c1"] = task

    st._end_all_tool_streams()

    assert st._tool_streams == {} and st._tool_flush == {} and st._tool_delta_wire == {}
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert task.cancelled()


def test_history_ordered_by_last_use_not_creation(tmp_path: Path) -> None:
    # History is ordered by when each session was last used (its latest event), not
    # by when it was created: a session with an older id but more recent activity
    # comes first. This is the "order I last used it" the user expects.
    from jutul_agent.session import sessions_root
    from jutul_agent.trace import TraceLog

    root = sessions_root()
    root.mkdir(parents=True, exist_ok=True)

    newer_id = root / "2026-06-22-2000-newr"  # newer id, but used earlier
    newer_id.mkdir()
    with TraceLog(newer_id / "trace.sqlite") as log:
        log.append("session_start", {"session_id": newer_id.name, "simulator": "battmo"})
        log.append("message_user", {"content": "older activity"})

    older_id = root / "2026-06-22-1000-oldr"  # older id, but used just now
    older_id.mkdir()
    with TraceLog(older_id / "trace.sqlite") as log:  # appended after → later timestamp
        log.append("session_start", {"session_id": older_id.name, "simulator": "battmo"})
        log.append("message_user", {"content": "most recent activity"})

    with _client(echo_agent, tmp_path) as client:
        sessions = client.get("/sessions/history").json()["sessions"]

    order = [s["id"] for s in sessions if s["id"].endswith(("-newr", "-oldr"))]
    assert order == ["2026-06-22-1000-oldr", "2026-06-22-2000-newr"]


def test_session_overview_reads_sim_first_prompt_and_last_activity(tmp_path: Path) -> None:
    from jutul_agent.interfaces.server.app import _session_overview
    from jutul_agent.trace import TraceLog

    sd = tmp_path / "s1"
    sd.mkdir()
    with TraceLog(sd / "trace.sqlite") as log:
        log.append("session_start", {"session_id": "s1", "simulator": "jutuldarcy"})
        log.append("message_user", {"content": "build a 5-spot waterflood"})
        log.append("message_user", {"content": "now plot it"})  # only the first is used
    sim, first_prompt, last_active = _session_overview(sd)
    assert (sim, first_prompt) == ("jutuldarcy", "build a 5-spot waterflood")
    assert last_active and last_active >= "2026"  # the most recent event's timestamp

    empty = tmp_path / "s2"
    empty.mkdir()
    with TraceLog(empty / "trace.sqlite") as log:
        log.append("session_start", {"session_id": "s2", "simulator": "jutuldarcy"})
    sim, first_prompt, _ = _session_overview(empty)
    assert (sim, first_prompt) == ("jutuldarcy", None)


def test_messages_endpoint_replays_conversation(tmp_path: Path) -> None:
    from jutul_agent.trace import TraceLog

    manager = _manager(echo_agent, tmp_path)
    with TestClient(create_app(manager)) as client:
        sid = client.post("/sessions", json={"sim": "jutuldarcy"}).json()["session_id"]
        state_dir = manager.get(sid).session.state_dir  # type: ignore[union-attr]
        with TraceLog(state_dir / "trace.sqlite") as log:  # own connection (test thread)
            log.append("message_user", {"content": "set up a reservoir"})
            log.append("message_reasoning", {"content": "I'll build a small grid"})
            log.append(
                "tool_call",
                {"id": "c1", "name": "run_julia", "args": {"code": "1+1"}},
            )
            log.append(
                "tool_result",
                {"tool_call_id": "c1", "name": "run_julia", "content": "2", "status": "success"},
            )
            log.append("message_assistant", {"content": "done — here it is"})
        msgs = client.get(f"/sessions/{sid}/messages").json()["messages"]
    assert {"type": "user", "text": "set up a reservoir"} in msgs
    assert {"type": "reasoning", "text": "I'll build a small grid"} in msgs
    assert {"type": "assistant", "text": "done — here it is"} in msgs
    # A tool replays as a requested card followed by its finished result, so the
    # resumed chat shows the full tool card with its output (not just text).
    requested = next(m for m in msgs if m["type"] == "tool" and m["event"] == "requested")
    assert requested["tool_call_id"] == "c1" and requested["args"] == {"code": "1+1"}
    finished = next(m for m in msgs if m["type"] == "tool" and m["event"] == "finished")
    assert finished["tool_call_id"] == "c1" and finished["content"] == "2"


def test_messages_endpoint_finishes_resultless_tool_calls(tmp_path: Path) -> None:
    # Some tool calls never record a result (e.g. a write_todos that ends a turn).
    # Replay must still emit a terminal event so the card resolves instead of
    # spinning forever on resume.
    from jutul_agent.trace import TraceLog

    manager = _manager(echo_agent, tmp_path)
    with TestClient(create_app(manager)) as client:
        sid = client.post("/sessions", json={"sim": "jutuldarcy"}).json()["session_id"]
        state_dir = manager.get(sid).session.state_dir  # type: ignore[union-attr]
        with TraceLog(state_dir / "trace.sqlite") as log:
            log.append("tool_call", {"id": "t1", "name": "write_todos", "args": {"todos": []}})
            # deliberately no tool_result for t1
        msgs = client.get(f"/sessions/{sid}/messages").json()["messages"]
    t1 = [m for m in msgs if m["type"] == "tool" and m["tool_call_id"] == "t1"]
    assert [m["event"] for m in t1] == ["requested", "finished"]


def test_first_turn_generates_llm_title(tmp_path: Path, monkeypatch: Any) -> None:
    # After the first turn the server replaces the first-prompt title with a
    # content-aware one (best-effort) and nudges the front end to refresh history.
    from jutul_agent import session as session_mod
    from jutul_agent.agent import titling

    async def fake_title(model_id: Any, conversation: str) -> str:
        return "Reservoir Sweep Study"

    monkeypatch.setattr(titling, "generate_session_title", fake_title)

    with _client(echo_agent, tmp_path) as client:
        sid = client.post("/sessions", json={"sim": "jutuldarcy"}).json()["session_id"]
        with client.websocket_connect(f"/sessions/{sid}/stream") as ws:
            ws.send_json({"type": "prompt", "text": "set up a small reservoir"})
            events = _drain_turn(ws)
            renamed = ws.receive_json()  # the post-turn history-refresh signal
    assert events[-1]["type"] == "turn_end"
    assert renamed["type"] == "ui" and renamed["action"] == "history_changed"
    assert renamed["payload"]["title"] == "Reservoir Sweep Study"
    # The new title is persisted, so a history listing shows it.
    titles = [s.title for s in session_mod.list_sessions(state_root=tmp_path)]
    assert "Reservoir Sweep Study" in titles


def test_upload_writes_to_workspace(tmp_path: Path) -> None:
    manager = _manager(echo_agent, tmp_path)
    with TestClient(create_app(manager)) as client:
        sid = client.post("/sessions", json={"sim": "jutuldarcy"}).json()["session_id"]
        manager.get(sid).workspace = tmp_path  # type: ignore[union-attr]
        resp = client.post(
            f"/sessions/{sid}/upload",
            files={"file": ("my data.csv", b"a,b\n1,2\n", "text/csv")},
        )
    assert resp.status_code == 200
    assert resp.json()["path"] == "uploads/my_data.csv"  # basename + sanitized
    assert (tmp_path / "uploads" / "my_data.csv").read_bytes() == b"a,b\n1,2\n"


@pytest.mark.parametrize("agent_factory", [echo_agent])
def test_unknown_simulator_is_400(agent_factory: Callable[[], Any], tmp_path: Path) -> None:
    # The default manager (no injected factory) resolves the simulator registry,
    # so an unknown name is a client error rather than a server crash.
    with TestClient(create_app(SessionManager())) as client:
        resp = client.post("/sessions", json={"sim": "does-not-exist"})
    assert resp.status_code == 400


def test_resume_rejects_malformed_session_id(tmp_path: Path) -> None:
    # A client-supplied id that isn't the server-generated shape is refused before
    # any disk access, so it can't become a path traversal into mkdir.
    with _client(echo_agent, tmp_path) as client:
        resp = client.post("/sessions/foo%24bar/resume", json={})
    assert resp.status_code == 404


def test_messages_rejects_malformed_session_id(tmp_path: Path) -> None:
    with _client(echo_agent, tmp_path) as client:
        resp = client.get("/sessions/foo%24bar/messages")
    assert resp.status_code == 404


def test_resume_refuses_when_session_in_use(tmp_path: Path) -> None:
    # Re-resuming a session another connection holds would tear its live kernel
    # down; the server returns 409 instead.
    manager = _manager(echo_agent, tmp_path)
    with TestClient(create_app(manager)) as client:
        sid = client.post("/sessions", json={"sim": "jutuldarcy"}).json()["session_id"]
        manager.get(sid).attach()  # type: ignore[union-attr]  # simulate an active connection
        resp = client.post(f"/sessions/{sid}/resume", json={})
    assert resp.status_code == 409


async def test_acquire_attaches_promotes_and_refuses_a_second_connection(tmp_path: Path) -> None:
    # acquire() is the atomic claim a WebSocket makes: it attaches an idle live host
    # (so eviction, which skips attached hosts, can't tear it down mid-connect), and
    # returns None for a host that is missing or already attached elsewhere.
    manager = _manager(echo_agent, tmp_path)
    host = await manager.create(sim="demo")
    sid = host.session_id

    first = await manager.acquire(sid)
    assert first is host and host.attached
    assert await manager.acquire(sid) is None  # already attached: a second tab is refused
    assert await manager.acquire("no-such-session") is None  # not live

    host.detach()
    assert await manager.acquire(sid) is host  # reattachable once idle again


async def test_close_require_idle_refuses_an_attached_session(tmp_path: Path) -> None:
    # A delete must not tear a kernel down under a live connection: with require_idle
    # the check and the pop are one atomic step, raising rather than closing.
    manager = _manager(echo_agent, tmp_path)
    host = await manager.create(sim="demo")
    sid = host.session_id
    host.attach()
    with pytest.raises(SessionBusyError):
        await manager.close(sid, require_idle=True)
    assert manager.get(sid) is host  # still registered, untouched
    host.detach()
    assert await manager.close(sid, require_idle=True) is True


def test_respond_decision_without_message_sends_an_empty_message(tmp_path: Path) -> None:
    # langchain's HITL reads decision["message"] by subscript for a respond (unlike
    # reject), so the server must always include it. A respond with no text resolves
    # the turn and the resume decision carries message="" rather than omitting it.
    with _client(interrupt_agent, tmp_path) as client:
        sid = client.post("/sessions", json={"sim": "demo"}).json()["session_id"]
        with client.websocket_connect(f"/sessions/{sid}/stream") as ws:
            ws.send_json({"type": "prompt", "text": "please run"})
            assert _drain_turn(ws)[-1]["type"] == "interrupt"
            ws.send_json({"type": "decision", "decision": "respond"})  # no "message" field
            resumed = _drain_turn(ws)
            agent = client.app.state.manager.get(sid).agent  # type: ignore[attr-defined]
        decision = next(iter(agent.resume_inputs[-1].resume.values()))["decisions"][0]
    assert resumed[-1]["type"] == "turn_end"
    assert decision["type"] == "respond"
    assert decision["message"] == ""


def test_resume_reattaches_to_a_live_idle_session(tmp_path: Path) -> None:
    # Navigating back to a session that's still live (idle, not attached) must
    # reattach to the existing host rather than rebuild it, so its Julia REPL
    # state survives — and the response says the kernel was not restarted.
    manager = _manager(echo_agent, tmp_path)
    with TestClient(create_app(manager)) as client:
        sid = client.post("/sessions", json={"sim": "jutuldarcy"}).json()["session_id"]
        before = manager.get(sid)
        resp = client.post(f"/sessions/{sid}/resume", json={})
        assert resp.status_code == 200
        assert resp.json()["kernel_restarted"] is False
        assert manager.get(sid) is before  # the same live host, not a rebuilt one


async def _accept_recording_server(recorder: list[bytes]):
    """A bare TCP listener that records anything a client sends it."""

    async def handle(reader, writer):
        with contextlib.suppress(Exception):
            recorder.append(await asyncio.wait_for(reader.read(1), timeout=0.3))
        writer.close()

    return await asyncio.start_server(handle, "127.0.0.1", 0)


def test_readiness_probe_never_reaches_the_plot_server(tmp_path: Path) -> None:
    """A HEAD on a live plot route must not be forwarded to Bonito.

    The canvas polls this route to learn when the live server is up, retrying with
    backoff. Bonito renders the figure's app for anything that reaches its route --
    HEAD included -- and every render leaves a WGLMakie screen and a session
    registered on the figure that nothing removes, because no browser ever attaches
    to a probe. Makie resolves a scene's screen by taking the first match, so those
    corpses shadow the live view and picking (with the click-to-select built on it)
    stops working.

    Asserting the upstream received *no bytes* is the point: a forwarding proxy
    would send it a request line, which is exactly what allocates the screen.
    """

    manager = _manager(echo_agent, tmp_path)
    client = TestClient(create_app(manager))
    sid = client.post("/sessions", json={"sim": "demo", "model": "ollama:qwen3"}).json()[
        "session_id"
    ]

    # Nothing serving live plots for this session yet.
    assert client.head(f"/live/{sid}/viz/anything").status_code == 404

    received: list[bytes] = []

    async def run() -> tuple[int, list[bytes]]:
        server = await _accept_recording_server(received)
        port = server.sockets[0].getsockname()[1]
        manager.get(sid).session.web_plot_port = port
        resp = client.head(f"/live/{sid}/viz/plot")
        server.close()
        await server.wait_closed()
        return resp.status_code, received

    status, sent = asyncio.run(run())
    assert status == 200, "a reachable plot server must read as ready"
    assert sent == [] or sent == [b""], f"probe was forwarded upstream: {sent!r}"


def test_readiness_probe_reports_an_unreachable_plot_server(tmp_path: Path) -> None:
    """A recorded port with nothing behind it must not read as ready.

    Otherwise the canvas mounts a live frame against a dead port instead of waiting.
    """

    manager = _manager(echo_agent, tmp_path)
    client = TestClient(create_app(manager))
    sid = client.post("/sessions", json={"sim": "demo", "model": "ollama:qwen3"}).json()[
        "session_id"
    ]

    with socket.socket() as probe:  # a port nothing is listening on
        probe.bind(("127.0.0.1", 0))
        dead_port = probe.getsockname()[1]
    manager.get(sid).session.web_plot_port = dead_port

    assert client.head(f"/live/{sid}/viz/plot").status_code == 502


def test_set_model_refused_while_approval_pending(tmp_path: Path) -> None:
    """Switching models would rebuild the agent out from under a paused approval."""
    with _client(interrupt_agent, tmp_path) as client:
        sid = client.post("/sessions", json={"sim": "jutuldarcy"}).json()["session_id"]
        with client.websocket_connect(f"/sessions/{sid}/stream") as ws:
            ws.send_json({"type": "prompt", "text": "run it"})
            assert _drain_turn(ws)[-1]["type"] == "interrupt"
            ws.send_json({"type": "command", "command": "set_model", "arg": "openai:gpt-5.4-mini"})
            err = ws.receive_json()
    assert err["type"] == "error"
    assert "pending approval" in err["message"]


def test_set_approval_auto_resolves_a_pending_interrupt(tmp_path: Path) -> None:
    """Switching to auto mode resumes an approval that is now a foregone conclusion."""
    manager = _manager(interrupt_agent, tmp_path)
    with TestClient(create_app(manager)) as client:
        sid = client.post("/sessions", json={"sim": "jutuldarcy"}).json()["session_id"]
        host = manager.get(sid)

        def fake_reconfigure(**kwargs) -> None:  # the real one rebuilds via build_agent
            host._approval_mode = kwargs.get("approval_mode")

        host.reconfigure = fake_reconfigure  # type: ignore[method-assign]
        with client.websocket_connect(f"/sessions/{sid}/stream") as ws:
            ws.send_json({"type": "prompt", "text": "run it"})
            assert _drain_turn(ws)[-1]["type"] == "interrupt"
            ws.send_json({"type": "command", "command": "set_approval", "arg": "auto"})
            events = _drain_turn(ws)
    assert events[-1]["type"] == "turn_end"
    assert events[-1]["text"] == "approval handled"


def test_proxied_plot_socket_does_not_keepalive_the_plot_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The proxy must not hold the plot server to a pong deadline.

    The peer on this hop is the Julia process, whose plot-server handler tasks share
    a thread pool with whatever the kernel is computing: a long solve leaves them
    unscheduled for far longer than any sane keepalive allows. A ping therefore
    measures how busy Julia is rather than whether the connection is alive, and the
    client library's default is to close on the timeout -- which the browser answers
    by reconnecting, once per timeout, for as long as the solve runs.
    """

    import websockets

    dial_kwargs: list[dict[str, Any]] = []

    class _IdleUpstream:
        """Connects, then says nothing -- a plot server whose thread pool is busy."""

        async def __aenter__(self) -> Any:
            return self

        async def __aexit__(self, *exc: Any) -> bool:
            return False

        async def send(self, data: Any) -> None:
            pass

        async def close(self) -> None:
            pass

        def __aiter__(self) -> Any:
            return self

        async def __anext__(self) -> Any:
            await asyncio.sleep(3600)
            raise StopAsyncIteration

    def spy(url: str, **kwargs: Any) -> _IdleUpstream:
        dial_kwargs.append(kwargs)
        return _IdleUpstream()

    monkeypatch.setattr(websockets, "connect", spy)

    manager = _manager(echo_agent, tmp_path)
    client = TestClient(create_app(manager))
    sid = client.post("/sessions", json={"sim": "demo", "model": "ollama:qwen3"}).json()[
        "session_id"
    ]
    manager.get(sid).session.web_plot_port = 9999  # never dialled for real

    with client.websocket_connect(f"/live/{sid}/9e53-4780-81c6"):
        pass

    assert dial_kwargs, "the proxy never dialled the plot server"
    # Absent is not the same as off: leaving it out is what selects the 20s default.
    assert "ping_interval" in dial_kwargs[0], "keepalive left at the library default"
    assert dial_kwargs[0]["ping_interval"] is None, (
        "keepalive on this hop closes a working socket whenever Julia is busy"
    )
    # The dial must also be patient: the plot server shares Julia's interactive
    # thread with the eval loop, so a long eval delays the upstream handshake for
    # exactly as long as it runs. The library default (10s) drops a merely-busy
    # server; anything under a solve's timescale reintroduces that.
    assert dial_kwargs[0].get("open_timeout", 10) >= 60, (
        "upstream handshake patience must cover a long eval, not a network RTT"
    )


def test_dead_plot_server_refuses_the_handshake(tmp_path: Path) -> None:
    """A dial the proxy cannot complete must refuse the browser's handshake,
    never accept-then-close.

    The distinction drives the client's retry state machine: Bonito's reconnect
    budgets ~30s of backed-off attempts and then gives up, but a successful open
    resets that budget. Accept-then-close grants every attempt a fresh budget, so
    a tab whose session was evicted (or whose kernel died) reconnects forever. A
    refused handshake lets the client's own give-up logic run.
    """

    from fastapi import WebSocketDisconnect

    manager = _manager(echo_agent, tmp_path)
    client = TestClient(create_app(manager))
    sid = client.post("/sessions", json={"sim": "demo", "model": "ollama:qwen3"}).json()[
        "session_id"
    ]
    with socket.socket() as probe:  # a port nothing is listening on
        probe.bind(("127.0.0.1", 0))
        dead_port = probe.getsockname()[1]
    manager.get(sid).session.web_plot_port = dead_port

    with pytest.raises(WebSocketDisconnect), client.websocket_connect(f"/live/{sid}/viz/plot"):
        raise AssertionError("handshake was accepted against a dead upstream")


def test_upstream_handshake_error_is_a_refusal_not_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A plot server answering the upgrade with an HTTP error must read as a
    refused handshake, not an unhandled exception.

    Bonito answers a WS upgrade for a route that no longer exists (one removed by
    close_plots while a browser still held its tab) with a plain HTTP response;
    the client library surfaces that as InvalidHandshake, which is not an OSError.
    Unhandled, every such reconnect attempt prints a full ASGI traceback.
    """

    import websockets
    from fastapi import WebSocketDisconnect
    from websockets.exceptions import InvalidHandshake

    def rejecting_dial(url: str, **kwargs: Any) -> Any:
        raise InvalidHandshake("upgrade rejected")

    monkeypatch.setattr(websockets, "connect", rejecting_dial)

    manager = _manager(echo_agent, tmp_path)
    client = TestClient(create_app(manager))
    sid = client.post("/sessions", json={"sim": "demo", "model": "ollama:qwen3"}).json()[
        "session_id"
    ]
    manager.get(sid).session.web_plot_port = 9999  # never dialled for real

    with pytest.raises(WebSocketDisconnect), client.websocket_connect(f"/live/{sid}/viz/plot"):
        raise AssertionError("handshake was accepted despite the upstream rejection")


def test_closing_a_vanished_client_does_not_raise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The proxy's final close must swallow a client that is already gone.

    Under uvicorn, sending the close frame to a browser that disconnected abruptly
    (tab closed, machine slept) raises ClientDisconnected, which starlette converts
    to WebSocketDisconnect; neither is a RuntimeError, so a close guard covering
    only that lets the exception escape and print an ASGI traceback per abandoned
    socket. The endpoint is driven directly with uvicorn's actual exception so the
    starlette conversion path is the thing under test.
    """

    import websockets
    from fastapi.routing import APIWebSocketRoute
    from starlette.websockets import WebSocket
    from uvicorn.protocols.utils import ClientDisconnected

    class _IdleUpstream:
        async def __aenter__(self) -> Any:
            return self

        async def __aexit__(self, *exc: Any) -> bool:
            return False

        async def send(self, data: Any) -> None:
            pass

        async def close(self) -> None:
            pass

        def __aiter__(self) -> Any:
            return self

        async def __anext__(self) -> Any:
            await asyncio.sleep(3600)
            raise StopAsyncIteration

    monkeypatch.setattr(websockets, "connect", lambda url, **kw: _IdleUpstream())

    manager = _manager(echo_agent, tmp_path)
    app = create_app(manager)
    client = TestClient(app)
    sid = client.post("/sessions", json={"sim": "demo", "model": "ollama:qwen3"}).json()[
        "session_id"
    ]
    manager.get(sid).session.web_plot_port = 9999  # dial is monkeypatched

    endpoint = next(
        r.endpoint
        for r in app.routes
        if isinstance(r, APIWebSocketRoute) and r.path == "/live/{session_id}/{path:path}"
    )

    # One connect, then the client vanishes; the close frame then hits a dead
    # transport, which is exactly when uvicorn raises ClientDisconnected.
    inbox = [{"type": "websocket.connect"}, {"type": "websocket.disconnect", "code": 1006}]

    async def receive() -> dict[str, Any]:
        if inbox:
            return inbox.pop(0)
        await asyncio.sleep(3600)
        raise AssertionError("unreachable")

    async def send(message: dict[str, Any]) -> None:
        if message["type"] == "websocket.close":
            raise ClientDisconnected()

    scope = {
        "type": "websocket",
        "path": f"/live/{sid}/viz/plot",
        "raw_path": f"/live/{sid}/viz/plot".encode(),
        "query_string": b"",
        "headers": [],
        "scheme": "ws",
        "server": ("testserver", 80),
        "client": ("testclient", 50000),
        "subprotocols": [],
    }
    ws = WebSocket(scope, receive=receive, send=send)
    # Must complete without raising WebSocketDisconnect out of the endpoint,
    # which uvicorn would log as "Exception in ASGI application".
    asyncio.run(endpoint(ws, sid, "viz/plot"))


def test_live_plot_get_waits_out_a_busy_julia(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The HTTP proxy's read timeout must cover a long eval, connect staying quick.

    The plot server shares Julia's interactive thread with the kernel's eval loop,
    so a mounted iframe's GET routinely sits unanswered for the length of a solve:
    the OS accepts the TCP connection and the request queues. A short read timeout
    turns every such busy spell into a 502 the canvas shows as a broken plot;
    waiting renders it when the eval yields.
    """

    import httpx

    captured: list[Any] = []

    async def fake_request(self: Any, method: str, url: str, **kwargs: Any) -> Any:
        captured.append(kwargs.get("timeout"))
        return httpx.Response(200, headers={"content-type": "text/html"}, content=b"ok")

    monkeypatch.setattr(httpx.AsyncClient, "request", fake_request)

    manager = _manager(echo_agent, tmp_path)
    client = TestClient(create_app(manager))
    sid = client.post("/sessions", json={"sim": "demo", "model": "ollama:qwen3"}).json()[
        "session_id"
    ]
    manager.get(sid).session.web_plot_port = 9999  # request is faked

    assert client.get(f"/live/{sid}/viz/plot").status_code == 200
    assert captured and isinstance(captured[0], httpx.Timeout)
    assert (captured[0].read or 0) >= 300, "a busy Julia must queue, not 502"
    assert (captured[0].connect or 0) <= 10, "a dead port must still fail fast"
