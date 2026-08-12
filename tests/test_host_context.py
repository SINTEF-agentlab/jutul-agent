"""The host application's selection: how it reaches the agent and survives a resume.

An application that embeds the web UI launches it with what it currently has
selected. That value becomes a layer of the agent's system prompt, is recorded on
the session, and is re-stated on every create, resume, and in-place change, so
what the agent believes is selected tracks what the user sees in the application.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from jutul_agent.agent.capabilities import (
    HOST_CONTEXT_CAPABILITY,
    Capability,
    host_context_capability,
    replace_host_context,
)
from jutul_agent.interfaces.server.app import create_app
from jutul_agent.interfaces.server.manager import SessionManager
from jutul_agent.lab.fakes import (
    FakeJulia,
    ScriptedV3Agent,
    echo_agent,
    interrupt_agent,
    make_fake_adapter,
)
from jutul_agent.session import (
    HOST_CONTEXT_FILENAME,
    Session,
    default_session_id,
    read_host_context,
)
from jutul_agent.session_host import SessionHost
from jutul_agent.trace import schema

# The payload is opaque to jutul-agent: whatever the embedding application calls
# its own objects. These stand in for that without borrowing any one application's
# vocabulary.
SELECTION: dict[str, Any] = {
    "primaryId": "11111111-1111-4111-8111-111111111111",
    "items": ["22222220-2222-4222-8222-222222222222"],
    "groups": ["33333330-3333-4333-8333-333333333333"],
}
OTHER_SELECTION: dict[str, Any] = {"primaryId": "44444444-4444-4444-8444-444444444444"}


@pytest.fixture(autouse=True)
def _provider_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Placeholder keys so the create-session credential guard never gates these tests."""
    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.setenv(var, "test-key")


def _manager(
    tmp_path: Path, agent_factory: Callable[[], Any] = echo_agent, *, max_live: int = 16
) -> SessionManager:
    """A fake-backed manager that forwards ``extensions``, which is what carries the
    selection into the session (the real host factory does the same)."""

    async def host_factory(
        *, sim, model, approval_mode, workspace, resume, session_id, extensions=()
    ) -> SessionHost:
        adapter = make_fake_adapter(tmp_path)
        sid = session_id or default_session_id()
        maker = Session.resume if resume else Session.create
        session = maker(julia=FakeJulia(), simulator=adapter, session_id=sid, state_root=tmp_path)
        return SessionHost(session=session, agent=agent_factory(), extensions=extensions)

    return SessionManager(host_factory=host_factory, max_live=max_live)


def _fragment(host: SessionHost) -> str:
    """The host-context layer's prompt text, or "" when the session has no selection."""
    return next(
        (cap.prompt_fragment for cap in host._extensions if cap.name == HOST_CONTEXT_CAPABILITY),
        "",
    )


# --- the capability layer --------------------------------------------------


def test_capability_states_the_selection_verbatim() -> None:
    cap = host_context_capability(SELECTION)
    assert cap is not None and cap.name == HOST_CONTEXT_CAPABILITY
    # Every identifier the host sent must be readable in the prompt, or the agent
    # cannot pass them to the host's tools.
    assert SELECTION["primaryId"] in cap.prompt_fragment
    assert SELECTION["items"][0] in cap.prompt_fragment
    assert SELECTION["groups"][0] in cap.prompt_fragment
    # It applies to every surface: a host app is free to speak the protocol itself.
    assert cap.surfaces == ()


def test_capability_is_absent_when_there_is_no_selection() -> None:
    assert host_context_capability(None) is None
    assert host_context_capability({}) is None


def test_replace_swaps_the_layer_in_place_and_leaves_the_others() -> None:
    others = [Capability(name="host-app"), Capability(name="sim-web")]
    with_context = replace_host_context([*others, host_context_capability(SELECTION)], SELECTION)
    swapped = replace_host_context(with_context, OTHER_SELECTION)

    assert [cap.name for cap in swapped] == ["host-app", "sim-web", HOST_CONTEXT_CAPABILITY]
    assert OTHER_SELECTION["primaryId"] in swapped[-1].prompt_fragment
    assert SELECTION["primaryId"] not in swapped[-1].prompt_fragment
    # Clearing the selection drops the layer rather than leaving an empty one.
    assert [cap.name for cap in replace_host_context(swapped, None)] == ["host-app", "sim-web"]


# --- recording it on the session -------------------------------------------


def test_session_persists_the_selection_and_records_the_change(tmp_path: Path) -> None:
    session = Session.create(
        julia=FakeJulia(), simulator=make_fake_adapter(tmp_path), state_root=tmp_path
    )
    assert session.adopt_host_context(SELECTION) is True
    assert session.host_context == SELECTION
    assert read_host_context(session.state_dir) == SELECTION

    # An unchanged value is not re-adopted: the caller uses this to skip a rebuild.
    assert session.adopt_host_context(dict(SELECTION)) is False

    assert session.adopt_host_context(OTHER_SELECTION) is True
    kinds = [e.kind for e in session.trace.iter_events()]
    assert kinds.count(schema.HOST_CONTEXT) == 2, "each change is recorded, the repeat is not"

    # Clearing removes the file, so a later resume does not revive a stale selection.
    assert session.adopt_host_context(None) is True
    assert not (session.state_dir / HOST_CONTEXT_FILENAME).exists()


def test_a_resumed_session_reads_its_stored_selection(tmp_path: Path) -> None:
    adapter = make_fake_adapter(tmp_path)
    session = Session.create(julia=FakeJulia(), simulator=adapter, state_root=tmp_path)
    session.adopt_host_context(SELECTION)

    reopened = Session.resume(
        julia=FakeJulia(), simulator=adapter, session_id=session.session_id, state_root=tmp_path
    )
    assert reopened.host_context == SELECTION


def test_unreadable_stored_context_is_no_context(tmp_path: Path) -> None:
    # Truncated or hand-edited state must not stop a session from opening.
    (tmp_path / HOST_CONTEXT_FILENAME).write_text("{not json", encoding="utf-8")
    assert read_host_context(tmp_path) is None
    (tmp_path / HOST_CONTEXT_FILENAME).write_text('["a list"]', encoding="utf-8")
    assert read_host_context(tmp_path) is None


# --- creating and resuming over the API ------------------------------------


def test_create_carries_the_selection_into_the_agent_and_onto_disk(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    with TestClient(create_app(manager)) as client:
        sid = client.post(
            "/sessions", json={"sim": "jutuldarcy", "host_context": SELECTION}
        ).json()["session_id"]
        host = manager.get(sid)

        assert SELECTION["primaryId"] in _fragment(host)
        assert host.session.host_context == SELECTION
        stored = json.loads((host.session.state_dir / HOST_CONTEXT_FILENAME).read_text())
        assert stored == SELECTION


def test_create_without_a_selection_adds_no_layer(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    with TestClient(create_app(manager)) as client:
        sid = client.post("/sessions", json={"sim": "jutuldarcy"}).json()["session_id"]
        host = manager.get(sid)

        assert _fragment(host) == ""
        assert host.session.host_context is None
        assert not (host.session.state_dir / HOST_CONTEXT_FILENAME).exists()


def test_resume_from_disk_falls_back_to_the_stored_selection(tmp_path: Path) -> None:
    # A UI opened outside the host application (a plain browser tab) sends no
    # selection. The session must come back knowing the one it was working on.
    manager = _manager(tmp_path)
    with TestClient(create_app(manager)) as client:
        sid = client.post(
            "/sessions", json={"sim": "jutuldarcy", "host_context": SELECTION}
        ).json()["session_id"]
        client.delete(f"/sessions/{sid}")  # drop the live host, forcing a from-disk resume

        body = client.post(f"/sessions/{sid}/resume", json={"sim": "jutuldarcy"}).json()
        assert body["kernel_restarted"] is True
        host = manager.get(sid)
        assert host.session.host_context == SELECTION
        assert SELECTION["primaryId"] in _fragment(host)


def test_resume_from_disk_prefers_the_application_over_the_stored_selection(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path)
    with TestClient(create_app(manager)) as client:
        sid = client.post(
            "/sessions", json={"sim": "jutuldarcy", "host_context": SELECTION}
        ).json()["session_id"]
        client.delete(f"/sessions/{sid}")

        client.post(
            f"/sessions/{sid}/resume", json={"sim": "jutuldarcy", "host_context": OTHER_SELECTION}
        )
        host = manager.get(sid)
        assert host.session.host_context == OTHER_SELECTION
        assert OTHER_SELECTION["primaryId"] in _fragment(host)


def test_reattaching_a_live_session_adopts_the_current_selection(tmp_path: Path) -> None:
    # The one thing a reattach does take from the resume request: the live host is
    # authoritative about its own settings, but not about what the *application*
    # has selected right now.
    manager = _manager(tmp_path)
    with TestClient(create_app(manager)) as client:
        sid = client.post(
            "/sessions", json={"sim": "jutuldarcy", "host_context": SELECTION}
        ).json()["session_id"]
        host = manager.get(sid)
        rebuilds: list[dict] = []
        host.reconfigure = lambda **kw: rebuilds.append(kw)  # type: ignore[method-assign]

        client.post(
            f"/sessions/{sid}/resume", json={"sim": "jutuldarcy", "host_context": OTHER_SELECTION}
        )
        assert host.session.host_context == OTHER_SELECTION
        assert OTHER_SELECTION["primaryId"] in _fragment(host)
        assert len(rebuilds) == 1, "the system prompt only changes by rebuilding the agent"

        # Reconnecting without one (or with the same one) leaves it alone.
        client.post(f"/sessions/{sid}/resume", json={"sim": "jutuldarcy"})
        client.post(
            f"/sessions/{sid}/resume", json={"sim": "jutuldarcy", "host_context": OTHER_SELECTION}
        )
        assert host.session.host_context == OTHER_SELECTION
        assert len(rebuilds) == 1


# --- where the host application listens ------------------------------------


def test_create_records_the_host_api_url(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    with TestClient(create_app(manager)) as client:
        sid = client.post(
            "/sessions", json={"sim": "jutuldarcy", "host_api": "http://127.0.0.1:55924"}
        ).json()["session_id"]
        host = manager.get(sid)

        assert host.session.host_api == "http://127.0.0.1:55924"
        # Never stored: it describes the application as it is running now.
        assert not any(host.session.state_dir.glob("*host_api*"))

        # A path prefix is kept, its trailing slash dropped so joining cannot double up.
        other = client.post(
            "/sessions",
            json={"sim": "jutuldarcy", "host_api": "http://127.0.0.1:55924/app/"},
        ).json()["session_id"]
        assert manager.get(other).session.host_api == "http://127.0.0.1:55924/app"


def test_resume_takes_the_url_from_the_request_only(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    with TestClient(create_app(manager)) as client:
        sid = client.post(
            "/sessions", json={"sim": "jutuldarcy", "host_api": "http://127.0.0.1:55924"}
        ).json()["session_id"]
        host = manager.get(sid)

        # The application restarted on a new port: a reattach must follow it.
        client.post(
            f"/sessions/{sid}/resume",
            json={"sim": "jutuldarcy", "host_api": "http://127.0.0.1:4254"},
        )
        assert host.session.host_api == "http://127.0.0.1:4254"

        # Resumed from outside the application: the tools have nowhere to call,
        # which is the honest state, and better than a stale port.
        client.post(f"/sessions/{sid}/resume", json={"sim": "jutuldarcy"})
        assert host.session.host_api is None


@pytest.mark.parametrize(
    "bad",
    [
        # A quote or backtick would escape the string literal a capability
        # interpolates the address into, turning a launch parameter into code.
        'http://127.0.0.1:5000/";run(`id`);x=raw"',
        "http://127.0.0.1:5000/$(read(`id`))",
        "http://127.0.0.1:5000/a\nb",
        "file:///etc/passwd",
        "javascript:alert(1)",
        "http://user:pw@127.0.0.1:5000",  # credentials belong nowhere near this
        "not a url",
    ],
)
def test_a_host_api_that_is_not_a_plain_address_is_refused(tmp_path: Path, bad: str) -> None:
    # The bundled UI checks this too, but a request can reach the server without
    # going through it, so this is the boundary that has to hold.
    manager = _manager(tmp_path)
    with TestClient(create_app(manager)) as client:
        resp = client.post("/sessions", json={"sim": "jutuldarcy", "host_api": bad})

        assert resp.status_code == 400, bad
        assert "host_api" in str(resp.json()["detail"])
        # Refused before anything was built: no session is left behind.
        assert manager.list_ids() == []


def test_a_bad_host_api_on_resume_leaves_the_session_alone(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    with TestClient(create_app(manager)) as client:
        sid = client.post(
            "/sessions", json={"sim": "jutuldarcy", "host_api": "http://127.0.0.1:55924"}
        ).json()["session_id"]
        host = manager.get(sid)

        resp = client.post(
            f"/sessions/{sid}/resume", json={"sim": "jutuldarcy", "host_api": 'http://x/"evil'}
        )

        assert resp.status_code == 400
        # The live session keeps the address it had rather than a half-applied one.
        assert host.session.host_api == "http://127.0.0.1:55924"


# --- changing it while a session is open -----------------------------------


def test_ws_selection_change_rebuilds_the_agent_and_tells_the_user(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    with TestClient(create_app(manager)) as client:
        sid = client.post(
            "/sessions", json={"sim": "jutuldarcy", "host_context": SELECTION}
        ).json()["session_id"]
        host = manager.get(sid)
        rebuilds: list[dict] = []
        host.reconfigure = lambda **kw: rebuilds.append(kw)  # type: ignore[method-assign]

        with client.websocket_connect(f"/sessions/{sid}/stream") as ws:
            ws.send_json({"type": "host_context", "context": OTHER_SELECTION})
            notice = ws.receive_json()
            # The same selection again is not a change, so nothing is rebuilt. The
            # error from a following bad message proves it was processed in order.
            ws.send_json({"type": "host_context", "context": OTHER_SELECTION})
            ws.send_json({"type": "host_context", "context": "not an object"})
            err = ws.receive_json()

    assert notice["type"] == "notice" and "selection" in notice["text"]
    assert err["type"] == "error" and "JSON object" in err["message"]
    assert len(rebuilds) == 1
    assert host.session.host_context == OTHER_SELECTION
    assert OTHER_SELECTION["primaryId"] in _fragment(host)


def test_a_selection_change_while_paused_on_approval_keeps_the_turn_answerable(
    tmp_path: Path,
) -> None:
    # A turn paused for approval has finished its task, so it does not count as
    # busy and the change is adopted at once, rebuilding the agent under a
    # pending interrupt. The interrupt lives in the checkpointed graph state, not
    # in the agent object, so answering it must still complete the turn.
    manager = _manager(tmp_path, interrupt_agent)
    with TestClient(create_app(manager)) as client:
        sid = client.post(
            "/sessions", json={"sim": "jutuldarcy", "host_context": SELECTION}
        ).json()["session_id"]
        host = manager.get(sid)
        host.reconfigure = lambda **kw: None  # type: ignore[method-assign]

        with client.websocket_connect(f"/sessions/{sid}/stream") as ws:
            ws.send_json({"type": "prompt", "text": "do something needing approval"})
            assert ws.receive_json()["type"] == "interrupt"

            ws.send_json({"type": "host_context", "context": OTHER_SELECTION})
            assert ws.receive_json()["type"] == "notice"

            ws.send_json({"type": "decision", "decision": "approve"})
            kinds = []
            while "turn_end" not in kinds:
                kinds.append(ws.receive_json()["type"])

    assert host.session.host_context == OTHER_SELECTION


def test_a_failed_rebuild_reports_and_keeps_the_session(tmp_path: Path) -> None:
    # The rebuild runs from the message loop, so a failure must not take the
    # connection down over a selection change.
    manager = _manager(tmp_path)
    with TestClient(create_app(manager)) as client:
        sid = client.post("/sessions", json={"sim": "jutuldarcy"}).json()["session_id"]
        host = manager.get(sid)

        def explode(**_kw):
            raise RuntimeError("no provider key")

        host.reconfigure = explode  # type: ignore[method-assign]

        with client.websocket_connect(f"/sessions/{sid}/stream") as ws:
            ws.send_json({"type": "host_context", "context": SELECTION})
            error = ws.receive_json()
            # The socket is still usable afterwards.
            ws.send_json({"type": "command", "command": "bogus"})
            after = ws.receive_json()

    assert error["type"] == "error" and "selection" in error["message"]
    assert after["type"] == "error" and "bogus" in after["message"]
    # Recorded even though the rebuild failed, so the next one states it.
    assert host.session.host_context == SELECTION


def test_ws_selection_change_during_a_turn_waits_for_the_turn_to_settle(tmp_path: Path) -> None:
    # Rebuilding the agent swaps the turn runner, so a change that arrives mid-turn
    # is held: the turn the user is watching finishes against the objects it
    # started on. The notice therefore lands *after* turn_end, not before it.
    class _SlowAgent(ScriptedV3Agent):
        async def astream_events(self, stream_input, **kwargs):
            await asyncio.sleep(0.3)
            return await super().astream_events(stream_input, **kwargs)

    def _slow_agent() -> ScriptedV3Agent:
        agent = echo_agent()
        return _SlowAgent(agent._events)

    manager = _manager(tmp_path, _slow_agent)
    with TestClient(create_app(manager)) as client:
        sid = client.post(
            "/sessions", json={"sim": "jutuldarcy", "host_context": SELECTION}
        ).json()["session_id"]
        host = manager.get(sid)
        host.reconfigure = lambda **kw: None  # type: ignore[method-assign]

        with client.websocket_connect(f"/sessions/{sid}/stream") as ws:
            ws.send_json({"type": "prompt", "text": "work on the selection"})
            ws.send_json({"type": "host_context", "context": OTHER_SELECTION})
            kinds: list[str] = []
            while "notice" not in kinds:
                kinds.append(ws.receive_json()["type"])

    assert "turn_end" in kinds, "the turn ran to completion"
    assert kinds.index("turn_end") < kinds.index("notice"), (
        "the selection was adopted only after the turn settled"
    )
    assert host.session.host_context == OTHER_SELECTION
