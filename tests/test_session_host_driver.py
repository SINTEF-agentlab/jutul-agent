"""The host-owned turn loop, settle hooks, and titling (shared by every surface)."""

from __future__ import annotations

from pathlib import Path

import pytest

from fakes import FakeJulia, echo_agent, interrupt_agent, make_fake_adapter
from jutul_agent.session import Session
from jutul_agent.session_host import SessionHost


def _host(agent, tmp_path: Path, *, approval_mode: str = "ask") -> SessionHost:
    session = Session.create(
        julia=FakeJulia(), simulator=make_fake_adapter(tmp_path), state_root=tmp_path
    )
    return SessionHost(
        session=session,
        agent=agent,
        model="openai:gpt-test",
        approval_mode=approval_mode,
        surface="tui",
    )


async def test_drive_turn_auto_resumes_in_auto_mode(tmp_path: Path) -> None:
    agent = interrupt_agent()
    host = _host(agent, tmp_path)

    result = await host.drive_turn(
        lambda: host.runner.run_prompt("do it"), approval_mode="auto"
    )

    assert result.interrupts == []
    assert len(agent.resume_inputs) == 1  # the loop approved and resumed once


async def test_drive_turn_stops_for_a_human_in_ask_mode(tmp_path: Path) -> None:
    agent = interrupt_agent()
    host = _host(agent, tmp_path)

    result = await host.drive_turn(
        lambda: host.runner.run_prompt("do it"), approval_mode="ask"
    )

    assert [i.interrupt_id for i in result.interrupts] == ["interrupt-1"]
    assert agent.resume_inputs == []


async def test_pending_interrupts_reads_the_persisted_state(tmp_path: Path) -> None:
    agent = interrupt_agent()
    host = _host(agent, tmp_path)
    await host.drive_turn(lambda: host.runner.run_prompt("do it"), approval_mode="ask")

    pending = await host.pending_interrupts()

    assert [p.interrupt_id for p in pending] == ["interrupt-1"]


async def test_maybe_title_upgrades_once_after_the_first_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_title(model_id, conversation):
        assert "User:" in conversation
        return "A Fine Title"

    monkeypatch.setattr("jutul_agent.agent.titling.generate_session_title", fake_title)
    host = _host(echo_agent(), tmp_path)
    await host.runner.run_prompt("hello")  # records the first user message

    seen: list[str] = []
    task = host.maybe_title(seen.append)
    assert task is not None
    await task

    assert seen == ["A Fine Title"]
    assert host.session.title == "A Fine Title"
    assert host.maybe_title(seen.append) is None  # once per session


async def test_maybe_title_skips_later_turns(tmp_path: Path) -> None:
    host = _host(echo_agent(), tmp_path)
    await host.runner.run_prompt("one")
    await host.runner.run_prompt("two")

    assert host.maybe_title() is None
    assert host.titled is False  # never claimed, so a fresh host could still title
