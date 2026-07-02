"""Tests for invalid-tool-call recovery."""

from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from jutul_agent.agent.builder import build_agent
from jutul_agent.agent.recovery import _recover
from jutul_agent.agent.turns import TurnRunner
from jutul_agent.lab.fakes import FakeJulia, make_fake_adapter, make_scripted_model, scripted_final
from jutul_agent.session import Session

_INVALID = {
    "name": "read_file",
    "args": '{"path": "a"}{"path": "b"}',
    "id": "call_bad_1",
    "error": "Failed to parse tool call arguments as JSON",
    "type": "invalid_tool_call",
}


def _invalid_message() -> AIMessage:
    return AIMessage(content="", invalid_tool_calls=[_INVALID])


def test_recover_feeds_errors_back_and_jumps() -> None:
    state = {"messages": [HumanMessage(content="go"), _invalid_message()]}
    update = _recover(state)
    assert update is not None
    assert update["jump_to"] == "model"
    [error] = update["messages"]
    assert isinstance(error, ToolMessage)
    assert error.tool_call_id == "call_bad_1"
    assert error.status == "error"
    assert "could not be parsed" in str(error.content)


def test_recover_ignores_valid_calls_and_plain_replies() -> None:
    valid = AIMessage(
        content="",
        tool_calls=[{"name": "x", "args": {}, "id": "1"}],
        invalid_tool_calls=[_INVALID],
    )
    assert _recover({"messages": [valid]}) is None  # the tool node handles these
    assert _recover({"messages": [AIMessage(content="done")]}) is None
    assert _recover({"messages": []}) is None


def test_recover_gives_up_after_consecutive_failures() -> None:
    recovery_result = ToolMessage(
        content="x",
        tool_call_id="call_bad_1",
        status="error",
        additional_kwargs={"invalid_tool_call_recovery": True},
    )
    messages = [
        HumanMessage(content="go"),
        _invalid_message(),
        recovery_result,
        _invalid_message(),
        recovery_result,
        _invalid_message(),
    ]
    assert _recover({"messages": messages}) is None
    # A fresh user message resets the budget.
    assert _recover({"messages": [*messages, HumanMessage(content="again"), _invalid_message()]})


def test_recover_nudges_empty_and_reasoning_only_turns() -> None:
    # A blank reply with no tool call: the model fell silent after thinking.
    update = _recover({"messages": [HumanMessage(content="go"), AIMessage(content="")]})
    assert update is not None
    assert update["jump_to"] == "model"
    [nudge] = update["messages"]
    assert isinstance(nudge, HumanMessage)
    assert nudge.additional_kwargs.get("empty_turn_recovery") is True

    # Reasoning-only content (no text block) is also empty as far as an answer goes.
    reasoning_only = AIMessage(content=[{"type": "reasoning", "reasoning": "thinking…"}])
    assert _recover({"messages": [reasoning_only]}) is not None

    # A real answer (string or text block) ends the turn — no nudge.
    assert _recover({"messages": [AIMessage(content="here is the result")]}) is None
    answered = AIMessage(content=[{"type": "text", "text": "the answer"}])
    assert _recover({"messages": [answered]}) is None


def test_recover_gives_up_after_consecutive_empty_turns() -> None:
    nudge = HumanMessage(content="x", additional_kwargs={"empty_turn_recovery": True})
    empty = AIMessage(content="")
    messages = [HumanMessage(content="go"), empty, nudge, empty, nudge, empty]
    assert _recover({"messages": messages}) is None
    # A fresh user message resets the budget.
    assert _recover({"messages": [*messages, HumanMessage(content="again"), empty]})


async def test_agent_recovers_from_empty_turn(tmp_path: Path) -> None:
    """End to end: an empty reasoning-only turn is nudged into an answer."""
    adapter = make_fake_adapter(tmp_path)
    session = Session.create(julia=FakeJulia(), state_root=tmp_path, simulator=adapter)
    model = make_scripted_model([AIMessage(content=""), scripted_final("now answered")])

    agent, _ = build_agent(session, model=model)
    runner = TurnRunner(agent, thread_id=session.session_id, trace=session.trace)
    result = await runner.run_prompt("do the thing")

    contents = [str(getattr(m, "content", "")) for m in result.messages]
    assert any("now answered" in c for c in contents)
    session.finalize()


async def test_agent_recovers_from_invalid_tool_call(tmp_path: Path) -> None:
    """End to end: a malformed tool call costs one round-trip, not the turn."""
    adapter = make_fake_adapter(tmp_path)
    session = Session.create(julia=FakeJulia(), state_root=tmp_path, simulator=adapter)
    model = make_scripted_model([_invalid_message(), scripted_final("recovered and done")])

    agent, _ = build_agent(session, model=model)
    runner = TurnRunner(agent, thread_id=session.session_id, trace=session.trace)
    result = await runner.run_prompt("do the thing")

    contents = [str(getattr(m, "content", "")) for m in result.messages]
    assert any("recovered and done" in c for c in contents)
    assert any("could not be parsed" in c for c in contents)
    session.finalize()
