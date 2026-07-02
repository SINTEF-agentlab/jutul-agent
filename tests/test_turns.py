"""Tests for the turn runner and HITL trace recording."""

from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from fakes import (
    ScriptedV3Agent,
    interrupt_agent,
    tool_call_events,
    v3_message_event,
    v3_tool_event,
    v3_values_event,
)
from jutul_agent.agent.turns import (
    TurnReasoningDelta,
    TurnRunner,
    TurnTextDelta,
    TurnTextEnd,
    TurnToolEvent,
)
from jutul_agent.trace import TraceLog


def _message_agent() -> ScriptedV3Agent:
    human = HumanMessage(content="hi")
    assistant = AIMessage(content="hello")
    return ScriptedV3Agent(
        [
            v3_message_event(human),
            v3_message_event(assistant),
            v3_values_event([human, assistant]),
        ]
    )


def _chunk_message_agent() -> ScriptedV3Agent:
    human = HumanMessage(content="hi")
    assistant = AIMessage(content="hello")
    return ScriptedV3Agent(
        [
            v3_message_event(AIMessageChunk(content="hel")),
            v3_message_event(AIMessageChunk(content="lo", chunk_position="last")),
            v3_values_event([human, assistant]),
        ]
    )


def _v3_reasoning_and_tool_agent() -> ScriptedV3Agent:
    human = HumanMessage(content="hi")
    assistant = AIMessage(content="hello")
    return ScriptedV3Agent(
        [
            v3_values_event([human]),
            v3_message_event(
                {
                    "event": "content-block-delta",
                    "delta": {"type": "reasoning-delta", "reasoning": "checking context"},
                }
            ),
            v3_message_event(
                {
                    "event": "content-block-delta",
                    "delta": {"type": "text-delta", "text": "hello"},
                }
            ),
            v3_message_event(
                {
                    "event": "content-block-finish",
                    "content": {
                        "type": "tool_call",
                        "id": "tool-1",
                        "name": "execute",
                        "args": {"path": "/"},
                    },
                }
            ),
            v3_tool_event(
                {
                    "event": "tool-started",
                    "tool_call_id": "tool-1",
                    "tool_name": "execute",
                    "input": {"path": "/"},
                }
            ),
            v3_tool_event(
                {
                    "event": "tool-finished",
                    "tool_call_id": "tool-1",
                    "output": ["a", "b"],
                }
            ),
            v3_message_event({"event": "message-finish"}),
            v3_values_event([human, assistant]),
        ]
    )


async def test_turn_runner_emits_text_as_deltas() -> None:
    """The runner surfaces text as TurnTextDelta events closed by a TurnTextEnd;
    full assembled messages live in ``result.messages``."""

    runner = TurnRunner(_message_agent(), thread_id="thread-1")
    seen: list[object] = []

    result = await runner.run_prompt("hi", on_message=lambda msg: seen.append(msg))

    deltas = [m for m in seen if isinstance(m, TurnTextDelta)]
    assert "".join(d.text for d in deltas) == "hello"
    assert isinstance(seen[-1], TurnTextEnd) or any(isinstance(m, TurnTextEnd) for m in seen)
    assert result.messages[-1].content == "hello"
    assert result.interrupts == []


async def test_turn_runner_streams_message_chunks() -> None:
    runner = TurnRunner(_chunk_message_agent(), thread_id="thread-1")
    seen: list[object] = []

    result = await runner.run_prompt("hi", on_message=lambda msg: seen.append(msg))

    deltas = [m for m in seen if isinstance(m, TurnTextDelta)]
    assert "".join(d.text for d in deltas) == "hello"
    assert result.messages[-1].content == "hello"
    assert result.interrupts == []


async def test_turn_runner_streams_v3_reasoning_and_tool_events() -> None:
    runner = TurnRunner(_v3_reasoning_and_tool_agent(), thread_id="thread-1")
    seen: list[object] = []

    result = await runner.run_prompt("hi", on_message=lambda msg: seen.append(msg))

    reasoning = [m for m in seen if isinstance(m, TurnReasoningDelta)]
    assert any(r.text == "checking context" for r in reasoning)

    text_deltas = [m for m in seen if isinstance(m, TurnTextDelta) and m.text]
    assert "".join(d.text for d in text_deltas) == "hello"

    tool_events = [m for m in seen if isinstance(m, TurnToolEvent)]
    events_by_kind = {e.event for e in tool_events}
    assert "requested" in events_by_kind
    assert "started" in events_by_kind
    assert "finished" in events_by_kind
    finished = next(e for e in tool_events if e.event == "finished")
    assert finished.content == '["a", "b"]'

    assert result.messages[-1].content == "hello"


async def test_tool_result_does_not_leak_into_assistant_text() -> None:
    """A ToolMessage rides ``run.messages`` (node='tools') with its full content
    as ``.text``. The turn runner must render it only through tool events — never
    as assistant prose — so file reads / skill / memory text don't get dumped."""

    tool_output = "---\nname: battmo-overview\ndescription: workflow\n---\nLOTS OF SKILL TEXT"
    agent = ScriptedV3Agent(
        tool_call_events(
            tool_name="read_file",
            tool_call_id="call_read_1",
            args={"file_path": "/skills/shared/battmo-overview/SKILL.md"},
            output=tool_output,
            final_text="Read the overview skill; ready to proceed.",
        )
    )
    runner = TurnRunner(agent, thread_id="thread-1")
    seen: list[object] = []

    await runner.run_prompt("read the skill", on_message=lambda msg: seen.append(msg))

    streamed_text = "".join(m.text for m in seen if isinstance(m, TurnTextDelta) and m.text)
    assert streamed_text == "Read the overview skill; ready to proceed."
    assert tool_output not in streamed_text
    assert "SKILL TEXT" not in streamed_text

    # The result still reaches the UI as a tool event (the tool card), not prose.
    finished = [m for m in seen if isinstance(m, TurnToolEvent) and m.event == "finished"]
    assert any(tool_output in (e.content or "") for e in finished)


async def test_turn_runner_parses_server_tool_call() -> None:
    agent = ScriptedV3Agent(
        [
            v3_message_event(
                {
                    "event": "content-block-finish",
                    "content": {
                        "type": "server_tool_call",
                        "id": "tool-2",
                        "name": "search",
                        "args": "query=jutul",
                    },
                }
            ),
            v3_values_event([]),
        ]
    )
    runner = TurnRunner(agent, thread_id="thread-1")
    seen: list[object] = []

    await runner.run_prompt("hi", on_message=lambda msg: seen.append(msg))

    requested = [m for m in seen if isinstance(m, TurnToolEvent) and m.event == "requested"]
    assert len(requested) == 1
    assert requested[0].tool_name == "search"
    assert requested[0].args == {"value": "query=jutul"}


async def test_turn_runner_streams_tool_output_deltas() -> None:
    human = HumanMessage(content="hi")
    assistant = AIMessage(content="done")
    agent = ScriptedV3Agent(
        [
            v3_tool_event(
                {
                    "event": "tool-started",
                    "tool_call_id": "tool-julia",
                    "tool_name": "run_julia",
                    "input": {"code": "run()"},
                }
            ),
            v3_tool_event(
                {
                    "event": "tool-output-delta",
                    "tool_call_id": "tool-julia",
                    "delta": "progress\n",
                }
            ),
            v3_tool_event(
                {
                    "event": "tool-output-delta",
                    "tool_call_id": "tool-julia",
                    "delta": "→ 42\n",
                }
            ),
            v3_tool_event(
                {
                    "event": "tool-finished",
                    "tool_call_id": "tool-julia",
                    "output": "progress\n→ 42\n",
                }
            ),
            v3_values_event([human, assistant]),
        ]
    )
    runner = TurnRunner(agent, thread_id="thread-1")
    seen: list[object] = []

    result = await runner.run_prompt("hi", on_message=lambda msg: seen.append(msg))

    tool_events = [m for m in seen if isinstance(m, TurnToolEvent)]
    deltas = [e for e in tool_events if e.event == "delta"]
    assert [d.content for d in deltas] == ["progress\n", "→ 42\n"]
    finished = next(e for e in tool_events if e.event == "finished")
    assert finished.content == "progress\n→ 42\n"
    assert result.messages[-1].content == "done"


async def test_turn_runner_records_user_message_and_hitl_round_trip(
    tmp_path: Path,
) -> None:
    agent = interrupt_agent()
    log = TraceLog(tmp_path / "trace.sqlite")
    runner = TurnRunner(agent, thread_id="thread-1", trace=log)

    try:
        first = await runner.run_prompt("need approval")
        assert len(first.interrupts) == 1
        assert first.interrupts[0].interrupt_id == "interrupt-1"

        second = await runner.resume(
            {"interrupt-1": {"decisions": [{"type": "approve"}]}},
        )

        assert hasattr(agent.inputs[1], "resume")
        assert second.messages[-1].content == "approval handled"

        kinds = [event.kind for event in log.iter_events()]
        assert kinds == ["message_user", "hitl_request", "hitl_response"]
    finally:
        log.close()
