"""Recover the turn when the model ends it without doing anything useful.

Two failure modes leave the agent silent with no work done, and the loop
reads both as a finished turn:

- **Unparseable tool calls.** A reply whose only tool calls have malformed
  JSON arguments never routes to the tool node. The motivating case was a
  local model that *intermittently* serializes two parallel calls into one
  call's argument string (``{...}{...}``). Parallel calls are normal and
  wanted, so the fix is a generic net here, not a prompt that suppresses
  parallel calls for every turn to hedge against the occasional bad one.
- **Empty turns.** A thinking-capable model sometimes ends a turn after its
  reasoning block without emitting an answer or a tool call, so the loop sees
  nothing to route and stops. The reasoning is separated into its own channel,
  so the reply's visible text is empty, which is the signal.

For both, this middleware sends the model around for another attempt (feeding
back the parse errors as error tool results, or nudging an empty turn to act or
answer), bounded so a model that cannot recover doesn't spin forever.
"""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, hook_config
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

# Consecutive recoveries allowed before giving up: a model that cannot emit a
# valid tool call (or any answer) twice in a row will not manage it on the
# third spin either, and the turn has to end somewhere the user can see.
_MAX_CONSECUTIVE_RECOVERIES = 2

_RECOVERY_MARK = "invalid_tool_call_recovery"
_EMPTY_TURN_MARK = "empty_turn_recovery"

_EMPTY_TURN_NUDGE = (
    "Your last turn had no reply and no tool call. If the task isn't finished, "
    "call a tool to continue; otherwise, state your final answer now."
)


class InvalidToolCallRecoveryMiddleware(AgentMiddleware):
    """Retry the model when a turn ends with bad tool calls or no output."""

    @hook_config(can_jump_to=["model"])
    def after_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return _recover(state)

    @hook_config(can_jump_to=["model"])
    async def aafter_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return _recover(state)


def _recover(state: Any) -> dict[str, Any] | None:
    messages = state.get("messages") if isinstance(state, dict) else getattr(state, "messages", [])
    if not messages:
        return None
    last = messages[-1]
    if not isinstance(last, AIMessage) or last.tool_calls:
        return None
    if last.invalid_tool_calls:
        return _recover_invalid_tool_calls(last, messages)
    if _is_empty_turn(last):
        return _recover_empty_turn(messages)
    return None


def _recover_invalid_tool_calls(last: AIMessage, messages: list[Any]) -> dict[str, Any] | None:
    if _consecutive_marked(messages, ToolMessage, _RECOVERY_MARK) >= _MAX_CONSECUTIVE_RECOVERIES:
        return None
    errors = [
        ToolMessage(
            content=(
                f"Tool call could not be parsed: {call.get('error') or 'invalid arguments'}. "
                "Re-issue it as a single tool call with one valid JSON object as arguments."
            ),
            tool_call_id=str(call.get("id") or "invalid_tool_call"),
            name=call.get("name") or "tool",
            status="error",
            additional_kwargs={_RECOVERY_MARK: True},
        )
        for call in last.invalid_tool_calls
    ]
    return {"messages": errors, "jump_to": "model"}


def _recover_empty_turn(messages: list[Any]) -> dict[str, Any] | None:
    if _consecutive_marked(messages, HumanMessage, _EMPTY_TURN_MARK) >= _MAX_CONSECUTIVE_RECOVERIES:
        return None
    nudge = HumanMessage(content=_EMPTY_TURN_NUDGE, additional_kwargs={_EMPTY_TURN_MARK: True})
    return {"messages": [nudge], "jump_to": "model"}


def _is_empty_turn(message: AIMessage) -> bool:
    """The model produced no answer text, only reasoning or nothing at all."""
    return not _visible_text(message).strip()


def _visible_text(message: AIMessage) -> str:
    """The reply's answer text, excluding reasoning/thinking content blocks."""
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block if isinstance(block, str) else str(block.get("text") or "")
            for block in content
            if isinstance(block, str) or (isinstance(block, dict) and block.get("type") == "text")
        ]
        return "".join(parts)
    return ""


def _consecutive_marked(messages: list[Any], kind: type, mark: str) -> int:
    """Count trailing ``kind`` messages bearing ``mark``, since the user last spoke.

    A real (unmarked) user message resets the count. The marked recovery
    messages this middleware injects are what we tally.
    """
    count = 0
    for message in reversed(messages):
        if isinstance(message, HumanMessage) and not (message.additional_kwargs or {}).get(mark):
            break
        if isinstance(message, kind) and (message.additional_kwargs or {}).get(mark):
            count += 1
    return count
