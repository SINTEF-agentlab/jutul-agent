"""Tests for tool-result clearing (ContextEditingMiddleware)."""

from __future__ import annotations

from unittest.mock import MagicMock

from langchain.agents.middleware import ClearToolUsesEdit, ContextEditingMiddleware
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from makker.agent.context_editing import (
    build_context_editing_middleware,
    clear_tool_uses_trigger_tokens,
    keep_recent_tool_results,
)


def test_clear_tool_uses_trigger_tokens() -> None:
    assert clear_tool_uses_trigger_tokens(65_536) == int(65_536 * 0.6)
    # Below the summarization trigger (0.8) so clearing is the first response.
    assert clear_tool_uses_trigger_tokens(100_000) < int(100_000 * 0.8)
    assert clear_tool_uses_trigger_tokens(None) == 60_000


def test_keep_recent_tool_results_scales_with_window() -> None:
    # A small (local) window keeps fewer, so the working set can't fill it.
    assert keep_recent_tool_results(65_536) == 3
    assert keep_recent_tool_results(200_000) == 6
    assert keep_recent_tool_results(None) == 6


def test_build_context_editing_middleware_config(monkeypatch) -> None:
    from makker import models

    monkeypatch.setattr(models, "context_window", lambda model_id: 50_000)
    middleware = build_context_editing_middleware(model_id="ollama:qwen3.6:27b")
    (edit,) = middleware.edits
    assert isinstance(edit, ClearToolUsesEdit)
    assert edit.trigger == int(50_000 * 0.6)
    assert edit.keep == 3  # small window → fewer recent results kept
    # The attempt log is referred to by value, so it is never cleared.
    assert "record_attempt" in edit.exclude_tools


def _tool_round(call_id: str, name: str, result: str) -> list:
    """An AIMessage requesting one tool call plus its ToolMessage result."""
    return [
        AIMessage(content="", tool_calls=[{"id": call_id, "name": name, "args": {}}]),
        ToolMessage(content=result, tool_call_id=call_id, name=name),
    ]


async def test_clears_old_tool_results_but_keeps_recent_and_excluded() -> None:
    """Old tool results become a placeholder; recent and excluded ones stay,
    and state["messages"] is left intact (the edit rides the request)."""
    edit = ClearToolUsesEdit(trigger=10, keep=2, exclude_tools=("record_attempt",))
    middleware = ContextEditingMiddleware(edits=[edit])

    messages = [
        HumanMessage(content="start"),
        *_tool_round("a", "sim_read", "OLD SOURCE DUMP " * 50),  # oldest → cleared
        *_tool_round("b", "record_attempt", "attempt #1 id=keep-me"),  # old but excluded
        *_tool_round("c", "run_julia", "recent output one"),  # recent → kept
        *_tool_round("d", "sim_read", "recent output two"),  # recent → kept
    ]
    captured: dict[str, list] = {}

    async def handler(req: ModelRequest) -> AIMessage:
        captured["messages"] = req.messages
        return AIMessage(content="ok")

    request = ModelRequest(
        model=MagicMock(profile={"max_input_tokens": 100_000}),
        messages=messages,
        system_message=None,
        runtime=MagicMock(),
        state={"messages": messages},
    )
    await middleware.awrap_model_call(request, handler)

    seen = {m.tool_call_id: m for m in captured["messages"] if isinstance(m, ToolMessage)}
    assert seen["a"].content == "[cleared]"  # old sim_read cleared
    assert "keep-me" in str(seen["b"].content)  # record_attempt excluded → intact
    assert seen["c"].content == "recent output one"  # within keep window
    assert seen["d"].content == "recent output two"

    # Non-mutating: the original messages still carry the full result.
    original = next(m for m in messages if isinstance(m, ToolMessage) and m.tool_call_id == "a")
    assert original.content.startswith("OLD SOURCE DUMP")
