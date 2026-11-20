"""Tool-result clearing to bound context growth between compactions.

Long sessions accumulate many medium-sized tool results — source reads, REPL
output — that no single eviction is large enough to catch and that summarization
only sweeps up once the window is nearly full. langchain's
``ContextEditingMiddleware`` clears the *older* tool results to a placeholder
once the context crosses a threshold, keeping the most recent ones verbatim. It
is non-mutating (it edits a copy of the request, leaving ``state["messages"]``
intact) and is wired to run ahead of
summarization, so most growth is handled by this cheap pass and the LLM summary
call is reserved for when clearing is not enough. Cleared results stay
re-derivable: the underlying files are still on disk and REPL commands can be
re-run.
"""

from __future__ import annotations

from typing import Any

# Clear old tool results once the context reaches this share of the model's
# window — below the summarization trigger, so clearing is the first, cheaper
# response and the summary call only fires when clearing cannot keep up.
_CLEAR_TRIGGER_FRACTION = 0.6
# Without a discoverable window, a fixed threshold below the summarization
# fallback (100k) keeps the same ordering.
_FALLBACK_TRIGGER_TOKENS = 60_000
# Below this window, keep fewer recent tool results: on a small (local) window
# the kept working set must not itself fill the window.
_SMALL_WINDOW = 100_000
# Tools whose small, structured result the agent refers back to by value
# (e.g. an attempt id passed as a later parent_attempt_id). Clearing them frees
# almost nothing and could break that reference, so leave them intact.
_NEVER_CLEAR: tuple[str, ...] = ("record_attempt",)


def clear_tool_uses_trigger_tokens(window: int | None) -> int:
    """The context size at which old tool results start being cleared."""
    return int(window * _CLEAR_TRIGGER_FRACTION) if window else _FALLBACK_TRIGGER_TOKENS


def keep_recent_tool_results(window: int | None) -> int:
    """How many recent tool results to keep verbatim, sized to the window.

    Clearing keeps the agent's working set, but that set must not itself fill a
    small window: a handful of medium results (each below the eviction
    threshold, so still inline) can otherwise exceed a local model's window.
    Keep fewer on a small window; large single results are evicted separately.
    """
    return 3 if (window and window < _SMALL_WINDOW) else 6


def build_context_editing_middleware(model_id: str | None = None) -> Any:
    """Middleware that clears older tool results when the context grows large.

    The trigger is sized to the model's window so it fires before summarization;
    the most recent results are kept so the active working set is untouched, with
    fewer kept on a small window so that set can't fill it.
    """
    from langchain.agents.middleware import ClearToolUsesEdit, ContextEditingMiddleware

    window = None
    if model_id is not None:
        from makker.models import context_window

        window = context_window(model_id)
    return ContextEditingMiddleware(
        edits=[
            ClearToolUsesEdit(
                trigger=clear_tool_uses_trigger_tokens(window),
                keep=keep_recent_tool_results(window),
                exclude_tools=_NEVER_CLEAR,
            )
        ],
    )
