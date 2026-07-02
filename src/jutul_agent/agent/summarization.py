"""Manual /compact, plus the auto-compaction trigger figure for /context.

Auto-compaction is deepagents' stock ``SummarizationMiddleware`` — installed by
``create_deep_agent`` and sized from the model profile, which
``builder._set_profile_window`` points at the real loaded window;
``TraceRecorder`` surfaces each compaction as a ``context_compaction`` trace
event. This module keeps two small things on top of it: the trigger figure
``/context`` displays, and ``compact_thread`` — the manual ``/compact`` command,
which drives the same deepagents engine on demand against the checkpointed
thread (the deepagents-cli ``/offload`` pattern), non-mutating and recoverable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from deepagents.middleware.summarization import SummarizationEvent, SummarizationMiddleware
from langchain_core.messages.utils import count_tokens_approximately

from jutul_agent.trace import TraceLog
from jutul_agent.trace.schema import CONTEXT_COMPACTION, compaction_payload

# The share of the window at which the stock summarizer triggers, mirrored here
# only so /context can show the figure. Matches deepagents'
# ``compute_summarization_defaults`` default; update if that default changes.
_TRIGGER_FRACTION = 0.85
# Without a discoverable window /context can't size the trigger; this mirrors
# deepagents' fixed no-profile fallback.
_FALLBACK_TRIGGER_TOKENS = 170_000
# Manual /compact keeps a shorter, predictable tail; the count is public so the
# TUI can explain why a short conversation has nothing to compact.
MANUAL_KEEP_MESSAGES = 8
_MANUAL_KEEP: tuple[str, int] = ("messages", MANUAL_KEEP_MESSAGES)

# Manual /compact composes deepagents' summarization engine from the same
# private methods upstream's own tool middleware builds on. Private means a
# deepagents bump may move them; the sweep below turns that into a clear
# "manual compaction is unavailable" instead of an AttributeError mid-command.
# Auto-compaction is untouched either way (it is the stock middleware).
_ENGINE_METHODS: tuple[str, ...] = (
    "_apply_event_to_messages",
    "_determine_cutoff_index",
    "_partition_messages",
    "_acreate_summary",
    "_aoffload_to_backend",
    "_build_new_messages_with_path",
    "_compute_state_cutoff",
)


def manual_compaction_available() -> bool:
    """Whether the installed deepagents still exposes the engine /compact drives."""
    return all(hasattr(SummarizationMiddleware, name) for name in _ENGINE_METHODS)


def auto_compact_trigger_tokens(window: int | None) -> int:
    """The context size at which auto-compaction triggers for ``window`` (display only)."""
    return int(window * _TRIGGER_FRACTION) if window else _FALLBACK_TRIGGER_TOKENS


@dataclass(frozen=True)
class CompactResult:
    messages_summarized: int
    messages_kept: int
    # Approximate input tokens the summary saves over the turns it replaces.
    # Anchors the live /context estimate so it drops immediately, before the
    # next model call measures the new size exactly.
    freed_tokens: int = 0
    # Whether the summarized turns were saved to the backend (recoverable).
    offloaded: bool = False


async def compact_thread(
    agent: Any,
    *,
    thread_id: str,
    model: Any,
    backend: Any,
    trace: TraceLog | None = None,
) -> CompactResult | None:
    """Summarize the thread's older turns now, on demand.

    Drives deepagents' summarization engine against the checkpointed state and
    records the result as a ``_summarization_event`` — the same non-mutating
    mechanism auto-compaction uses, so the raw conversation log is preserved and
    the offloaded turns stay recoverable from ``/conversation_history``. Returns
    ``None`` when there is nothing to compact (too few messages, or no state).
    """
    if not manual_compaction_available():
        raise RuntimeError(
            "manual compaction is unavailable with this deepagents version "
            "(auto-compaction still runs); see agent/summarization.py"
        )
    aget_state = getattr(agent, "aget_state", None)
    aupdate_state = getattr(agent, "aupdate_state", None)
    if aget_state is None or aupdate_state is None:
        return None

    config = {"configurable": {"thread_id": thread_id}}
    state = await aget_state(config)
    values = getattr(state, "values", None) or {}
    messages = values.get("messages") or []
    prior_event = values.get("_summarization_event")

    middleware = SummarizationMiddleware(
        model=model,
        backend=backend,
        keep=_MANUAL_KEEP,
        trim_tokens_to_summarize=None,
    )
    # The effective conversation is what the model would see now (a prior
    # summary plus the turns since), and the cutoff respects the keep window.
    effective = middleware._apply_event_to_messages(messages, prior_event)
    cutoff = middleware._determine_cutoff_index(effective)
    if cutoff <= 0:
        return None

    to_summarize, to_keep = middleware._partition_messages(effective, cutoff)
    summary = await middleware._acreate_summary(to_summarize)
    file_path = await middleware._aoffload_to_backend(backend, to_summarize)
    summary_msg = middleware._build_new_messages_with_path(summary, file_path)[0]
    state_cutoff = middleware._compute_state_cutoff(prior_event, cutoff)
    new_event: SummarizationEvent = {
        "cutoff_index": state_cutoff,
        "summary_message": summary_msg,
        "file_path": file_path,
    }
    await aupdate_state(config, {"_summarization_event": new_event})

    freed = max(
        0,
        int(count_tokens_approximately(to_summarize))
        - int(count_tokens_approximately([summary_msg])),
    )
    if trace is not None:
        trace.append(
            CONTEXT_COMPACTION,
            compaction_payload(
                summarized=len(to_summarize),
                kept=len(to_keep),
                offloaded=file_path is not None,
                manual=True,
            ),
        )
    return CompactResult(
        messages_summarized=len(to_summarize),
        messages_kept=len(to_keep),
        freed_tokens=freed,
        offloaded=file_path is not None,
    )
