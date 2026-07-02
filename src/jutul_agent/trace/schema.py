"""Single home for the trace event vocabulary.

Writers and readers import these constants instead of restating string
literals, so a typo is an ImportError instead of a silently dropped or
never-rendered event. Payload shapes are documented in docs/trace.md
(kept additive: new fields and kinds only, no migrations); the payload
constructors below cover the two shapes built in several places.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

SESSION_START = "session_start"
SESSION_RESUME = "session_resume"
SESSION_TITLE = "session_title"
SESSION_END = "session_end"

MESSAGE_USER = "message_user"
MESSAGE_ASSISTANT = "message_assistant"
MESSAGE_REASONING = "message_reasoning"
MODEL_USAGE = "model_usage"

TOOL_CALL = "tool_call"
TOOL_RESULT = "tool_result"
HITL_REQUEST = "hitl_request"
HITL_RESPONSE = "hitl_response"
CONTEXT_COMPACTION = "context_compaction"

ARTIFACT = "artifact"
ATTEMPT = "attempt"
UPLOAD = "upload"
UI_EVENT = "ui_event"

EVAL_TARGET = "eval_target"
EVAL_RESULT = "eval_result"

ALL_KINDS: frozenset[str] = frozenset(
    {
        SESSION_START,
        SESSION_RESUME,
        SESSION_TITLE,
        SESSION_END,
        MESSAGE_USER,
        MESSAGE_ASSISTANT,
        MESSAGE_REASONING,
        MODEL_USAGE,
        TOOL_CALL,
        TOOL_RESULT,
        HITL_REQUEST,
        HITL_RESPONSE,
        CONTEXT_COMPACTION,
        ARTIFACT,
        ATTEMPT,
        UPLOAD,
        UI_EVENT,
        EVAL_TARGET,
        EVAL_RESULT,
    }
)


def compaction_payload(
    *,
    summarized: int | None,
    kept: int | None,
    offloaded: bool,
    manual: bool = False,
) -> dict[str, Any]:
    """Payload for a ``CONTEXT_COMPACTION`` event.

    ``summarized``/``kept`` are message counts (None when a writer cannot
    know one); ``offloaded`` says whether the summarized turns were saved
    to a recoverable file.
    """
    return {"summarized": summarized, "kept": kept, "offloaded": offloaded, "manual": manual}


def artifact_payload(
    *,
    path: str,
    mime: str,
    caption: str,
    tool_call_id: str | None = None,
    format: str | None = None,
    kind: str | None = None,
    size_px: Sequence[int] | None = None,
    dpi: int | None = None,
    slot: str | None = None,
    source_code: str | None = None,
    poster: str | None = None,
    live_url: str | None = None,
) -> dict[str, Any]:
    """Payload for an ``ARTIFACT`` event: one field set for every writer.

    ``path`` is relative to the session output dir. ``kind`` routes the
    browser viz ("plot", "report"); ``poster`` and ``live_url`` carry the
    interactive plot's thumbnail and live view when they exist.
    """
    return {
        "path": path,
        "mime": mime,
        "caption": caption,
        "tool_call_id": tool_call_id,
        "format": format,
        "kind": kind,
        "size_px": list(size_px) if size_px is not None else None,
        "dpi": dpi,
        "slot": slot,
        "source_code": source_code,
        "poster": poster,
        "live_url": live_url,
    }
