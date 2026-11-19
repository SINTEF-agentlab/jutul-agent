"""Best-effort, content-aware session titles.

A session is named from its first prompt immediately (``derive_session_title``)
so it is never nameless in a history list. After the first turn this asks the
model for a short title that reflects what the exchange was actually about and
replaces the first-prompt one. It is strictly best-effort: no API key, an
offline model, or a slow/empty reply just keeps the first-prompt title, so the
feature can never make naming worse.
"""

from __future__ import annotations

import re
from typing import Any

# Tight instruction so cheap/fast models still return something usable; the
# reply is sanitised below regardless of how well a given model follows it.
_SYSTEM = (
    "You label chat sessions. Read the exchange and reply with a short title of "
    "three to six words capturing the task. Use Title Case, no surrounding quotes, "
    "no trailing punctuation. Reply with the title only."
)
_MAX_CHARS = 60


async def generate_session_title(model_id: str | None, conversation: str) -> str | None:
    """A short title for ``conversation`` from the model, or ``None`` on any failure.

    ``model_id`` is the session's model spec (``provider:model``); ``conversation``
    is a small excerpt (the first user prompt and the reply). Everything is wrapped
    so a caller can fire-and-forget without guarding: a failure returns ``None``.
    """
    if not model_id or not conversation.strip():
        return None
    try:
        from langchain.chat_models import init_chat_model

        model = init_chat_model(model_id)
        resp = await model.ainvoke(
            [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": conversation[:4000]},
            ]
        )
    except Exception:
        return None
    return _clean(getattr(resp, "content", resp))


def _clean(content: Any) -> str | None:
    """Reduce a model reply to one tidy title line, or ``None`` if nothing usable."""
    from makker.session import truncate_title
    from makker.trace.messages import content_to_str

    text = content if isinstance(content, str) else content_to_str(content)
    first = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    first = re.sub(r"\s+", " ", first).strip().strip("\"'").rstrip(".").strip()
    if not first:
        return None
    return truncate_title(first, _MAX_CHARS)
