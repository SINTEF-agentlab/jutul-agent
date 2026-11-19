"""Helpers for working with LangChain message content.

LangChain messages may carry their content as a plain string or as a list
of structured parts. ``content_to_str`` projects the prose; ``reasoning_to_str``
projects just the reasoning blocks.
"""

from __future__ import annotations

from typing import Any


def content_to_str(content: Any) -> str:
    """Flatten message content to plain prose; non-text parts are dropped."""

    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
            continue
        if isinstance(part, dict) and part.get("type") == "text":
            text = part.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts)


def reasoning_to_str(content: Any) -> str:
    """Flatten reasoning blocks to plain text, separate from assistant prose."""

    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for part in content:
        if not isinstance(part, dict) or part.get("type") != "reasoning":
            continue
        reasoning = part.get("reasoning") or part.get("text")
        if isinstance(reasoning, str):
            parts.append(reasoning)
    return "\n".join(parts)
