"""Normalize streamed tool output for display and interrupt detection."""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import ToolMessage

# Some langgraph paths stringify ``[ToolMessage(content='...', ...)]``
# instead of returning the structured object. The regex below pulls the
# original content back out of that repr.
_TOOL_MESSAGE_CONTENT = re.compile(
    r"content=(?P<quote>['\"])(?P<body>(?:\\.|(?!\1).)*)\1",
    re.DOTALL,
)


def normalize_tool_output(value: Any) -> str:
    """Return human-readable tool output for display."""

    if value is None:
        return ""
    if isinstance(value, ToolMessage):
        return normalize_tool_output(value.content)
    if isinstance(value, list):
        if value and all(isinstance(item, ToolMessage) for item in value):
            parts = [normalize_tool_output(item) for item in value]
            return "\n".join(part for part in parts if part)
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, str):
        if is_interrupt_payload(value):
            return value
        if "ToolMessage(content=" in value:
            extracted = _extract_tool_messages_from_repr(value)
            if extracted:
                return extracted
        return value
    return str(value)


def is_interrupt_payload(text: str) -> bool:
    lowered = text.strip().lower()
    return lowered.startswith("interrupt(") or "interrupt(value=" in lowered


# One ``read_file`` gutter row: a right-justified line marker, optionally with a
# ``.<n>`` suffix marking the continuation of an over-long line, then the
# separator. Two spaces is the separator the framework emits today; the tab is
# the older ``cat -n`` form, which stored transcripts still carry. The separator
# belongs to the framework and has changed before, so this pattern lives in one
# place and every surface that shows ``read_file`` output strips through it.
READ_FILE_GUTTER = re.compile(r"^ *\d+(?:\.\d+)?(?:\t|  )")


def strip_read_file_gutter(text: str) -> str:
    """Drop the line-number gutter from each ``read_file`` row that carries one."""

    return "\n".join(READ_FILE_GUTTER.sub("", line, count=1) for line in text.splitlines())


def _extract_tool_messages_from_repr(text: str) -> str:
    """Pull content strings back out of a ``[ToolMessage(content='...', …)]`` repr."""

    parts: list[str] = []
    for match in _TOOL_MESSAGE_CONTENT.finditer(text):
        body = match.group("body")
        quote = match.group("quote")
        if quote == "'":
            body = body.replace("\\'", "'").replace("\\n", "\n").replace("\\t", "\t")
        else:
            body = body.replace('\\"', '"').replace("\\n", "\n").replace("\\t", "\t")
        parts.append(body)
    return "\n".join(parts)
