"""Tiny display-policy helpers shared by approval and tool rendering.

Also the single home for the streamed tool-output display policy: every
surface accumulates a bounded tail of the raw stream and re-renders it through
the terminal emulator on a throttle, so a progress bar's carriage returns
collapse to one updating block and a chatty tool cannot burn the event loop.
The scheduling machinery differs per surface (a Textual timer in the TUI, an
asyncio task on the server); the policy numbers must not.
"""

from __future__ import annotations

import re

# Largest raw-stream tail kept per tool call for live re-rendering.
TOOL_STREAM_TAIL_CAP = 256 * 1024
# Minimum seconds between re-renders of one tool's stream.
TOOL_STREAM_RENDER_INTERVAL = 0.1

_MAX_PREVIEW_LINES = 120
_MAX_PREVIEW_CHARS = 4000


def truncate_preview(
    text: str,
    *,
    max_lines: int = _MAX_PREVIEW_LINES,
    max_chars: int = _MAX_PREVIEW_CHARS,
    marker: str = "\n... [preview truncated]",
) -> str:
    """Trim ``text`` to fit within ``max_lines``/``max_chars`` for inline previews."""

    lines = text.splitlines()
    truncated = False

    if len(lines) > max_lines:
        lines = lines[:max_lines]
        truncated = True

    rendered = "\n".join(lines)
    if len(rendered) > max_chars:
        rendered = rendered[:max_chars].rstrip()
        truncated = True

    if truncated:
        rendered += marker
    return rendered


def fenced_block(text: str, *, language: str = "") -> str:
    """Wrap ``text`` in a backtick fence long enough to survive embedded backticks."""

    runs = re.findall(r"`+", text)
    fence = "`" * max(3, max((len(run) for run in runs), default=0) + 1)
    head = f"{fence}{language}" if language else fence
    return f"{head}\n{text}\n{fence}"


def shorten(text: str, limit: int) -> str:
    """Trim ``text`` to ``limit`` characters, appending ``...`` if it was clipped."""

    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def shorten_single_line(text: str, limit: int) -> str:
    """Collapse whitespace and trim to ``limit`` characters."""

    return shorten(" ".join(text.split()), limit)
