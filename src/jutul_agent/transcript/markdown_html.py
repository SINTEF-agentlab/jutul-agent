"""Convert markdown prose to HTML for transcript renderers."""

from __future__ import annotations

import re
from functools import lru_cache

from markdown_it import MarkdownIt

from jutul_agent.agent.tool_output import READ_FILE_GUTTER, strip_read_file_gutter

_MD_MARKERS = re.compile(
    r"(^|\n)(#{1,6}\s|\*\*[^*]+\*\*|```|^\s*[-*+]\s|^\s*\d+\.\s|\|.+\|)",
    re.MULTILINE,
)
_YAML_FRONTMATTER = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)


@lru_cache(maxsize=1)
def _markdown_renderer() -> MarkdownIt:
    # html=False escapes any raw HTML in the source rather than passing it
    # through. The prose we render is untrusted (LLM output, tool results, files
    # the agent read), so a literal ``<script>`` or ``<img onerror=...>`` must
    # show as text, never execute. Markdown syntax (headings, tables, code,
    # links, images) is unaffected. Intentional HTML goes through the report's
    # explicit ``html`` block instead, which is backed by a strict CSP.
    return MarkdownIt("gfm-like", {"html": False})


def looks_like_markdown(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if "```" in stripped or "**" in stripped:
        return True
    if _MD_MARKERS.search(stripped):
        return True
    return stripped.startswith("# ") and "\n" in stripped


def strip_line_number_prefixes(text: str) -> str:
    """Remove the line-number gutter from ``read_file`` tool output.

    Gated on most of the opening lines carrying a gutter, because this runs on
    stored transcript text where the tool name may be unreliable and a file of
    numbered prose should be left alone.
    """
    lines = text.splitlines()
    if len(lines) < 2:
        return text
    sample = lines[: min(len(lines), 24)]
    numbered = sum(1 for ln in sample if READ_FILE_GUTTER.match(ln))
    if numbered < max(3, len(sample) // 2):
        return text
    return strip_read_file_gutter(text)


def strip_yaml_frontmatter(text: str) -> str:
    """Drop YAML frontmatter blocks from skill files and similar markdown."""
    return _YAML_FRONTMATTER.sub("", text, count=1)


def render_markdown_html(text: str) -> str:
    """Render markdown to an HTML fragment (no wrapper element)."""
    return _markdown_renderer().render(text.strip())
