"""Syntax-highlight code blocks for HTML transcripts (Pygments, inlined CSS)."""

from __future__ import annotations

import html
import re
from functools import lru_cache

from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import TextLexer, get_lexer_by_name, guess_lexer
from pygments.util import ClassNotFound

_LANG_ALIASES: dict[str, str] = {
    "jl": "julia",
    "py": "python",
    "md": "markdown",
    "sh": "bash",
    "yml": "yaml",
    "text": "text",
}

_FENCE_RE = re.compile(r"^```(\w*)\n?", re.MULTILINE)
_MD_CODE_BLOCK_RE = re.compile(
    r'<pre><code(?: class="language-([^"]*)")?>(.*?)</code></pre>',
    re.DOTALL | re.IGNORECASE,
)
PLAIN_OUTPUT_TOOLS = frozenset({"grep", "glob", "ls"})
SIMULATOR_TOOLS = frozenset({"run_julia", "plot_julia"})


def is_julia_source(text: str) -> bool:
    """Heuristic for Julia source / literate notebooks returned by tools."""
    stripped = text.strip()
    if not stripped:
        return False
    head = stripped[:800]
    if "using " in head or "function " in head:
        return True
    lines = [ln for ln in stripped.splitlines() if ln.strip()]
    if len(lines) < 3:
        return False
    sample = lines[: min(len(lines), 40)]
    comments = sum(1 for ln in sample if ln.lstrip().startswith("#"))
    codeish = sum(
        1
        for ln in sample
        if any(token in ln for token in ("=", "setup_", "simulate_", "end", "bar"))
    )
    return comments >= 2 and codeish >= 2


@lru_cache(maxsize=1)
def pygments_css() -> str:
    """CSS for ``.highlight`` blocks, tuned for light and dark backgrounds."""
    light = HtmlFormatter(style="friendly", cssclass="highlight", nowrap=False)
    dark = HtmlFormatter(style="native", cssclass="highlight")
    light_defs = re.sub(
        r"\.highlight\s*\{[^}]+\}",
        "",
        light.get_style_defs(".highlight"),
    )
    dark_defs = re.sub(
        r"\.highlight\s*\{[^}]+\}",
        "",
        dark.get_style_defs(".highlight"),
    )
    return (
        light_defs
        + "\n@media (prefers-color-scheme: dark) {\n"
        + dark_defs
        + "\n}\n"
        + """
.highlight {
  margin: 0.5rem 0;
  border: 1px solid var(--border);
  border-radius: 8px;
  overflow-x: auto;
  background: var(--code-bg) !important;
}
.highlight pre {
  margin: 0;
  padding: 0.75rem 0.85rem;
  background: transparent !important;
  border: none;
  border-radius: 0;
  line-height: 1.45;
  font-size: 0.84rem;
}
"""
    )


def _resolve_lexer(language: str | None, code: str):
    if language:
        lang = _LANG_ALIASES.get(language.lower(), language.lower())
        try:
            return get_lexer_by_name(lang)
        except ClassNotFound:
            pass
    try:
        return guess_lexer(code)
    except ClassNotFound:
        return TextLexer()


def highlight_code(code: str, language: str | None = None) -> str:
    """Return a ``<div class="highlight">`` fragment for ``code``."""
    lexer = _resolve_lexer(language, code)
    formatter = HtmlFormatter(
        cssclass="highlight",
        nowrap=False,
        wrapcode=True,
    )
    return highlight(code.rstrip("\n"), lexer, formatter)


def highlight_code_blocks_in_html(fragment: str) -> str:
    """Replace markdown ``<pre><code>`` blocks with Pygments output."""

    def _replace(match: re.Match[str]) -> str:
        language = match.group(1) or None
        code = html.unescape(match.group(2))
        return highlight_code(code, language)

    return _MD_CODE_BLOCK_RE.sub(_replace, fragment)


def infer_tool_code_language(text: str, tool_name: str | None = None) -> str | None:
    if tool_name in PLAIN_OUTPUT_TOOLS:
        return "text"
    if tool_name in SIMULATOR_TOOLS:
        return "julia"
    if tool_name == "read_file" and is_julia_source(text):
        return "julia"
    fence = _FENCE_RE.search(text)
    if fence and fence.group(1):
        return fence.group(1)
    if is_julia_source(text):
        return "julia"
    if text.lstrip().startswith(("{", "[")):
        return "json"
    return "text"
