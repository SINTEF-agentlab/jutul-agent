"""Tests for markdown rendering helpers."""

from __future__ import annotations

from jutul_agent.agent.tool_output import READ_FILE_GUTTER
from jutul_agent.transcript.markdown_html import (
    looks_like_markdown,
    render_markdown_html,
    strip_line_number_prefixes,
    strip_yaml_frontmatter,
)


def test_looks_like_markdown() -> None:
    assert looks_like_markdown("### Heading")
    assert looks_like_markdown("plain **bold** text")
    assert not looks_like_markdown("just one line")


def test_render_markdown_html() -> None:
    html = render_markdown_html("**bold**")
    assert "<strong>bold</strong>" in html


def test_render_markdown_escapes_raw_html() -> None:
    """Prose is untrusted, so raw HTML must be shown as text, never passed
    through where it could execute or load a remote resource."""
    for raw in (
        "<script>alert('x')</script>",
        "<img src=x onerror=\"fetch('https://evil')\">",
        '<iframe src="https://evil"></iframe>',
        '<a href="javascript:alert(1)">x</a>',
    ):
        out = render_markdown_html(f"before\n\n{raw}\n")
        tag = raw.split(">", 1)[0].split(" ", 1)[0] + ">"  # e.g. "<script>"
        assert tag not in out, f"raw {tag!r} leaked through markdown"
        assert "&lt;" in out  # it was escaped to text instead


def test_render_markdown_keeps_real_markdown() -> None:
    # Disabling raw HTML must not weaken ordinary markdown rendering.
    assert "<h2>" in render_markdown_html("## Heading")
    assert "<table>" in render_markdown_html("| a | b |\n|---|---|\n| 1 | 2 |\n")
    assert "<code>" in render_markdown_html("`x = 1`")


def test_strip_line_number_prefixes() -> None:
    # The legacy `cat -n` gutter, which older stored transcripts still carry.
    text = "1\t---\n2\tname: wells\n3\tdescription: test\n"
    assert strip_line_number_prefixes(text).startswith("---\nname: wells")

    padded = "     1\t---\n     2\tname: wells\n     3\tdescription: test\n"
    assert strip_line_number_prefixes(padded).startswith("---\nname: wells")


def test_strip_line_number_prefixes_matches_the_real_read_file_gutter() -> None:
    """Strip the gutter the framework actually emits, not a hand-written copy.

    The separator is the framework's to choose and it has changed before; a
    hand-written sample keeps passing while the renderer silently stops
    stripping, so build the input with the same formatter `read_file` uses.
    """
    from deepagents.backends.utils import format_content_with_line_numbers

    lines = ["---", "name: wells", "description: test", "---", "", "# Title"]
    numbered = format_content_with_line_numbers("\n".join(lines))
    assert strip_line_number_prefixes(numbered).splitlines() == lines

    # Over-long lines are split into `<n>.<chunk>` continuation rows.
    wide = format_content_with_line_numbers(["x" * 5000, "short", "also short"])
    assert not any(
        READ_FILE_GUTTER.match(ln) for ln in strip_line_number_prefixes(wide).splitlines()
    )


def test_strip_yaml_frontmatter() -> None:
    text = "---\nname: wells\ndescription: test\n---\n\n# Title\n"
    assert strip_yaml_frontmatter(text).startswith("# Title")
