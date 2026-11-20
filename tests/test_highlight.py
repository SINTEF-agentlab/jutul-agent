"""Tests for syntax highlighting helpers."""

from __future__ import annotations

from makker.transcript.highlight import (
    highlight_code,
    highlight_code_blocks_in_html,
    infer_tool_code_language,
    is_julia_source,
)


def test_highlight_julia_code() -> None:
    html = highlight_code("using CairoMakie\nx = 1", "julia")
    assert 'class="highlight"' in html
    assert "using" in html
    assert "<span" in html


def test_highlight_markdown_code_fence() -> None:
    fragment = '<pre><code class="language-julia">a = 1</code></pre>'
    html = highlight_code_blocks_in_html(fragment)
    assert 'class="highlight"' in html
    assert "<span" in html


def test_infer_julia_from_tool_name() -> None:
    assert infer_tool_code_language("x = 1", "run_julia") == "julia"


def test_infer_plaintext_for_grep() -> None:
    assert infer_tool_code_language("examples/foo.jl", "grep") == "text"


def test_is_julia_source() -> None:
    text = "# comment\nusing Jutul\nProd = setup_vertical_well(domain, 1, 1)\n"
    assert is_julia_source(text)
