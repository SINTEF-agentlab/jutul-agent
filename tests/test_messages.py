"""Tests for message content helpers."""

from __future__ import annotations

from makker.trace.messages import content_to_str, reasoning_to_str


def test_content_to_str_plain_string() -> None:
    assert content_to_str("hello") == "hello"


def test_content_to_str_list_of_text_parts() -> None:
    content = [{"type": "text", "text": "line one"}, {"type": "text", "text": "line two"}]
    assert content_to_str(content) == "line one\nline two"


def test_content_to_str_mixed_list_keeps_text_only() -> None:
    content = [
        {"type": "text", "text": "answer"},
        {"type": "reasoning", "reasoning": "thinking"},
    ]
    assert content_to_str(content) == "answer"


def test_reasoning_to_str_extracts_reasoning_blocks() -> None:
    content = [
        {"type": "reasoning", "reasoning": "step one"},
        {"type": "text", "text": "answer"},
        {"type": "reasoning", "text": "step two"},
    ]
    assert reasoning_to_str(content) == "step one\nstep two"


def test_reasoning_to_str_non_list_returns_empty() -> None:
    assert reasoning_to_str("plain") == ""
