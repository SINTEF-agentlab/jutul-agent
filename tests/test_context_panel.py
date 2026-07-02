"""Tests for context usage rendering and the /context command."""

from __future__ import annotations

from jutul_agent.context_panel import (
    format_tokens,
    render_context_panel,
    status_label,
    usage_alert,
)

_USAGE = {
    "input_tokens": 28_100,
    "output_tokens": 1_300,
    "total_tokens": 29_400,
    "input_token_details": {"cache_read": 12_000},
    "output_token_details": {"reasoning": 200},
}


def test_format_tokens_scales() -> None:
    assert format_tokens(950) == "950"
    assert format_tokens(1_400) == "1.4k"
    assert format_tokens(29_400) == "29k"
    assert format_tokens(1_200_000) == "1.2M"


def test_status_label_and_alert() -> None:
    assert status_label(None, 400_000) is None
    assert status_label(_USAGE, 400_000) == "ctx 7%"
    assert status_label(_USAGE, None) == "ctx 29k"
    assert usage_alert(_USAGE, 400_000) == "ok"
    assert usage_alert({"input_tokens": 300_000, "output_tokens": 0}, 400_000) == "warn"
    assert usage_alert({"input_tokens": 390_000, "output_tokens": 0}, 400_000) == "high"


def test_render_context_panel_categories() -> None:
    body = render_context_panel(
        model_label="openai:gpt-5.4-mini",
        usage=_USAGE,
        window=400_000,
        first_usage={"input_tokens": 5_300, "output_tokens": 100},
        model_calls=7,
        system_prompt_tokens=2_100,
        memory_index_tokens=300,
        memory_notes=3,
        compact_trigger_tokens=320_000,
        clear_trigger_tokens=240_000,
    )
    assert "7%" in body
    assert "29k of 400k tokens" in body
    assert "Estimated usage by category:" in body
    # The measured first call split into approximate components.
    assert "system prompt: ~2.1k (0.5%)" in body
    assert "memory index: ~300 (0.1%) — 3 notes load on demand" in body
    assert "tools, skills & framework: ~2.9k (0.7%)" in body  # 5300 - 2100 - 300
    assert "conversation: ~24k (6.0%)" in body  # 29400 - 5300
    # Free space counts up to the compaction trigger; the rest is the buffer.
    assert "free space: 291k" in body  # 320000 - 29400
    # Clearing fires before summarization, and the panel says so.
    assert "old tool results start clearing at 240k (before any summary)" in body
    assert "auto-compact buffer: 80k (20.0%) — summarization triggers at 320k" in body
    assert "Last call: input 28k (cache-read 12k) · output 1.3k (reasoning 200)" in body
    assert "Conversation growth: +24k over 7 model calls" in body
    assert "`/compact`" in body


def test_render_context_panel_without_component_estimates() -> None:
    body = render_context_panel(
        model_label=None,
        usage=_USAGE,
        window=400_000,
        first_usage={"input_tokens": 5_300},
        model_calls=3,
        compact_trigger_tokens=320_000,
    )
    assert "fixed per call: 5.3k" in body
    assert "system prompt" not in body  # no estimates → no made-up breakdown
    assert "conversation: ~24k" in body
    assert "free space: 291k" in body


def test_render_context_panel_trigger_without_window() -> None:
    body = render_context_panel(
        model_label="m",
        usage=_USAGE,
        window=None,
        first_usage={"input_tokens": 5_300},
        model_calls=2,
        compact_trigger_tokens=100_000,
    )
    assert "window size unknown" in body
    assert "auto-compact triggers at 100k tokens" in body
    assert "- free space:" not in body  # needs a window to mean anything


def test_render_context_panel_without_data() -> None:
    body = render_context_panel(model_label=None, usage=None, window=None)
    assert "No model calls yet" in body
    body = render_context_panel(model_label="m", usage=_USAGE, window=None)
    assert "window size unknown" in body


def test_render_context_panel_marks_post_compaction_estimate() -> None:
    estimated_usage = {"input_tokens": 17_000, "output_tokens": 0, "estimated": True}
    body = render_context_panel(
        model_label="anthropic:claude-haiku-4-5",
        usage=estimated_usage,
        window=200_000,
        first_usage={"input_tokens": 8_500},
        model_calls=17,
        compact_trigger_tokens=160_000,
    )
    assert "17k of 200k" in body  # the reduced figure, not the stale 31k
    assert "estimated after compaction" in body
    # Growth-over-model-calls compares to a pre-compaction baseline → suppressed.
    assert "Conversation growth" not in body
