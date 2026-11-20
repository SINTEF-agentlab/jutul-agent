"""Tests for the recently-used models (MRU) store."""

from __future__ import annotations

from makker.recent_models import (
    RECENT_LIMIT,
    load_recent_models,
    record_recent_model,
)

# state_home is redirected to a tmp dir per test by the autouse conftest fixture,
# so each test starts with an empty recents file.


def test_starts_empty() -> None:
    assert load_recent_models() == []


def test_records_most_recent_first() -> None:
    record_recent_model("openai:gpt-5.5")
    record_recent_model("anthropic:claude-sonnet-4-6")
    assert load_recent_models() == ["anthropic:claude-sonnet-4-6", "openai:gpt-5.5"]


def test_existing_entry_moves_to_front_without_duplicating() -> None:
    record_recent_model("a:1")
    record_recent_model("b:2")
    record_recent_model("a:1")
    recents = load_recent_models()
    assert recents[0] == "a:1"
    assert recents.count("a:1") == 1


def test_capped_at_limit() -> None:
    for i in range(RECENT_LIMIT + 3):
        record_recent_model(f"p:{i}")
    assert len(load_recent_models()) == RECENT_LIMIT


def test_empty_id_ignored() -> None:
    record_recent_model("")
    assert load_recent_models() == []
