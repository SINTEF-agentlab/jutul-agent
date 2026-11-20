"""Unit tests for eval scorers whose logic is worth pinning directly."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("inspect_ai", reason="eval extra not installed")

from inspect_ai.scorer import CORRECT, INCORRECT

from makker.eval.scorers import reads_digit


def _score(scorer, completion: str):
    state = SimpleNamespace(output=SimpleNamespace(completion=completion))
    return asyncio.run(scorer(state, None)).value


def test_reads_digit_matches_the_reported_digit() -> None:
    s = reads_digit("7")
    assert _score(s, "7") == CORRECT
    assert _score(s, "The numeral is **7**.") == CORRECT
    assert _score(s, "It looks like a 3 to me") == INCORRECT
    assert _score(s, "no digits here") == INCORRECT


def test_reads_digit_ignores_multi_digit_recipe_values() -> None:
    s = reads_digit("7")
    # The plot recipe (600x600 figure, markersize 18) must not satisfy the
    # check on its own: only a standalone single digit counts.
    assert _score(s, "I used a 600x600 figure with markersize 18.") == INCORRECT
    assert _score(s, "600x600, markersize 18, and the digit is 7.") == CORRECT
