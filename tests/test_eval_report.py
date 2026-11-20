"""Unit tests for the benchmark report renderer (no logs needed)."""

from __future__ import annotations

import pytest

pytest.importorskip("inspect_ai", reason="eval extra not installed")

from makker.eval.report import PRICES, SampleRecord, _classify, _price, render_markdown


def test_classify_ignores_numeric_efficiency_scorers() -> None:
    """Efficiency scorers report a number, not pass/fail — never a failed check."""
    from types import SimpleNamespace

    ok = SimpleNamespace(error=None, limit=None)
    # A passing answer plus numeric efficiency scorers ("5.0", "2.0") is a pass.
    verdict, _category, failed = _classify(
        ok, {"includes": "C", "tool_call_count": "5.0", "file_op_count": "2.0"}
    )
    assert verdict == "pass"
    assert failed == []
    # An explicit INCORRECT still fails; the numeric scorer stays out of `failed`.
    verdict, _category, failed = _classify(ok, {"numeric_close": "I", "tool_call_count": "9.0"})
    assert verdict == "fail"
    assert failed == ["numeric_close"]


def _record(**overrides) -> SampleRecord:
    base = dict(
        run="2026-06-12T18-41-16",
        task="ensembles",
        sample="ens-jutuldarcy-parallel-sweep",
        simulator="jutuldarcy",
        model="openai/gpt-5.4-mini",
        verdict="pass",
        category="pass",
        failed_scorers=[],
        tokens_in=8_000,
        tokens_out=400,
        tokens_cache_read=50_000,
        tokens_cache_write=0,
        cost_usd=0.012,
        wall_s=240.0,
        messages=12,
        commit="4b9ab20",
    )
    base.update(overrides)
    return SampleRecord(**base)


def test_render_has_overview_suite_and_simulator_sections() -> None:
    records = [
        _record(),
        _record(
            model="google/gemini-3.1-flash-lite",
            verdict="fail",
            category="mechanism",
            failed_scorers=["julia_code_matches"],
            cost_usd=0.15,
        ),
    ]
    page = render_markdown(records)
    assert "## Overview" in page
    assert "## By suite" in page
    assert "## By simulator" in page
    # Status is carried by coloured spans, with the per-sample failure
    # category shown in the detail table.
    assert 'class="bench-pass"' in page
    assert 'class="bench-fail"' in page
    assert "serial / mechanism" in page
    assert "$0.15" in page


def test_humanize_tokens_scales_units() -> None:
    from makker.eval.report import _humanize_tokens

    assert _humanize_tokens(850) == "850"
    assert _humanize_tokens(85_000) == "85k"
    assert _humanize_tokens(4_600_000) == "4.6M"


def test_overview_shows_input_with_cache_annotation_and_output() -> None:
    # Input = in 8000 + cache_read 50000 + cache_write 0 = 58k, of which 50k (86%)
    # cached. Output = 400.
    page = render_markdown([_record()])
    assert "Input tokens / run" in page
    assert "Output tokens / run" in page
    assert "58k (50k cached, 86%)" in page
    assert "| 400 |" in page


def test_suite_grouping_collapses_task_variants() -> None:
    from makker.eval.report import _suite

    assert _suite("ensembles_fimbul") == "ensembles"
    assert _suite("usage_battmo") == "usage"
    assert _suite("plotting_vision") == "plotting"
    assert _suite("canary") == "canary"


def test_load_records_round_trips_a_json_snapshot(tmp_path) -> None:
    import json
    from dataclasses import asdict

    from makker.eval.report import load_records

    records = [_record(), _record(model="ollama/qwen3", run="2026-06-14T10-00-00")]
    snapshot = tmp_path / "records.jsonl"
    snapshot.write_text("\n".join(json.dumps(asdict(r)) for r in records) + "\n", encoding="utf-8")
    loaded = load_records(snapshot)
    assert [r.model for r in loaded] == [r.model for r in records]
    assert loaded[0] == records[0]


def test_dedupe_keeps_distinct_runs_but_drops_exact_duplicates() -> None:
    from makker.eval.report import _dedupe

    run_a = _record(run="2026-06-13T00-00-00", verdict="fail", category="answer")
    run_b = _record(run="2026-06-14T00-00-00", verdict="pass", category="pass")
    duplicate = _record(run="2026-06-14T00-00-00", verdict="pass", category="pass")
    # Two distinct runs of the same key are both kept (so they can aggregate);
    # the exact same execution read twice collapses to one.
    deduped = _dedupe([run_a, run_b, duplicate])
    assert len(deduped) == 2


def test_overview_pass_rate_aggregates_across_runs() -> None:
    records = [
        _record(run="r1", epoch=1, verdict="pass", category="pass"),
        _record(run="r1", epoch=2, verdict="fail", category="answer"),
        _record(run="r1", epoch=3, verdict="pass", category="pass"),
    ]
    page = render_markdown(records)
    assert "3 times" in page  # ran the suite three times
    assert 'class="bench-partial">2/3' in page  # two of three runs passed


def test_models_order_self_hosted_last() -> None:
    records = [
        _record(model="ollama/qwen3.6:27b"),
        _record(model="openai/gpt-5.4-mini"),
    ]
    page = render_markdown(records)
    assert page.index("gpt-5.4-mini") < page.index("qwen3.6:27b")


def test_price_uses_cache_read_rate() -> None:
    usage = {"in": 1_000_000, "out": 0, "cr": 1_000_000, "cw": 0}
    cost = _price("anthropic/claude-haiku-4-5", usage)
    prices = PRICES["claude-haiku-4-5"]
    assert cost == pytest.approx(prices["input"] + prices["cache_read"])


def test_unknown_model_has_no_price() -> None:
    assert _price("ollama/qwen3", {"in": 1, "out": 1, "cr": 0, "cw": 0}) is None
