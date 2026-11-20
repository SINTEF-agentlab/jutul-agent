"""Render benchmark results from eval logs into a markdown page.

The published artifact is a *snapshot*: each model runs the suite a few times
and the cells aggregate those runs (a 2/3 fraction is two passes of three
runs). Cross-run history beyond the snapshot is deliberately not shown;
skills, tasks, and the harness all move between snapshots, so older rows
rarely measure the same problem. The raw ``.eval`` logs stay local; the
rendered page and its JSON source are committed.

The committed ``docs/benchmark-records.jsonl`` is the snapshot's source data,
so the page can be regenerated, and extended with a new model, without the
original ``.eval`` logs: ``--records`` merges that file with fresh logs, with
runs accumulating (one record per sample-run, deduped on run and epoch).

Usage::

    uv run python -m makker.eval.report 2026-06-12T18 -o docs/benchmark.md
    uv run python -m makker.eval.report 2026-06-12T18 --json records.jsonl
    # add a model's fresh run to the published snapshot:
    uv run python -m makker.eval.report <new-prefix> \
        --records docs/benchmark-records.jsonl \
        --json docs/benchmark-records.jsonl -o docs/benchmark.md
"""

from __future__ import annotations

import argparse
import datetime as _dt
import io
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from makker.paths import state_home

# Prices in USD per million tokens, last reconciled on PRICES_AS_OF against
# provider / OpenRouter pricing pages. Cache write applies to Anthropic only
# (1.25x input for the 5-minute TTL); Gemini's cache-storage-per-hour fee is not
# modeled. A self-hosted model has no metered cost, so it is priced against a
# hosted reference (OpenRouter) to keep the cost column comparable. Update
# deliberately when providers reprice, and bump PRICES_AS_OF with it.
PRICES_AS_OF = "2026-06-15"
PRICES: dict[str, dict[str, float]] = {
    "gpt-5.4-mini": {"input": 0.75, "output": 4.50, "cache_read": 0.075, "cache_write": 0.0},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00, "cache_read": 0.10, "cache_write": 1.25},
    "gemini-3.1-flash-lite": {
        "input": 0.25,
        "output": 1.50,
        "cache_read": 0.025,
        "cache_write": 0.0,
    },
    # Self-hosted via Ollama; priced against OpenRouter for comparison
    # (qwen3.6-27b aggregate, 2026-06-15: $0.29 in / $3.17 out per MTok). Open
    # hosts give no prompt-cache discount, so cache_read = input, cache_write = 0
    # (and Ollama reports no cache tokens, so these never actually apply here).
    "qwen3.6:27b": {"input": 0.29, "output": 3.17, "cache_read": 0.29, "cache_write": 0.0},
}


@dataclass
class SampleRecord:
    run: str  # log timestamp prefix, identifies the run batch
    task: str
    sample: str
    simulator: str
    model: str
    verdict: str  # pass | fail | infra
    category: str  # pass | answer | mechanism | budget | infra
    failed_scorers: list[str]
    tokens_in: int
    tokens_out: int
    tokens_cache_read: int
    tokens_cache_write: int
    cost_usd: float | None
    wall_s: float | None
    messages: int
    commit: str
    epoch: int = 1  # which repetition of the sample this run is
    tool_calls: int = 0  # tool calls the agent made this run (harness efficiency)


def _model_short(model: str) -> str:
    return model.split("/")[-1]


def _model_sort_key(model: str) -> tuple[bool, str]:
    """Order models for display: self-hosted (ollama) last, else by name."""
    return (model.split("/")[0] == "ollama", model)


def _price(model: str, usage: dict[str, int]) -> float | None:
    prices = PRICES.get(_model_short(model))
    if prices is None:
        return None
    return (
        usage["in"] * prices["input"]
        + usage["out"] * prices["output"]
        + usage["cr"] * prices["cache_read"]
        + usage["cw"] * prices["cache_write"]
    ) / 1_000_000


def _classify(sample, scores: dict[str, str]) -> tuple[str, str, list[str]]:
    """(verdict, category, failed scorer names) for one sample.

    Only explicit ``INCORRECT`` ("I") scorers count as failures. The efficiency
    scorers (tool_call_count, …) report a numeric value, not pass/fail, so they
    must never be read as a failed check.
    """
    if sample.error is not None:
        return "infra", "infra", []
    failed = [name for name, value in scores.items() if value == "I"]
    if not failed:
        return "pass", "pass", []
    limit = getattr(sample, "limit", None)
    if limit is not None:
        return "fail", "budget", failed
    if all(name.startswith("julia_code_matches") for name in failed):
        return "fail", "mechanism", failed
    return "fail", "answer", failed


def _dedupe(records: list[SampleRecord]) -> list[SampleRecord]:
    """Drop exact duplicate executions, keeping every distinct run and epoch.

    Aggregation is across runs, so two runs of the same (task, sample, model)
    are both kept; a fraction like 2/3 means it passed two of three runs.
    The key includes run and epoch, so only the same execution read twice
    (overlapping log-prefix globs) collapses; a later read of the same key wins.
    """
    unique: dict[tuple[str, str, str, str, int], SampleRecord] = {}
    for r in records:
        unique[(r.task, r.sample, r.model, r.run, r.epoch)] = r
    return sorted(unique.values(), key=lambda r: (r.task, r.sample, r.model, r.run, r.epoch))


def load_records(path: Path) -> list[SampleRecord]:
    """Load records previously written with ``--json`` (one JSON object per line)."""
    return [
        SampleRecord(**json.loads(line))
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def collect(prefixes: str | list[str], log_dir: Path | None = None) -> list[SampleRecord]:
    """One record per sample-run (every epoch) from logs matching the prefixes."""
    from inspect_ai.log import read_eval_log

    if isinstance(prefixes, str):
        prefixes = [prefixes]
    log_dir = log_dir or (state_home() / "eval-logs")
    paths = sorted({p for prefix in prefixes for p in log_dir.glob(f"{prefix}*.eval")})
    records: list[SampleRecord] = []
    for path in paths:
        log = read_eval_log(path)
        run = path.name[:19]
        for s in log.samples or []:
            usage = {"in": 0, "out": 0, "cr": 0, "cw": 0}
            for mu in (s.model_usage or {}).values():
                usage["in"] += mu.input_tokens or 0
                usage["out"] += mu.output_tokens or 0
                usage["cr"] += mu.input_tokens_cache_read or 0
                usage["cw"] += mu.input_tokens_cache_write or 0
            tool_calls = sum(len(getattr(m, "tool_calls", None) or []) for m in (s.messages or []))
            scores = {k: str(v.value) for k, v in (s.scores or {}).items()}
            verdict, category, failed = _classify(s, scores)
            runconfig = (s.store or {}).get("jutul/runconfig") or {}
            # The commit lives under the ``makker`` block; a dirty working
            # tree is flagged so a snapshot built from one is never read as a
            # pristine commit (see docs/evaluation.md on dirty-tree runs).
            # Logs written before the rename keep theirs under the old name.
            jutul = runconfig.get("makker") or runconfig.get("jutul_agent") or {}
            commit = str(jutul.get("commit") or "")[:9]
            if commit and jutul.get("dirty"):
                commit += "-dirty"
            record = SampleRecord(
                run=run,
                task=log.eval.task.split("/")[-1],
                sample=str(s.id),
                simulator=str((s.metadata or {}).get("simulator", "")),
                model=log.eval.model,
                verdict=verdict,
                category=category,
                failed_scorers=failed,
                tokens_in=usage["in"],
                tokens_out=usage["out"],
                tokens_cache_read=usage["cr"],
                tokens_cache_write=usage["cw"],
                cost_usd=_price(log.eval.model, usage),
                wall_s=getattr(s, "total_time", None),
                messages=len(s.messages or []),
                commit=commit,
                epoch=getattr(s, "epoch", 1) or 1,
                tool_calls=tool_calls,
            )
            records.append(record)
    return _dedupe(records)


# Failure categories shown in the detail table; pass and infra handled separately.
_CATEGORY_LABEL = {
    "pass": "pass",
    "answer": "wrong answer",
    "mechanism": "serial / mechanism",
    "budget": "hit budget",
    "infra": "infra error",
}


def _suite(task: str) -> str:
    """Suite (module) a task belongs to: ``ensembles_fimbul`` -> ``ensembles``."""
    return task.split("_", 1)[0]


def _runs(records: list[SampleRecord]) -> int:
    """How many times the sample set ran (records per distinct sample, rounded)."""
    samples = len({(r.task, r.sample) for r in records})
    return round(len(records) / samples) if samples else 0


def _per_run_total(records: list[SampleRecord], attr: str) -> float | None:
    """One full pass's worth of ``attr``: per-sample mean across runs, summed.

    Gives the cost (or wall time) of running the suite once, averaged over
    however many times each sample ran, so it stays comparable whether a model
    ran once or three times. ``None`` when no record carries the value.
    """
    from collections import defaultdict

    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in records:
        value = getattr(r, attr)
        if value is not None:
            groups[(r.task, r.sample)].append(value)
    if not groups:
        return None
    return sum(sum(values) / len(values) for values in groups.values())


def _humanize_tokens(n: float) -> str:
    """Compact token count: 850 -> '850', 85_000 -> '85k', 4_600_000 -> '4.6M'."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}k"
    return f"{n:.0f}"


def _frac(passed: int, total: int) -> str:
    """A coloured ``passed/total`` cell (green all, red none, amber partial)."""
    if total == 0:
        return '<span class="bench-na">—</span>'
    cls = "bench-pass" if passed == total else ("bench-fail" if passed == 0 else "bench-partial")
    return f'<span class="{cls}">{passed}/{total}</span>'


def _matrix(w, records, models, short, row_key, row_label, rows) -> None:
    """A rows x models heatmap of coloured pass fractions, with a Total row."""
    w("| " + row_label + " | " + " | ".join(short[m] for m in models) + " |\n")
    w("|---" * (len(models) + 1) + "|\n")
    for row in rows:
        cells = []
        for m in models:
            rs = [r for r in records if row_key(r) == row and r.model == m and r.verdict != "infra"]
            cells.append(_frac(sum(r.verdict == "pass" for r in rs), len(rs)))
        # Plain text (not a code span) so single-word labels don't wrap mid-word.
        w(f"| {row} | " + " | ".join(cells) + " |\n")
    totals = []
    for m in models:
        done = [r for r in records if r.model == m and r.verdict != "infra"]
        totals.append(_frac(sum(r.verdict == "pass" for r in done), len(done)))
    w("| **all** | " + " | ".join(totals) + " |\n\n")


def render_markdown(records: list[SampleRecord]) -> str:
    models = sorted({r.model for r in records}, key=_model_sort_key)
    short = {m: _model_short(m) for m in models}
    today = _dt.date.today().isoformat()
    runs = sorted({r.run for r in records})
    commits = sorted({r.commit for r in records if r.commit})
    n_runs = max((_runs([r for r in records if r.model == m]) for m in models), default=1)
    times = "once" if n_runs <= 1 else f"{n_runs} times"
    out = io.StringIO()
    w = out.write
    w("# Benchmark results\n\n")
    w(
        f"Snapshot generated {today} from runs {runs[0]} … {runs[-1]}"
        + (f" (Makker {', '.join(commits)})" if commits else "")
        + ". Every sample runs the real agent end to end in a fresh workspace and "
        "is graded on the session trace as well as the answer. See "
        "[how evaluation works](evaluation.md). Each model ran the suite "
        f"**{times}**, and cells aggregate across runs, so a fraction like 2/3 means "
        "the sample passed two of three runs.\n\n"
    )

    # Overview: pass rate, cost, and wall time per model, all in one place.
    w("## Overview\n\n")
    w(
        "Pass rate is passing runs over runs that completed (infrastructure errors "
        "excluded). Tool calls and tokens are the per-run totals across the suite, "
        "the harness-efficiency signals: at equal pass rate, fewer means the harness "
        "got the agent there in less work. Input tokens note how many were served "
        "from the prompt cache (a cheap fraction of the input price); a model that "
        "caches aggressively processes a large input cheaply, which is why cost "
        "doesn't track raw token counts and is shown alongside them. "
        "Cost and wall time are for **one** pass over the suite (the "
        "per-run average), measured on a single machine. Within a model samples run "
        "one at a time, but wall time still depends on that machine and on how many "
        "models shared it during the run, so read it as indicative and comparable "
        "only within this snapshot; pass rate and cost are unaffected by either. "
        f"Dollar costs use provider prices as of {PRICES_AS_OF} (see "
        "`eval/report.py`) and include prompt-cache reads/writes; the self-hosted "
        "model is priced against a hosted reference.\n\n"
    )
    w(
        "| Model | Pass rate | Tool calls / run | Input tokens / run | "
        "Output tokens / run | Cost / run | Wall / run |\n"
    )
    w("|---|---|---|---|---|---|---|\n")
    for m in models:
        rs = [r for r in records if r.model == m]
        done = [r for r in rs if r.verdict != "infra"]
        passed = sum(r.verdict == "pass" for r in done)
        calls_run = _per_run_total(rs, "tool_calls")
        # Input is everything fed to the model (fresh input + cache reads +
        # cache writes). The cache-read share is annotated because it is billed
        # at a fraction of the input price, so a heavily cached model processes a
        # large input cheaply. Output is shown separately because it is priced highest.
        input_run = sum(
            _per_run_total(rs, attr) or 0
            for attr in ("tokens_in", "tokens_cache_read", "tokens_cache_write")
        )
        cached_run = _per_run_total(rs, "tokens_cache_read") or 0
        output_run = _per_run_total(rs, "tokens_out") or 0
        cost_run = _per_run_total(rs, "cost_usd")
        wall_run = _per_run_total(rs, "wall_s")
        calls = f"{calls_run:.0f}" if calls_run is not None else "—"
        input_tok = _humanize_tokens(input_run) if input_run else "—"
        if cached_run:
            input_tok += f" ({_humanize_tokens(cached_run)} cached, {cached_run / input_run:.0%})"
        output_tok = _humanize_tokens(output_run) if output_run else "—"
        cost = f"${cost_run:.2f}" if cost_run is not None else "—"
        wall = f"{wall_run / 3600:.1f} h" if wall_run else "—"
        w(
            f"| {short[m]} | {_frac(passed, len(done))} | {calls} | {input_tok} | "
            f"{output_tok} | {cost} | {wall} |\n"
        )
    w("\n")

    # Aggregations: by suite and by simulator.
    w("## By suite\n\n")
    suites = sorted({_suite(r.task) for r in records})
    _matrix(w, records, models, short, lambda r: _suite(r.task), "Suite", suites)

    w("## By simulator\n\n")
    w("Cross-cut of the same samples by the simulator they exercise ")
    w("(`general` = sim-agnostic tasks like canary, calibration, plotting).\n\n")
    sims = sorted({(r.simulator or "general") for r in records})
    _matrix(w, records, models, short, lambda r: r.simulator or "general", "Simulator", sims)

    # Full detail, folded away: one row per (sample, model) aggregated across
    # runs. markdown="1" lets md_in_html parse the table inside the raw
    # <details> block (without it the pipes render as text).
    w(
        '<details markdown="1">\n'
        "<summary>All samples (pass count, tool calls, tokens, cost, wall time)</summary>\n\n"
    )
    w(
        "| Suite | Sample | Sim | Model | Passed | Failures | "
        "Tool calls | Input | Output | Cost | Wall |\n"
    )
    w("|---|---|---|---|---|---|---|---|---|---|---|\n")
    keyed = sorted(
        {(r.task, r.sample, r.model) for r in records},
        key=lambda k: (_suite(k[0]), k[1], _model_sort_key(k[2])),
    )
    for task, sample, model in keyed:
        rs = [r for r in records if r.task == task and r.sample == sample and r.model == model]
        done = [r for r in rs if r.verdict != "infra"]
        passed = sum(r.verdict == "pass" for r in done)
        fails = sorted(
            {_CATEGORY_LABEL.get(r.category, r.category) for r in done if r.verdict != "pass"}
        )
        infra = len(rs) - len(done)
        notes = (
            ", ".join(fails) + (f"; {infra} infra" if infra else "") if (fails or infra) else "—"
        )
        calls_run = _per_run_total(rs, "tool_calls")
        input_run = sum(
            _per_run_total(rs, attr) or 0
            for attr in ("tokens_in", "tokens_cache_read", "tokens_cache_write")
        )
        output_run = _per_run_total(rs, "tokens_out") or 0
        cost_run = _per_run_total(rs, "cost_usd")
        wall_run = _per_run_total(rs, "wall_s")
        calls = f"{calls_run:.0f}" if calls_run is not None else "—"
        input_tok = _humanize_tokens(input_run) if input_run else "—"
        output_tok = _humanize_tokens(output_run) if output_run else "—"
        cost = f"${cost_run:.2f}" if cost_run is not None else "—"
        wall = f"{wall_run / 60:.0f} min" if wall_run else "—"
        w(
            f"| {_suite(task)} | `{sample}` | {rs[0].simulator or 'general'} | "
            f"{short[model]} | {_frac(passed, len(done))} | {notes} | "
            f"{calls} | {input_tok} | {output_tok} | {cost} | {wall} |\n"
        )
    w("\n</details>\n\n")

    w("## Reading the results\n\n")
    w(
        "A sample passes only when every scorer passes: the answer checks *and* "
        "the trace checks (the required mechanism appears in code the agent "
        "actually ran). Failures fall into:\n\n"
        "- **wrong answer**: the reported values failed the golden or structural check.\n"
        "- **serial / mechanism**: the answer may be right, but a required mechanism "
        "is missing from the trace (e.g. a sweep that ran serially when the prompt "
        "asked for a parallel ensemble).\n"
        "- **hit budget**: the sample reached its message or time cap before finishing.\n"
        "- **infra error**: the run failed before the agent could work (provider or "
        "harness error); excluded from pass rates, not a model result.\n\n"
        "Composite tasks are noisy at a single epoch, so each model runs the suite a "
        "few times and the cells aggregate the runs. Regenerate this page with:\n\n"
        "```sh\nuv run makker eval <suite> --model <provider/model> --epochs 3\n"
        "uv run python -m makker.eval.report <log-prefix> -o docs/benchmark.md\n```\n\n"
        "To add a model without re-running the others, merge the committed snapshot "
        "instead: pass `--records docs/benchmark-records.jsonl` and write it back with "
        "`--json docs/benchmark-records.jsonl`.\n"
    )
    return out.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "prefixes", nargs="*", help="eval-log filename prefix(es), e.g. 2026-06-12T18"
    )
    parser.add_argument(
        "--records",
        type=Path,
        action="append",
        help="merge records from a --json snapshot (repeatable); runs accumulate.",
    )
    parser.add_argument("-o", "--output", type=Path, help="write markdown here")
    parser.add_argument("--json", type=Path, help="write one JSON record per line here")
    parser.add_argument(
        "--log-dir",
        type=Path,
        help="directory to read .eval logs from (default: the Makker home eval-logs/)",
    )
    args = parser.parse_args()
    if not args.prefixes and not args.records:
        parser.error("give at least one log prefix or a --records file")
    records = collect(args.prefixes, log_dir=args.log_dir) if args.prefixes else []
    for path in args.records or []:
        records += load_records(path)
    records = _dedupe(records)
    if not records:
        print(f"no samples in logs matching {args.prefixes}", file=sys.stderr)
        return 1
    if args.json:
        args.json.write_text(
            "\n".join(json.dumps(asdict(r)) for r in records) + "\n", encoding="utf-8"
        )
    markdown = render_markdown(records)
    if args.output:
        args.output.write_text(markdown, encoding="utf-8")
        print(f"wrote {args.output} ({len(records)} records)")
    else:
        sys.stdout.buffer.write(markdown.encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
