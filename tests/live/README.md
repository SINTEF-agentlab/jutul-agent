# Live tests

These tests call a **real LLM provider** and a **real Julia subprocess**. They are
**not** collected by the default `pytest` run.

## When to run

Run manually when validating that makker works end-to-end against a live
model — not during routine agent development loops.

```bash
pytest tests/live/
```

Override the model (optional):

```bash
MAKKER_LIVE_MODEL=anthropic:claude-sonnet-4-5 pytest tests/live/
```

## Requirements

- Julia on `PATH` (the live smoke runs Julia via the kernel on base Julia)
- One of: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`

Tests skip cleanly when requirements are missing.

## What is exercised

`test_live_workspace_read_and_run_julia` demonstrates makker's core value:

1. The LLM reads a file from the **workspace** via `read_file` (deep-agents stock tool).
2. The LLM evaluates the file contents in the **persistent Julia REPL** via `run_julia`.
3. The turn runs through the production **`TurnRunner`** path and records a trace.

The workspace file contains `sum([7, 14, 21, 28, 35])` — the answer `105` cannot be
guessed without reading the file and running Julia.

## Cost and reliability

- Bounded cost: one short turn at temperature 0 (~$0.01 with the default mini model).
- One automatic retry on assertion failure (LLM non-determinism).
- For deterministic wiring coverage without API cost, use
  [`tests/test_end_to_end.py`](../test_end_to_end.py) instead.

## Expected runtime

Includes one Julia cold start (~20–30 s on a typical workstation) plus one LLM round-trip.
