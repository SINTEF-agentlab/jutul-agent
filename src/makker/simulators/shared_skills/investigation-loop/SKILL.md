---
name: investigation-loop
description: Pattern for iterative investigations (calibration, parameter sweeps, sensitivity, etc.) using record_attempt and write_report
---

# Investigation loop

Use this when an investigation involves repeatedly running a simulation,
tweaking inputs, observing how outputs change, and eventually summarising
what happened. Examples: matching a model to data, parameter sweeps,
sensitivity analyses, comparing approaches, hunting a regression.

## Two capabilities

- ``record_attempt`` logs one step of the investigation to the session
  trace — a rationale, optional metrics, optional plot path, and the
  parent step's id when this is a branch or refinement. It returns a
  status line ending in ``id=<uuid>``; pass that as
  ``parent_attempt_id`` next time so the trace builds a tree.
- ``write_report(narrative, output_path=...)`` writes a self-contained
  HTML report at the given path. The narrative markdown is yours to
  write; the rendered HTML also embeds the attempt tree, metrics, and
  any plots referenced by ``record_attempt``.

Both are optional. Use them when they help, skip them when they don't.

## One attempt per tried thing

Record every meaningful run as its own ``record_attempt`` — not one
combined entry at the end. The baseline is itself an attempt (the
"root"); every parameter change you actually evaluate is a child.

For a typical calibration this means:

1. Run the baseline. Capture its metrics.
   ``record_attempt(rationale="baseline", metrics=..., plot_artifact_path=...)``
   → keep its id as ``baseline_id``.
2. For each hypothesis you try: run it, plot it, then
   ``record_attempt(rationale=..., metrics=..., plot_artifact_path=...,
   parent_attempt_id=baseline_id)``.
3. If you refine a hypothesis, pass that hypothesis's id as the new
   parent so the tree reflects branching.

The report's "Exploration map" relies on this tree — collapsing every
hypothesis into one final "summary" attempt loses it.

## One plot per attempt

A single scalar (rmse, end-time, end-voltage) rarely tells the full
story — a calibration that lowers RMSE can still look obviously wrong on
the curve. **Plot every attempt that you record.** A good pattern:

- The baseline's plot shows the measurement vs. the baseline curve.
- Each subsequent attempt's plot shows the measurement plus the
  baseline (as a dashed reference) plus this attempt's curve. The
  reader can flip through attempts in the report and watch the curve
  evolve.
- Use the ``slot`` argument so the file name is descriptive
  (``slot="baseline"``, ``slot="thinner_coating"``, …). Slots overwrite
  on reuse, so a refined attempt can keep the same slot.

A typical evaluation cycle:

```julia
include("candidate.jl")
t, y = run_candidate()
rmse = sqrt(mean((interp(obs.t, t, y) .- obs.y).^2))
```

```python
plot_julia(
    code="""
        fig = Figure()
        ax = Axis(fig[1, 1], xlabel="time (s)", ylabel="V")
        lines!(ax, obs.t, obs.y, label="observed")
        lines!(ax, baseline_t, baseline_y, label="baseline", linestyle=:dash)
        lines!(ax, t, y, label="candidate")
        axislegend(ax)
        fig
    """,
    slot="thinner_coating",  # → artifacts/thinner_coating.png
)
record_attempt(
    rationale="reduce positive-electrode coating thickness by 0.5%",
    metrics={"rmse_V": rmse, "t_end_s": t[end]},
    plot_artifact_path="artifacts/thinner_coating.png",  # match the slot
    parent_attempt_id=baseline_id,
)
```

When you set ``slot="…"`` on ``plot_julia`` the artifact path is
``artifacts/<slot>.<format>``. Pass that exact string to
``record_attempt(plot_artifact_path=…)`` — the report locates the file
relative to the session and embeds it next to the attempt's metrics.

Don't wait to be asked. The report degrades gracefully when a plot is
missing, but it's much more useful when every attempt has one.

``CSV``, ``DataFrames``, ``Statistics``, and ``Interpolations`` are in
the simulator Julia env. The persistent REPL keeps loaded packages and
defined helpers across calls — define helpers once.
