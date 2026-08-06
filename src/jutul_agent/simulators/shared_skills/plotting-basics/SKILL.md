---
name: plotting-basics
description: "How to plot with plot_julia: native plotters, live windows, and seeing your own plots"
---

# Plotting basics

## When to use

Use this skill whenever the user asks for a plot, chart, figure, or visualization,
or when a plot would make simulation results easier to understand.

## The model: `plot_julia` captures whatever you draw

Use `plot_julia` for **anything that produces a figure** — never build a `Figure`
in `run_julia` (that saves no artifact and the user can't see it). `plot_julia`:

- activates the right Makie backend for this session (you don't manage backends),
- evaluates your code, and
- captures the resulting figure to an image file — whether your code **returns** a
  `Figure`, returns a `(fig, ax, plot)` tuple, or calls a native plotter that opens
  a window / calls `display` internally.

So you do **not** need to end on `fig`, and you do **not** need to avoid `display`.
Just call the plotter.

## Prefer native plotters

Call your simulator's own documented plotters first — they are the canonical,
best-looking views, they run on GLMakie, and `plot_julia` captures them
automatically. The `<sim>-overview` skill lists the plotters for your simulator.

Build a `Figure` inline when no native plotter fits, or when you want a specific
custom view:

```julia
fig = Figure(size = (600, 400))
ax = Axis(fig[1, 1], title = "History match", xlabel = "t", ylabel = "rate")
lines!(ax, t, sim); scatter!(ax, t, obs)
fig
```

## Seeing your own plot: `view=true`

Pass `view=true` to get the (downscaled) image back so **you** can look at it —
to verify a curve overlays the data, spot an anomaly, or check a 3D view. Use it
deliberately, not on every plot (each image costs tokens). The user always sees
the saved artifact regardless of `view`.

```text
plot_julia(code="<your plot code>", view=true)   # then reason about what you see
```

## How the user sees a plot

`plot_julia` shows the figure to the user and saves a PNG record. *How* it is shown
depends on the interface you are driving (your interface note states it): a live
Makie window on the desktop they can rotate, zoom, and step; an interactive figure
pinned in a side panel on the web; or just the saved PNG on a headless/one-shot
run. You always just call `plot_julia` — you don't manage that difference, and
there's no separate "interactive" mode.

- `view` is for **you**, not the user — it returns the image to you; it shows
  them nothing extra.
- Pass `window=false` only to compute/inspect a plot without surfacing it.
- Don't force a backend or window from a plotter's own kwargs (e.g. `gui=true`):
  that opens a separate desktop window outside `plot_julia`. Just call the plotter
  normally — the tool already shows the figure the right way for the interface.
- If the user says they can't see a plot, you probably built it in `run_julia`
  (use `plot_julia`) or set `window=false` — fix that, don't just re-describe it.

## Slots, recapture, close

Each plot is keyed by its `slot` — the same `slot` refreshes that view in place
(reuse it when iterating so you don't spawn a new view per attempt), and distinct
plots get distinct slots (`slot="reservoir"`, `slot="wells"`) so they stay
addressable as separate views. This applies on every interface (a desktop window,
or a tab in the web side panel).

Both act on whichever kind of view the interface gives you.

- **Recapture** after the user rotates/zooms/steps a plot:
  `recapture_plot(slot="reservoir")` captures that plot at its current state and
  returns the image. Omit `slot` for the most recent. On the desktop it re-renders
  the figure jutul-agent still holds, so it works even after the user closed the
  window (you get its last state); only `close_plots` discards it. On the web the
  browser owns the camera, so a plot can only be snapshotted once its view has been
  drawn in the canvas; recapturing one that hasn't been says what to do about it.
  Don't recapture a plot you just drew, though: nothing can have changed yet, and
  `view=true` on `plot_julia` is how you look at your own work. You can't advance
  the timestep yourself either, so ask the user to step the plot first.
- **Close** plots with `close_plots(slot="reservoir")` (one) or `close_plots()`
  (all). Each live plot holds its data until closed or superseded, so closing ones
  the user is done with is worth doing in a long session.

## Diagnostics & model structure

- **Solver performance/convergence**: build these **inline** from the result's
  reports (the native `plot_solve_breakdown`/`plot_cumulative_solve` family is
  finicky about argument types — prefer this). `reports = result.result.reports`,
  then per report step `r`: `length(r[:ministeps])`, `r[:total_time]`, and per
  ministep `m`: `m[:linear_iterations]`, `m[:convergence_time]`. Plot vs report
  step with `Figure`/`Axis`/`lines!` through `plot_julia`.
- **Model structure** (Jutul-based simulators): the dependency graph is drawn by a
  Jutul package extension that only activates once the GraphMakie stack is loaded.
  Load all three first, in the same `plot_julia` call, then call the plotter:
  ```julia
  using GraphMakie, NetworkLayout, LayeredLayouts
  plot_variable_graph(reservoir_model(model))   # per-variable dependencies
  # or: plot_model_graph(model)
  ```
  If it errors with "no method matching plot_variable_graph", the extension hasn't
  loaded yet — re-run the `using` line and the plotter together in one call.

## Sizing and slots

- Output is **PNG**. `size=(width, height)` sets a specific resolution.
- `slot="name"` overwrites the same artifact path when refreshing a comparison
  during calibration (e.g. `slot="saturation_final"`).

## Loading tabular data

```julia
using CSV, DataFrames
obs = CSV.read("experiments/observations/data.csv", DataFrame)   # workspace-relative, no leading slash
```

## Performance

- **Do not re-run a full simulation inside `plot_julia`** if the REPL already holds
  the result/arrays from a recent `run_julia` — plot from the cached objects.
  Plotting should take seconds, not minutes.
- Reuse `slot=` for comparison plots you refresh during calibration.
