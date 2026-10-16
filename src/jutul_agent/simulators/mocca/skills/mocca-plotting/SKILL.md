---
name: mocca-plotting
description: Plot Mocca breakthrough, spatial states, individual cell histories, and optimization results with native plotters and correct time/component indexing.
---

# Mocca plotting

Use the native plotters through `plot_julia`; reuse `case`, `states`, and
`timesteps` from `Mocca.simulate_process`. The tool selects the backend and
captures the figure. No manual `display`, backend activation, or new solve
is needed. These are native multi-panel figures, not a reservoir GUI.

## Breakthrough / outlet results

```julia
Mocca.plot_outlet(case, states, timesteps)
```

This plots every primary field at the outlet, including a labeled curve for
each component. It is the first choice for “plot the breakthrough results”.
Check that the stored history contains multiple states. A solve with only one
report step and `output_substates=false` cannot show a breakthrough curve;
the native line plots may look empty. Keep `output_substates=true` when setting
up a new solve. If the existing output contains only a final state, explain
that limitation instead of presenting it as a time history.

## Spatial state and cell history

```julia
Mocca.plot_state(states[end], case.model)
```

Pass one state dictionary, not the entire history, to `plot_state`.
For the feed-end history:

```julia
Mocca.plot_cell(states, case.model, timesteps, 1)
```

`timesteps` contains **step durations**, not cumulative times. The plotters
call `cumsum` internally. Keep it paired with the returned states; passing
cumulative times stretches the time axis incorrectly. Use
`cumsum(timesteps)` only when building an explicitly requested custom curve.

## Components and units

`states[i][:y]` and `:AdsorbedConcentration` have one row per component and
one column per cell. Inspect `case.model.system.component_names`; do not
assume row 1 is CO2 in a custom system. `:Pressure` is in Pa and temperatures
are in K. Native plots label those units. A requested custom bar or Celsius
view needs an explicit conversion.

Cell 1 is the feed/product boundary and the last cell is the outlet. A mole
fraction in a cell is not cycle-integrated product purity or recovery. For
those quantities, follow the boundary-flux integration in the installed
`examples/optimization.jl` and define the stages being integrated.

## Optimization history

Use `Mocca.plot_optimization_history(dprm)` after an optimization has recorded
objective values. Check that `dprm.history` is not `missing` and that
`dprm.history.objectives` is nonempty first; a failed optimization may leave no
history to plot. Its default y scale is logarithmic and requires positive
values. For recovery maximization or an objective that may reach zero:
`Mocca.plot_optimization_history(dprm; yscale=identity, ylabel="Recovery")`.
Do not invent history entries to present a result.

Custom axes are appropriate for a specifically requested overlay or layout
that these plotters do not provide. Keep `slot`, `window`, and `view` on the
`plot_julia` tool; they are not Mocca plotting keywords.
