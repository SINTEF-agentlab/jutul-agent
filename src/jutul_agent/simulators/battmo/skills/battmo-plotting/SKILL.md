---
name: battmo-plotting
description: Plot BattMo results with native dashboards, timestep controls, variable expressions, cell curves, and 3D viewers.
---

# BattMo plotting

Run these recipes through `plot_julia`. Reuse the `sol = solve(sim)` already
in the REPL; never solve again just to plot. The tool selects WGLMakie for the
browser and GLMakie for native windows. BattMo supports both.

## Results and interactive controls

For a general results request, use the native current/voltage dashboard:

```julia
plot_dashboard(sol; new_window=false)
```

For concentrations and potentials along the cell, with a timestep slider:

```julia
plot_dashboard(sol; plot_type="line", new_window=false)
```

Other supported dashboard styles are `"contour"` (space versus time) and
`"breakdown"` (approximate voltage-loss decomposition). Pass the style through
`plot_type`; don't replace a requested native dashboard with handwritten axes.
`new_window=false` suppresses BattMo's own display call; `plot_julia` still
publishes the live figure and its controls.

## Selected output variables

`plot_output` requires a second positional argument: a nonempty vector of
variable expressions. A call with only `sol` raises a method error.

```julia
plot_output(sol, ["Voltage vs Time", "Current vs Time"]; new_window=false)
```

Use a nested vector to overlay expressions on one axis. Inspect
`keys(sol.time_series)` and the installed `examples/beginner_tutorials/3_handle_outputs.jl`
for available variables and spatial expressions; don't guess names or indices.
`sol.time_series` is a dictionary of named vectors, not a vector of states.

## Parameter curves and other views

```julia
plot_cell_curves(cell_parameters; new_window=false)
```

For a solved 3D model, `plot_interactive_3d(sol)` wraps Jutul's multi-model
viewer. In BattMo 0.2.9 that wrapper opens a native GL window and does not expose
`new_window`. In a browser session, call the same native viewer directly:

```julia
Jutul.plot_multimodel_interactive(sol.jutul_output.multimodel,
    sol.jutul_output.states; colormap=:curl, new_window=false)
```

Use a spatial model, not a 1D line model. For a short pouch example, keep the
published geometry and shorten `Simulation(...; time_steps=...)`; reducing
the grid below its tab dimensions can produce invalid geometry.

- `plot_thermal_source_contributions(time, source_parts; new_window=false)`
  needs the per-timestep source decomposition from `get_energy_source!`.
  Read the thermal example first; these are not ordinary `time_series` entries.
  `total_source` adds the residual, and `normalize=true` shows percentages.

Custom figures are appropriate for a specifically requested comparison/layout
that the native plotters cannot express. `slot`, `view`, and `window` belong to
the `plot_julia` tool, not the BattMo plotter. If a native call fails, inspect its
installed implementation in `ext/BattMoMakieExt.jl` before switching plotters.
