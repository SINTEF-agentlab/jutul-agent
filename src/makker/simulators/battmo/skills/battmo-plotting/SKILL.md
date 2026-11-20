---
name: battmo-plotting
description: Select native BattMo dashboards, spatial controls, output plots, parameter curves, and 3D views.
---

# BattMo plotting

Use native plotters through `plot_julia`, reusing the solved `sol`.
Include `using BattMo` if needed. Pass `new_window=false` so BattMo leaves
display to the tool.

| View | Call |
| --- | --- |
| General results | `plot_dashboard(sol; new_window=false)` |
| Spatial profiles with timestep controls | `plot_dashboard(sol; plot_type="line", new_window=false)` |
| Selected outputs | `plot_output(sol, ["Voltage vs Time", "Current vs Time"]; new_window=false)` |
| Parameter curves | `plot_cell_curves(cell_parameters; new_window=false)` |

Dashboard styles also include `"contour"` (space versus time) and
`"breakdown"` (voltage-loss contributions).
`plot_output` requires a nonempty expression vector; inspect
`keys(sol.time_series)` and `examples/example_1d_plotting.jl` for other variables.
Nest expressions in a vector to overlay them on one axis.
That example also covers spatial and cycle-metric expressions.

For a solved 3D model in the browser, call the underlying native viewer:

```julia
import Jutul
Jutul.plot_multimodel_interactive(sol.jutul_output.multimodel,
    sol.jutul_output.states; colormap=:curl, new_window=false)
```

The `plot_interactive_3d(sol)` wrapper opens a native window and has no
`new_window` option. Use a spatial model for the 3D viewer; use the line
dashboard for a P2D cell.

`plot_thermal_source_contributions(time, source_parts; new_window=false)`
needs the source decomposition from `get_energy_source!`, not ordinary
`time_series` entries. See the installed thermal example.
Its `total_source` option allows a residual contribution and
`normalize=true` displays relative contributions.

Inspect `ext/BattMoMakieExt.jl` when the requested view needs additional options.
Use a custom figure when the requested comparison or layout is not supported
by these native views.
