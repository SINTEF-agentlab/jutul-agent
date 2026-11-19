---
name: jutuldarcy-plotting
description: Select native JutulDarcy reservoir fields and phases, well views, and requested static plots.
---

# JutulDarcy plotting

Use native plotters through `plot_julia`, reusing the solved results.
For ordinary reservoir views, keep the GUI and full timestep history:

```julia
plot_reservoir(model, states)
```

The default `gui=true, fancy=true` viewer includes fields, timesteps, filters,
and well toggles. Selecting a field does not require a static plot.
Before solving, `plot_reservoir(model)` shows rock properties and wells.
Select a field and report step with `key` and `step`; the default step is 1.
For example, use `key=:Pressure, step=length(states)`.

Matrix fields have GUI entries named `"<field> row <n>"`:

```julia
plot_reservoir(model, states; key="Saturations row 2", step=length(states))
```

Check `JutulDarcy.get_phases(reservoir_model(model).system)` before identifying
a row as water, oil, or gas. `key=:Saturations` selects the first entry.
The GUI does not accept `row`; use the full key rather than switching viewers.
The same row naming applies to composition matrices. Scalar names such as
`:Pressure` match their displayed names with units.
Known fields are converted from SI to metric automatically; do not convert twice.

For custom cell data, pass a named state dictionary or a history of dictionaries,
not a bare cell vector. The second argument also accepts `result` directly,
`result.states`, or one state dictionary; a single state loses the time history.
For example:

```julia
changes = [Dict(:PressureChange => s[:Pressure] .- states[1][:Pressure]) for s in states]
plot_reservoir(model, changes; key=:PressureChange, step=length(states))
```

Custom field names have no automatic unit conversion; label their units.
For an explicitly requested bare cell-field plot, use
`plot_cell_data(physical_representation(reservoir_domain(model)), values)`.

Use `gui=false` for an explicitly requested static view. This path accepts `row`:

```julia
plot_reservoir(model, states; gui=false, key=:Saturations, row=2, step=length(states))
```

`fancy=false` selects another viewer with different options; it is not needed
for browser plotting.

Other native views:

- `plot_well_results(result; field=:bhp)` or `plot_well_results(wd; field=:rate)`
  for well curves. Use separate plot slots when both should remain visible.
- `plot_summary(result)` for a dashboard; its docstring covers multi-run comparisons.
- `plot_co2_inventory(t, inv)` after computing inventory with the installed
  example's API.
- `plot_model_graph(model)` or `plot_variable_graph(reservoir_model(model))`
  after loading `GraphMakie, NetworkLayout, LayeredLayouts` to activate the extension.

For units, use `unit_sys="SI"` on `plot_well_results` and `unit_system="SI"`
on `plot_summary`. Unit names are strings: `"Metric"`, `"SI"`, or `"Field"`.

`slot`, `view`, and `window` belong to the `plot_julia` tool, not the native
plotter. Keep the default aspect and backend settings unless a change is requested.
For keyword errors, inspect `src/ext/ext_makie.jl` and the forwarded Jutul plotter.
Preserve the requested viewer when correcting a call. Custom figures are useful
for requested overlays or layouts the native plotters cannot provide.
