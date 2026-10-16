---
name: jutuldarcy-plotting
description: Plot JutulDarcy reservoirs and results with the native GUI, select fields and phases, show well dashboards, or make an explicitly requested static export.
---

# JutulDarcy plotting

Use this skill for reservoir plots, simulation-result views, and well curves.
Run the Julia examples below through **`plot_julia`**, using existing REPL
bindings. The tool selects the backend and captures native viewers automatically.

## Default: the reservoir GUI

For “plot the reservoir”, “show the results”, or an interactive 3D view, start with:

```julia
plot_reservoir(model, states)
```

Keep the defaults `gui=true, fancy=true`. This gives the user the field selector,
timestep slider, filters, and well toggles. Pass **all** report-step states to
retain the timestep slider; `states[end]` gives only one step. Before simulating,
`plot_reservoir(model)` shows the mesh, rock properties, and wells.

For a particular field or report step, keep the GUI and select it with `key` and
`step`. The initial step is 1 unless specified:

```julia
plot_reservoir(model, states; key=:Pressure, step=length(states))
```

Matrix fields appear in the GUI as separate entries named `"<field> row <n>"`.
Select a saturation component through **`key`**, not a `row` keyword:

```julia
# For a model whose phase order is (AqueousPhase(), LiquidPhase()).
plot_reservoir(model, states; key="Saturations row 1", step=length(states))
```

For the second phase in that model:

```julia
plot_reservoir(model, states; key="Saturations row 2", step=length(states))
```

Check the model's phase order before identifying a row as water, oil, or gas;
`JutulDarcy.get_phases(reservoir_model(model).system)` exposes that order.
The same naming applies to composition matrices. `key=:Saturations` matches the
first saturation entry; it does not select an arbitrary phase.

For scalar fields, original names such as `:Pressure` and `:Temperature` match
their displayed names with units. Values are converted from SI to metric by
default. Do not convert the values manually and then let the plotter convert
them again. Keep the plotter's default aspect and backend settings unless the
user asks for a change.

## Data shapes and derived fields

The second argument accepts a result (`result`), a vector of state dictionaries
(`result.states` / `states`), or one state dictionary (`states[end]`). It does
**not** accept a bare vector of cell values: that is interpreted as a sequence
of states and fails in `convert_for_plotting`.

For a derived field, put the cell vector in a named dictionary and use the same
native viewer. A history is a vector of such dictionaries. For example, pressure
change relative to the first report step:

```julia
pressure_change = [Dict(:PressureChange => s[:Pressure] .- states[1][:Pressure])
                   for s in states]
plot_reservoir(model, pressure_change; key=:PressureChange, step=length(states))
```

Here the custom field remains in Pa: unfamiliar field names have no automatic
unit conversion. State the units and reference step in the caption. Use
`plot_cell_data(physical_representation(reservoir_domain(model)), values)` when
the user specifically wants a bare scalar-field plot without the reservoir GUI.

## Static plots only when requested

Use `gui=false` for a requested still image, report figure, or view without
controls. This path accepts **`row`** for matrix fields:

```julia
plot_reservoir(model, states; gui=false, key=:Saturations,
    step=length(states), row=2)
```

`row` is a static-plot keyword; passing it to the default GUI causes a Makie
`InvalidAttributeError`. Do not switch to `gui=false` just to recover from that
error: use the GUI key spelling above. `fancy=false` is another viewer with
different controls; it is not needed for browser plotting.

## Other native views

- Well rates and BHP: `plot_well_results(result)` or `plot_well_results(wd)`.
  Choose the initial response with `field=:rate` or `field=:bhp`. Use separate
  `plot_julia` calls and slots if the user wants both visible at once.
- Summary dashboard: `plot_summary(result)`; inspect the installed docstring for
  response codes and multi-run comparisons.
- CO₂ inventory: `plot_co2_inventory(t, inv)` after computing the inventory with
  the installed example's API.
- Model graphs: load `GraphMakie, NetworkLayout, LayeredLayouts` first, then use
  `plot_model_graph(model)` or `plot_variable_graph(reservoir_model(model))`.

Custom figures fit requested overlays or layouts the native viewers do not
provide. A standard reservoir or well-results request should use the native
viewer first.

## Avoid display-option mixups

`slot`, `view`, and `window` belong to the **`plot_julia` tool**. Do not pass
`window=false` to `plot_reservoir`: it becomes an invalid Makie attribute. No
display-suppression keyword or backend import is needed inside the plot code.
See `plotting-basics` for capture, recapture, and closing views.

If another keyword fails, inspect the installed `src/ext/ext_makie.jl` and the
forwarded plotter in Jutul's Makie extension. Repair the call while preserving
the requested viewer and reuse the same plot slot. Report a genuine library
limitation before substituting a different view.
