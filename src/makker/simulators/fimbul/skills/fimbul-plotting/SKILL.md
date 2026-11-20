---
name: fimbul-plotting
description: Select native Fimbul temperature viewers, well curves, and pressure–enthalpy phase diagrams.
---

# Fimbul plotting

Use native plotters through `plot_julia`, reusing the solved `case` and `result`.
For temperature, keep the reservoir GUI and full timestep history:

```julia
plot_reservoir(case.model, result.states;
    key="Temperature_relative / °C", step=length(result.states))
```

Use `"Temperature / °K"` for Kelvin. Select the full field label to make units
explicit. Raw `:Temperature` values and static `key=:Temperature` use Kelvin.
The plotter converts the Celsius field; do not subtract 273.15 from its input.
Add `gui=false` only for a requested static view, and still display the figure.
Before solving, `plot_reservoir(case.model)` shows rock properties;
`plot_reservoir(case)` includes the initial state.

For matrix fields, select a GUI entry such as `key="Saturations row 2"`;
check the model's phase order before identifying the phase. The GUI does not
accept `row`. For custom cell fields, pass a named state dictionary or history
of dictionaries, not a bare vector; custom names need explicit unit labels.

Fixed-axis examples using `fancy=false` accept options such as `cells` and
`axis_args` that the default GUI does not. Use the GUI's filters for exploration.
Keep the fixed-axis example's viewer and options together for a requested cutaway.
Keep default aspect and backend settings unless a change is requested. For
keyword errors, inspect the installed plotter and preserve the requested viewer
when correcting the call.

## Well results

For well temperatures:

```julia
plot_well_results(result.wells; field=:temperature, new_window=false)
```

This well field uses Kelvin. Discover names and fields from
`result.wells.wells`, rather than sampling a reservoir cell for produced-fluid
temperature.

Fimbul's `plot_well_data!` compares full well-state histories, not
`result.wells`. Set `names` to labels for the reference and comparison runs,
`wells` to well names, and `field` to a case-sensitive name such as
`:Temperature` or `:Energy`; its `:Temperature` view converts to Celsius.
`plot_mswell_values!` takes values along multisegment-well nodes,
not a reservoir vector or a scalar well time series. Inspect
`ext/FimbulGLMakieExt/well_plots.jl` before using either helper.

## Pressure–enthalpy views

For pressure–enthalpy states, use `Fimbul.plot_reservoir_state_ph(state)` or
`Fimbul.plot_reservoir_state_phase_diagram(model, state)`. For table contours,
use `Fimbul.plot_phase_diagram_contours(Fimbul.steam_tables_h2o())`.
These return `(figure, handles)` and take SI inputs (Pa and J/kg).
Return the figure from the plotting code.
The axes display MPa and kJ/kg. The corresponding `!` methods draw into
an existing axis. These views require pressure–enthalpy fields (and steam
tables for contours); an ordinary thermal case may not supply them.
Inspect the installed example for contour variables, limits, and options.

`slot`, `view`, and `window` belong to `plot_julia`, not the reservoir
plotter. For annotations, use the `Makie` namespace already bound by the tool;
`Makie` is not a direct dependency of this environment.
