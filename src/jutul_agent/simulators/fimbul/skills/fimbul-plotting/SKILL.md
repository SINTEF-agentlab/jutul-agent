---
name: fimbul-plotting
description: Plot Fimbul temperature histories, geothermal wells, and pressure–enthalpy phase diagrams using native plotters from its supported JutulDarcy environment.
---

# Fimbul plotting

Run native plotters through `plot_julia` and reuse the solved `case` and
`result`. The tool selects the backend. Check `Base.pkgversion(JutulDarcy)`:
Fimbul 0.3.6 currently resolves JutulDarcy 0.3, while a locally updated Fimbul
may use JutulDarcy 0.4. Their viewer APIs differ.

## Temperature viewer

With JutulDarcy 0.4, keep the default native reservoir GUI:

```julia
plot_reservoir(case.model, result.states;
    key="Temperature_relative / °C", step=length(result.states))
```

Pass the whole history so the viewer retains its timestep slider. For a requested
static image, add `gui=false` to this call. The static figure should still be
shown to the user: leave the tool's `window` option at its default. Removing
GUI controls does not mean hiding the plot.

With the released Fimbul/JutulDarcy 0.3 stack in a browser, use the native
fixed-axis interactive viewer:

```julia
plot_reservoir(case.model, result.states; fancy=false, new_window=false,
    key="Temperature_relative / °C", step=length(result.states), aspect=:data)
```

This keeps `gui=true`, the field menu, filters, and timestep slider. Pass the
whole state history. `aspect=:data` shows the actual geometric proportions,
including thin 1D analytical cases. The older default fancy viewer allocates
OpenGL buffers even under WGLMakie; use the supported fixed-axis viewer in the
browser instead of inventing a custom plot or silently dropping the GUI.
`new_window=false` prevents this older plotter from opening a native GL window;
`plot_julia` still serves the interactive browser view.

Before a solve, use `plot_reservoir(case.model)` for rock properties; omit the
temperature key and timestep. On JutulDarcy 0.3 in the browser, retain
`fancy=false, new_window=false, aspect=:data` on that model-only call.
Raw `:Temperature` values are Kelvin. The converted native fields are
`"Temperature / °K"` and `"Temperature_relative / °C"`; select the full label
to make the displayed units explicit. A static plot with `key=:Temperature`
uses Kelvin, so do not caption it as Celsius. Do not subtract 273.15 before
passing a `:Temperature` field to the native plotter.
On JutulDarcy 0.3, use Jutul's native `plot_cell_data` for a requested static
field image. For Celsius, convert the vector explicitly and label it:

```julia
temperature_c = vec(result.states[end][:Temperature]) .- 273.15
mesh = JutulDarcy.reservoir_mesh(case.model)
plot_mesh = mesh isa Jutul.CartesianMesh ? Jutul.UnstructuredMesh(mesh) : mesh
fig, ax, p = Jutul.plot_cell_data(plot_mesh, temperature_c; colorbar=false)
Makie.Colorbar(fig[2, 1], p; vertical=false, label="Temperature (°C)")
fig
```

The Cartesian-to-unstructured conversion uses Jutul's existing mesh API to
avoid a triangulation defect in the older direct Cartesian plotter. It only
changes the mesh passed to the plot; the simulation model is unchanged.
The plotting tool already binds `Makie` through the active backend. Use that
binding for labels and colorbars; do not add `using Makie`, since Makie is a
transitive dependency rather than a directly loadable package in this env.

## Well results

```julia
plot_well_results(result.wells; field=:temperature, new_window=false)
```

This native well field uses absolute temperature (Kelvin in Metric/SI). Keep
its label and explain the displayed units; do not call it Celsius.

Discover available wells/fields from `result.wells.wells`; do not read a
reservoir cell as the produced-fluid temperature. Fimbul's own
`plot_well_data!` compares full well-state histories (not `result.wells`):
provide `names`, the well names, and the case-sensitive `field=:Temperature`
or `:Energy`. Check the installed `ext/FimbulGLMakieExt/well_plots.jl` first.
`plot_mswell_values!` plots values along multi-segment-well nodes; pass node
values, not a reservoir-cell vector or a scalar well time series.

## High-temperature phase diagrams

These plotters use pressure in Pa and specific enthalpy in J/kg, and label
the axes in MPa and kJ/kg. They return `(figure, handles)`; return the figure
from the tool code. For a model with `H2OSystem` and pressure–enthalpy states:

```julia
fig, handles = Fimbul.plot_reservoir_state_ph(result.states[end])
fig
```

```julia
fig, handles = Fimbul.plot_reservoir_state_phase_diagram(case.model, result.states[end])
fig
```

For the table contours alone:

```julia
fig, handles = Fimbul.plot_phase_diagram_contours(Fimbul.steam_tables_h2o())
fig
```

The corresponding `!` functions draw into an existing axis. Inspect the
installed example for supported contour variables and ranges. A standard
single-phase thermal case need not contain the pressure–enthalpy fields or
steam tables these plotters require.

`slot`, `view`, and `window` belong to the `plot_julia` tool. Keep them out of
native Julia plotting calls. Report a real backend limitation before choosing
another view, and preserve the requested interactivity whenever possible.
