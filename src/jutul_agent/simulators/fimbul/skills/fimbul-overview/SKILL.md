---
name: fimbul-overview
description: High-level Fimbul workflow, geothermal case factories, and result inspection
---

# Fimbul orientation

## When to use

Use this skill for any geothermal task on Fimbul — aquifer or borehole
thermal energy storage (ATES, BTES, FTES, HTATES), conventional and
enhanced geothermal systems (doublet, EGS, AGS, closed-loop coaxial BHE),
or analytical sanity checks.

## Mental model

Fimbul extends JutulDarcy with an energy-conservation equation that
transports heat by advection and conduction. A Fimbul simulation is
*structurally* a JutulDarcy reservoir simulation with a temperature
field, so JutulDarcy's primitives (grids, `reservoir_domain`, wells,
`simulate_reservoir`) carry over unchanged.

The simplest entry point is a **case factory** — a function in Fimbul
that returns a `JutulCase` for a named example geometry:

```julia
using Fimbul, JutulDarcy
case = egg_geothermal_doublet()        # case object
result = simulate_reservoir(case; info_level = 1)  # run it; info_level = 2 if convergence struggles
T = result.states[end][:Temperature]   # final-state temperature field
```

When you need a custom case, drop down to JutulDarcy primitives and add
geothermal physics through the same Fimbul/JutulDarcy API surface.

## Finding what you need

Fimbul's source is read-only depot source; its path is given to you up front, so
browse it directly with the file tools (see the `workspace-and-source` skill):

```text
# Fimbul source path is in your system prompt -> /.../Fimbul/<hash>
glob("/.../Fimbul/examples/**/*.jl")              # analytical / production / storage
grep("function egg_geothermal", path="/.../Fimbul/src")   # a case factory's source
read_file("/.../Fimbul/examples/production/doublet_demo.jl")
```

Fimbul builds on JutulDarcy, whose source sits alongside at `pkgdir(JutulDarcy)`.
Reach for it (and the JutulDarcy skills) for the grid, mesh, and well primitives
Fimbul reuses:

```text
glob("/.../JutulDarcy/examples/**/*.jl")          # reservoir + well setup
grep("setup_well", path="/.../JutulDarcy/src")
```

For docstrings, stay in the REPL: `run_julia("@doc egg_geothermal_doublet")`.

## Result inspection

Fimbul cases return JutulDarcy result objects. The key extra field is
`:Temperature` on each state:

```julia
states = result.states
T_end = states[end][:Temperature]            # temperature at the final step
T_at_cell = [s[:Temperature][cell] for s in states]   # series at one cell
```

Well data (`result.wells`) and reservoir states follow the same
conventions as JutulDarcy. See `jutuldarcy-overview` for result unpacking
and `jutuldarcy-wells` for well construction details that apply here too.

Produced-water quantities come from the well series, not the reservoir
field: `result.states[end][:Temperature]` is the grid; what a well delivers
over time is

```julia
prod_T = result.wells.wells[:Producer][:temperature]   # K, one value per step
```

(`keys(result.wells.wells)` lists the well names of the active case.)

All quantities are SI: temperatures are **Kelvin**, not Celsius. When the
user asks for degrees Celsius, convert (`T - 273.15`) before reporting; a
"temperature" near 350 in a geothermal answer is almost certainly an
unconverted Kelvin value.

## Plotting

Fimbul reuses JutulDarcy's **native plotters** (GLMakie, the default backend),
captured by `plot_julia` automatically — headless or interactive:

- `plot_reservoir(case.model, result.states)` — 3D reservoir (e.g. `key = :Temperature`)
- `plot_well_results(result.wells)` — well dashboard
- `plot_cell_data!` / `plot_mesh_edges!` to compose a custom 3D view
- Fimbul's own `plot_well_data!` / `plot_mswell_values!` for multi-segment wells

Just call them — no backend juggling. In an interactive session a live window
opens for the user by default (`window=false` to suppress); pass `view=true` to
inspect a plot yourself. For a custom 2D view, build inline against the live
result object (probe `keys` / `propertynames` first).

JutulDarcy's `plot_reservoir` defaults to its fancy 3D viewer and converted
display units; use `fancy=false` or `convert_units=false` when a fixed-axis view
or unconverted values are needed.
