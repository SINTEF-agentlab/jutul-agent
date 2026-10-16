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
Reach for its installed examples for the grid, mesh, and well primitives
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

Well data (`result.wells`) and reservoir states follow the conventions in the
installed JutulDarcy source. Fimbul has its own resolved environment; do not
assume APIs from a newer standalone JutulDarcy installation apply here.

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

## High-temperature water and steam

Fimbul 0.3.6 adds the pressure–enthalpy `H2OSystem`, distributed steam tables,
and high-temperature benchmarks. `Fimbul.steam_tables_h2o()` loads the table
artifact on first use. CoolProp is optional and needed to generate new tables,
not to use the distributed tables. Do not install it for ordinary simulations.
Read `examples/validation/validation_ht_1d.jl` and the `benchmark_ht_1d` /
`benchmark_ht_2d` docstrings before choosing a benchmark and resolution.

Fimbul 0.3.6 declares Jutul 0.4 and JutulDarcy 0.3 compatibility. A locally
updated Fimbul may support newer versions: inspect the installed Project.toml
and `Base.pkgversion(JutulDarcy)`. Use that environment's supported APIs rather
than changing dependencies during a plotting task.

## Plotting

Read `fimbul-plotting` for native temperature viewers, well curves, and the new
pressure–enthalpy phase diagrams. Use the existing `case` and `result`; do not
solve again to change the view. Raw temperatures and the native well-temperature
field are Kelvin. The reservoir viewer also offers a converted Celsius field;
select its full label as shown in the plotting skill.
