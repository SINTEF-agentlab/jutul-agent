---
name: fimbul-overview
description: Set up geothermal Fimbul cases and interpret reservoir and well results.
---

# Fimbul

Fimbul covers geothermal production and thermal energy storage: doublets,
enhanced and closed-loop systems, borehole storage, and high-enthalpy systems.
Start from a case factory and the closest installed example under `examples/`.
Fimbul extends JutulDarcy with geothermal physics, so its grid, well, and
simulation APIs also apply.

## Set up and run

Case factories assemble the model, parameters, initial state, schedule, and
forces into a `JutulCase`:

```julia
using Fimbul, JutulDarcy
case = egg_geothermal_doublet()
result = simulate_reservoir(case; info_level=1)
```

`simulate_reservoir` validates the case by default. Review its messages;
warnings and errors may be reported without stopping the solve. A separate
validation call is useful when checking setup without running a simulation.

For custom geometry or wells, follow the installed Fimbul example and the
JutulDarcy grid/well constructors. Use the JutulDarcy source resolved in this
environment; a separate installation may expose different APIs.
Analytical examples can return additional reference solutions alongside the case.

Find production and closed-loop examples under `examples/production/`,
ATES/BTES and other storage setups under `examples/storage/`, and reference
heat solutions under `examples/analytical/`.
Preserve the factories' energy formulation and connected-well settings.
For short closed-loop checks, shorten the schedule before coarsening geometry
that contains well trajectories.

## Results

`result.states`, `result.wells`, and `result.time` contain reservoir states,
well outputs, and report times. Use these properties rather than integer
indexing of `result`.
For example, `result.states[end][:Temperature]` is the final cell-temperature
vector; `[s[:Temperature][cell] for s in result.states]` is one cell's history.
Produced-fluid temperature comes from the named well series, for example
`result.wells.wells[:Producer][:temperature]`, not a nearby reservoir cell.
Inspect the available well names and fields before selecting them.
Inputs and raw results use SI units. Temperatures are Kelvin; subtract 273.15
when reporting Celsius values.

## High-temperature water and steam

For high-temperature water/steam, inspect the `H2OSystem` examples. Distributed
steam tables load through `Fimbul.steam_tables_h2o()`; CoolProp is only needed
to generate new tables. This system uses pressure and enthalpy, deriving
temperature from the tables. The validation examples under
`examples/validation/` show the high-temperature benchmarks.

`examples/production/magmatic_intrusion_demo.jl` separates natural-state
equilibration from production. Use `num_years=0` for the natural state without
wells; start production on the same grid with `natural_state_time=0` and
`initial_state=result_ns.states[end]`. The full example is long; `test/cases.jl`
contains a smaller case.

An `H2OSystem` total-energy run can fail with `KeyError(:Temperature)` in
convergence checking; report that limitation rather than silently changing
physics.

Use `fimbul-plotting` for native reservoir, well, and phase-diagram views.
