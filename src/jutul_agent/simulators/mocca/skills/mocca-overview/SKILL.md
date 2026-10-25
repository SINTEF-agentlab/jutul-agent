---
name: mocca-overview
description: Set up Mocca adsorption processes, run cycles or cyclic steady state, and interpret performance and results.
---

# Mocca

Mocca models adsorption-based gas separation as beds and flow devices joined
in a flowsheet, operated through a cycle of steps. It supports multi-bed and
wet-flue-gas processes, custom materials, optimization and history matching.
Start from the closest installed example under `examples/`.

## Set up and run

Use a library process or a process JSON file:

```julia
import Mocca
process = Mocca.read_process("haghpanah_2013_four_stage_vsa")
result = Mocca.simulate(process; cycles=3)
```

`Mocca.library_processes()` and `Mocca.library_adsorbents()` list the bundled
choices. `four_stage_vsa_process`, `two_bed_vsa_process`, `lpp_vsa_process`
and `dual_adsorbent_vsa_process` are configurable constructors.
`examples/breakthrough.jl` builds a one-step adsorption process for a DCB run.

`simulate` runs the requested cycles and retains the solver's substeps.
For steady-cycle performance, use `Mocca.cyclic_steady_state(process)` and
check `result.converged` before calling the result steady. A fixed number of
cycles does not establish steady state; its `converged` field is `nothing`.
The steady-state result contains the final cycle, whereas `simulate` retains
all requested cycles. Inspect `docs/src/steady_state.md` for convergence,
warm starts and solver options; `progress=true` reports progress in scripts.

Inspect named settings with `Mocca.variable_names(process)` and change them
with, for example, `process[:P_L] = 5e3` or `process[:t_ads] = 30.0` when those
variables are present. Use separate processes for independent comparisons:
results retain their process, which can be mutated by later changes.
`Mocca.validate(process)` returns a list of input issues; inspect their
meaning rather than treating it as a boolean. A breakthrough without an
extract stream cannot report capture purity/recovery.

For custom processes, follow `examples/process_from_parts.jl`: `Gas`, `Feed`,
`Adsorbent` and `Column` → `Flowsheet` with `add_bed!`/`add_device!` → `Cycle`
→ `Process`. Device roles determine which streams each step opens and which
streams count as feed, light product and extract. Keep the example's thermal
model (`WithWall`, `Adiabatic` or `Isothermal`) unless changing it is intended.
Inputs use SI units; `Mocca.Units` provides conversions such as `bar` and
`celsius(25)`. Old constants/JSON inputs can be migrated with
`Mocca.convert_constants`; `setup_mocca_case` is no longer a simulation path.

## Interpret and export results

`result.states[k]` contains all beds and devices. For a bed named `:Bed`:

```julia
bed = process.flowsheet[:Bed]
bed_states = [s[:Bed] for s in result.states]
times = cumsum(result.timesteps)
```

`result.timesteps` holds step durations in seconds. Keep it paired with the
states. Bed cells run from bottom (cell 1) to top (last cell); actual feed and
product ends depend on device connections. Inspect unit names in a multi-bed
process instead of assuming `:Bed`.

- `:Pressure`: cell vector in Pa.
- `:Temperature`: cell vector in K; `:WallTemperature` exists for `WithWall` beds.
- `:y`: component-by-cell gas mole fractions.
- `:AdsorbedConcentration`: component-by-cell loading in mol/m³ of adsorbent.
  Divide by adsorbent density for mol/kg.

Component order follows `Mocca.component_names(process.flowsheet.gas)`.
A device's `:MolarFlow` is in mol/s, positive from inlet to outlet; its inlet
and outlet composition/temperature fields describe the ports.

Use `Mocca.performance(result)` for purity, recovery, specific energy (kWh/t)
and productivity (mol/(m³·s)). It integrates device streams over the last
cycle, not pointwise bed composition. State whether the result is transient
or converged. `performance(process)` runs a steady-state solve; use the
existing result when one is already available.

`Mocca.write_results("results/base_case", result)` exports CSV summaries,
streams and bed histories/profiles plus the process JSON. Exported columns
carry engineering units (including bara, °C and mol%); raw states remain SI.
Use `write_process(path, process)` / `read_process(path)` to save/reload a setup.
See `mocca-plotting` for native result views.

## Optimization and extensions

`add_variable!` exposes named design variables; `performance_gradient` and
`optimise!` use steady-state gradients. Follow `examples/optimization.jl` for
bounds, objective scaling and constraints. Targets such as minimum purity
are penalties, so inspect the achieved metrics rather than assuming they
were met. `examples/history_matching.jl` uses Jutul's dictionary optimizer
for transient data; its result and history differ from `optimise!`.

Custom chemistry needs suitable isotherm, heat and mass-transfer data.
`docs/src/adsorbents.md` and `docs/src/extending.md` cover material/step/unit
extensions; `examples/unit_network.jl` shows lower-level coupling. Check these
before claiming a requested material or cycle is built in.
