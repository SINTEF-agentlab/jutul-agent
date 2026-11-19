---
name: mocca-overview
description: High-level Mocca workflow for adsorption-based CO2 capture simulations
---

# Mocca orientation

## When to use

Use this skill for any adsorption-based CO2 capture task on Mocca:
direct column breakthrough (DCB), cyclic vacuum swing adsorption (VSA),
parameter optimization, or history matching against measured
breakthrough data.

## Maturity

Mocca is a young package with a deliberately small scope: a CO2/N2
system on Zeolite 13X (dual-site Langmuir isotherm, linear driving
force mass transfer) in DCB and four-stage VSA configurations. Other
isotherms, chemistries, and cycle types are not implemented; verify
against the installed source before promising a capability.

## Mental model

A Mocca simulation has three composable layers:

1. **Inputs**: physical and operational parameters. `Mocca.parse_input`
   accepts either a JSON file path (reference inputs ship inside the
   package under `models/json/`, with a schema) **or an input `Dict`** —
   the model library provides `haghpanah_cyclic_input()` /
   `haghpanah_DCB_input()` returning such dicts. Inputs can also be
   constructed directly in Julia: `HaghpanahConstants` for the published
   reference setup, or explicit isotherm, mass-transfer, and column
   objects.
2. **Case**: model, initial state, parameters, and staged forces
   assembled into a `MoccaCase`. The JSON route does this in one call,
   `setup_mocca_case(constants, info)`.
3. **Simulate**: `simulate_process(case; ...)` returns
   `(states, timesteps)`; pass `output_substates = true` to keep every
   substep.

The quickest run is JSON-driven:

```julia
using Mocca
json_dir = joinpath(dirname(pathof(Mocca)), "../models/json/")
(constants, info) = Mocca.parse_input(joinpath(json_dir, "dcb_haghpanah_2013_co2_n2_input_simple.json"))
case, ts_config = Mocca.setup_mocca_case(constants, info)
states, timesteps = Mocca.simulate_process(
    case;
    timestep_selector_cfg = ts_config,
    output_substates = true,
    info_level = 1,            # see progress; use 2 if convergence struggles
)
```

The no-file equivalent for the cyclic reference setup:

```julia
constants, info = Mocca.parse_input(Mocca.haghpanah_cyclic_input())
```

Mind `info.num_cycles` for cyclic inputs: the shipped cyclic input runs
500 cycles (to steady state), which takes a long time. `info` is mutable —
set `info.num_cycles = 3` before `setup_mocca_case` for a short run.

## Building a case directly

The examples are the ground truth. Mocca's source path is given to you up front;
list the examples there and read the one that fits your task before writing your
own chain:

```text
glob("/.../Mocca/examples/*.jl")
```

The shipped set covers the cyclic four-stage VSA reference, the single-pass
breakthrough (DCB) variant, an explicit-physics composition for
non-reference parameters, and the JSON, optimization, and history-matching
routes. Read whichever fits, and follow its own call chain rather than
reconstructing it from memory — the internal API moves. For a docstring,
stay in the REPL: `run_julia("@doc Mocca.setup_mocca_case")`.

## Reading results

`states` is a vector of per-timestep states. Each state maps a field to
its per-cell values, in SI units:

- `state[:Pressure]`: vector over cells, in Pa (divide by `si_unit(:bar)`).
- `state[:y]`: 2×ncells matrix of gas mole fractions, rows `[CO2, N2]`.
- `state[:Temperature]`, `state[:WallTemperature]`: kelvin.

Cell 1 is the feed/product end (LHS), cell `ncells` the outlet (RHS).
Example probes:

```julia
states[end][:y][1, 1]                                   # final CO2 fraction at the feed end
maximum(maximum(s[:Temperature]) for s in states)       # peak column temperature
maximum(s[:Pressure][end] for s in states) / bar        # peak outlet pressure, bar
```

There is no built-in product purity or CO2 recovery output: those are
cycle-integrated quantities over stage boundary flows. If a task needs
them, derive them explicitly from the states and say how; do not pass
off a pointwise mole fraction as a purity.

## Result export and plotting

```julia
Mocca.export_cell_results("results.csv", case, states, timesteps; format = "csv")
```

Call Mocca's native plotters through `plot_julia`; they are backend-agnostic
and captured automatically:

- `plot_outlet(case, states, timesteps)`: outlet concentrations / breakthrough
- `plot_state(state, model)`, `plot_cell(states, model, timesteps, cell)`: field/cell views
- `plot_optimization_history(dict_parameters)`: calibration history

No need to avoid `display`. Pass `view=true` to inspect a plot yourself.

## Notes

- The first run in a fresh REPL compiles a substantial dependency
  graph; budget several minutes.
