---
name: jutuldarcy-overview
description: Set up JutulDarcy reservoir simulations, find examples, and inspect results.
---

# JutulDarcy

JutulDarcy simulates multiphase flow in porous media. Start from the closest
installed example under `examples/`; its setup is the reference for the
chosen physics and input format.

## Set up and run

1. Build a grid with `CartesianMesh(dims, physical_dims)`, or follow the
   unstructured-grid or imported-deck example for a more complex geometry.
2. Attach rock data with
   `reservoir_domain(grid; permeability=K, porosity=ϕ)`. Use SI inputs;
   `si_unit(:darcy)`, `si_unit(:bar)`, and `si_unit(:day)` provide conversions.
3. Construct wells and choose the fluid system, such as `ImmiscibleSystem`,
   `BlackOilSystem`, or `CompositionalSystem`. Match the phases to the problem.
4. Build the model, initial state, and forces, then simulate. See
   `jutuldarcy-wells` for controls and time-varying schedules.

```julia
model, parameters = setup_reservoir_model(domain, system; wells, extra_out=true)
# Assemble state0, dt and forces for this model.
result = simulate_reservoir(state0, model, dt; parameters, forces, info_level=1)
```

Without `extra_out=true`, `setup_reservoir_model` returns only the model.
An assembled `JutulCase` can be run with `simulate_reservoir(case)`.

`simulate_reservoir` validates the case by default. Review its messages:
validation can report problems without stopping the solve. For a separate
setup check, use `ok, messages = JutulDarcy.CaseValidation.validate(case)`.
`ok` is false for warnings as well as errors; interpret the messages against
the intended physical setup. Use `info_level=2` to inspect nonlinear iterations
when convergence is difficult.

Use `setup_case_from_data_file` for industry `.DATA` decks and the built-in
`JutulDarcy.CSP11` module for SPE11/CSP11 benchmarks.
GeoEnergyIO supplies example decks through
`GeoEnergyIO.test_input_file_path("SPE1", "SPE1.DATA")`; inspect the installed
deck examples for setup and import options.

## Results

`result.states`, `result.wells`, and `result.time` provide the outputs;
`wd, states, t = result` also works. Integer indexing of `result` does not.

- Each `states[i]` is a flat reservoir dictionary: `:Pressure` is a cell vector
  and `:Saturations` is a `(nphases, ncells)` matrix.
- Initial states from `setup_reservoir_state` are keyed by submodel, such as
  `state0[:Reservoir]`. Wells may share submodels; use `well_symbols(model)`
  rather than assuming a submodel per well name.
- Well outputs remain keyed by name, for example `wd[:Producer][:bhp]`.
  Inspect available fields with `keys(wd[:Producer])`.

Use `pore_volume(model, parameters)` for the cell pore volumes; these are not
obtained by indexing the simulation result.

Use `plot_reservoir(model, states)` through `plot_julia` for ordinary result
views, retaining the GUI and full history. Read `jutuldarcy-plotting` for field
selection and other native views.

## Larger simulations and optimization

`simulate_reservoir(case; mode=:ka)` selects the kernel-based CPU backend;
benchmark it on the actual case, since small cases may not benefit. It does not
change the Julia session's thread count.

GPU modes require a supported device and a package outside the default agent
environment: `CUDA` for `mode=:ka_cuda`, or `AMDGPU` for `mode=:ka_amd`.
Inspect the installed `docs/src/man/advanced/gpu.md` and backend extension
before configuring a GPU run.

For history matching, inspect `setup_reservoir_dict_optimization`,
`optimize_reservoir`, and the installed adjoint examples. Pass `mode` directly
to optimization helpers, not inside `simulator_arg`, where it is overridden.
The guidance in `docs/src/man/basics/adjoints.md` recommends
`optimizer=:lbfgs_qp`, bounded/log-scaled permeability, and balancing the
weights of rates and pressure observations.
