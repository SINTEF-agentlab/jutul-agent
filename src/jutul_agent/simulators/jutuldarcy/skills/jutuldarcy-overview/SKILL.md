---
name: jutuldarcy-overview
description: High-level JutulDarcy workflow, example discovery, and result unpacking
---

# JutulDarcy orientation

## When to use

Use this skill for general JutulDarcy setup, example discovery, and result inspection.

JutulDarcy is a reservoir simulator on the Jutul AD framework. A simulation is
six steps that you compose in order:

1. **Grid** - `CartesianMesh(dims, physical_dims)` for structured cases.
   Unstructured grids come from `UnstructuredMesh` or `.DATA`/MRST imports.
2. **Domain** - `reservoir_domain(grid; permeability=K, porosity=ϕ)` attaches
   petrophysics. SI units throughout: `Darcy = si_units(:darcy)`.
3. **Wells** - `setup_vertical_well(domain, i, j; name=:P)` or
   `setup_well(domain, [(i,j,k), ...]; name=:I)`. Pass via `wells=[...]` to
   `setup_reservoir_model`.
4. **Fluid system** - `ImmiscibleSystem`, `BlackOilSystem`,
   `CompositionalSystem`. Pick the smallest system that captures the physics.
5. **Model + state + forces** - `setup_reservoir_model(domain, sys; wells=...)` ->
   `setup_reservoir_state(...)` -> controls + forces.
6. **Validate, then run** - assemble the `JutulCase`, run
   `JutulDarcy.CaseValidation.validate` on it, reconcile anything it flags, then
   `simulate_reservoir`. Step 6 is part of running a simulation, not an optional
   add-on — see below.

## Validate the case before simulating

JutulDarcy 0.4 runs case validation by default in `simulate_reservoir`.
Keep it enabled. Also inspect the assembled case explicitly before every run,
including small and example-based setups: the default validator reports problems
but does not necessarily stop the solve. Its reservoir-engineering checks cover
SI ranges for permeability, porosity, time steps, well rates and pressures:

```julia
case = JutulCase(model, dt, forces; state0, parameters)   # what simulate_reservoir builds internally
ok, messages = JutulDarcy.CaseValidation.validate(case)
```

It prints a report and returns `(ok, messages)`; `ok` is false when there are
warnings or errors. **Errors** (e.g. negative porosity) mean the simulation will
likely fail — fix them before running. **Warnings** flag values outside the usual
range (permeability that looks like millidarcy left unconverted, time steps that
look like days not seconds, rates that look per-day) — reconcile each against what
you intended; the printed hints name the likely fix (e.g.
`convert_to_si(val, "millidarcy")`). Then simulate the validated case with
progress visible: `result = simulate_reservoir(case; info_level = 1)`. Prefer
`info_level = 1` so you can see the run progress; use `info_level = 2` for a
case that may struggle to converge, to watch the nonlinear iterations.

Treat these warnings as stop-and-fix signals, not noise. Do not run past an
unphysical value (or one the solver flags during the run, such as an unreasonable
pressure) by forcing the solve through; fix the input it points to first. When a
value was only a rough guess (a rate, an injected volume), pick a physically
reasonable one rather than preserving the guess.

## Finding what you need

Example layout and APIs change between versions, so find them on disk rather
than guessing. JutulDarcy's source path is given to you up front (read-only
depot source); browse it directly with the file tools (see the
`workspace-and-source` skill):

```text
# JutulDarcy source path is in your system prompt -> /.../JutulDarcy/<hash>
glob("/.../JutulDarcy/examples/**/*.jl")                 # discover layout
grep("setup_well", path="/.../JutulDarcy/src")           # find an API
read_file("/.../JutulDarcy/examples/introduction/wells_intro.jl")
```

For docstrings, stay in the REPL: `run_julia("@doc setup_reservoir_model")`.

For idiomatic "how do I do X" patterns, the examples directory is the
authoritative reference. `setup_well` and `setup_vertical_well` are the
recommended well constructors.

For SPE11/CSP11 cases, inspect the built-in `JutulDarcy.CSP11` module and its
`src/CSP11/setup.jl` rather than recreating the benchmark setup.

Industry-standard `.DATA` decks run through `setup_case_from_data_file`;
the classic SPE benchmark decks ship with GeoEnergyIO's test data:

```julia
pth = GeoEnergyIO.test_input_file_path("SPE1", "SPE1.DATA")
case = setup_case_from_data_file(pth)
result = simulate_reservoir(case; info_level = 1)
```

## Result unpacking

```julia
result = simulate_reservoir(state0, model, dt; parameters, forces)
wd, states, t = result   # well data, reservoir states, time vector
```

Well outputs are indexed by name: `wd[:Producer][:bhp]`, `wd[:Producer][:rate]`,
etc. `keys(wd[:Producer])` lists what is available.

Three shape facts that are easy to guess wrong (each wrong guess is a KeyError):

- `states[i]` is **reservoir-only and flat**: `states[end][:Saturations]` is an
  `(nphases, ncells)` matrix, `states[end][:Pressure]` a vector. There is no
  `:Reservoir` key here.
- `state0` from `setup_reservoir_state` is the opposite — **keyed by submodel**:
  `state0[:Reservoir][:Saturations]`, with well and facility submodels alongside.
  Wells can be merged into shared submodels; do not assume each well name is a
  top-level state key. Use `well_symbols(model)` and the named well outputs.
- `result` itself is property-accessed (`result.states`, `result.wells`,
  `result.time`); `result[2]` is a MethodError. Per-cell pore volume comes from
  `pore_volume(model, parameters)`.

## Plotting

Use the **native plotters** through `plot_julia`. The tool selects WGLMakie for
the web interface and GLMakie for native capture, and saves the figure automatically:

- `plot_reservoir(model)` — 3D reservoir mesh + well trajectories
- `plot_reservoir(model, states[end])` — a state colored on the 3D mesh. Pass the
  whole state dict, or a per-cell **vector** like `states[end][:Saturations][2, :]`
  — not a scalar slice, which raises "No plottable properties found".
- `plot_well_results(wd)` — well rate / BHP dashboard
- `plot_cell_data(physical_representation(domain), field)` — a scalar field on the mesh
- `plot_co2_inventory(t, inv)` — CO2 inventory over time (2D)
- `plot_model_graph(model)` / `plot_variable_graph(reservoir_model(model))` —
  model / variable dependency graphs. These live in a Jutul package **extension**,
  so run `using GraphMakie, NetworkLayout, LayeredLayouts` first to activate them
  (the packages are already in the env). Use the real plotter; don't hand-draw a
  graph.

Just call them — no backend juggling, no need to return a `Figure`. In an
interactive session the tool shows the figure by default. Set `window=false` on
**plot_julia**, not on `plot_reservoir`, to suppress it; pass `view=true` to inspect
a plot yourself.

`plot_reservoir` opens the fancy 3D viewer by default and converts displayed
values to metric units. Pass `gui=false` for a static figure, `fancy=false` for
the fixed-axis viewer with controls, or `convert_units=false` for raw values.
Let the plotter choose its default proportions unless the user requests a change.

**Plot from the live REPL bindings.** Your run already left `model`, `states`,
`wd` in the REPL — call the plotter straight on them. Do **not** rebuild the case
and re-run `simulate_reservoir` inside `plot_julia`; that re-simulates on every
plot. Re-run only if you changed the setup.

For a **static artifact** with the field and wells but no GUI menus, use the
native plotter. Select the field with `key`, report step with `step`, and phase
or component of a matrix field with `row`:

```julia
plot_reservoir(model, states; gui=false, key=:Saturations,
    step=length(states), row=2)
```

Both original field names and displayed names with units are accepted. The
defaults are the first alphabetically sorted field, step 1 and row 1, so specify
them when showing a particular result.

## Parallel simulation and optimization

For larger cases, `simulate_reservoir(case; mode=:ka)` uses the kernel-based CPU
backend. Benchmark it against the default on the actual case; small cases may
not benefit. This does not change the Julia session's thread count.

GPU modes require a supported device and the corresponding package in the active
environment: load `CUDA` for `mode=:ka_cuda`, or `AMDGPU` for `mode=:ka_amd`.
These packages are not part of the default agent environment. Inspect the installed
`docs/src/man/advanced/gpu.md` and backend extensions before configuring a GPU run.

For history matching, inspect `setup_reservoir_dict_optimization`,
`optimize_reservoir` and `docs/src/man/basics/adjoints.md`. The optimization and
sensitivity helpers accept `mode` directly; pass e.g. `mode=:ka` at the top level,
not inside `simulator_arg`, where the new top-level default would override it.
Upstream recommends `optimizer=:lbfgs_qp` for history matching, with bounded,
log-scaled permeability and appropriately weighted observations.
