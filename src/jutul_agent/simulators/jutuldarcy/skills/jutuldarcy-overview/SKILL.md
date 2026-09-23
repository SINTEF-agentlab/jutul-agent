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

Validating the assembled case before `simulate_reservoir` is a required step on
every run — including quick, small, or example-based setups, and even when the
user only said "set up and run". Do not skip it to "keep things simple": an
unvalidated solve is the single most common way a setup silently produces wrong
numbers. JutulDarcy ships a reservoir-engineering check that knows the expected SI
ranges for permeability, porosity, time steps, well rates, pressures, etc. — it
catches the unit-conversion mistakes and unphysical values that a plain solve
would silently run with or fail cryptically on:

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
  `state0[:Reservoir][:Saturations]`, with `:Injector`/`:Producer`/`:Facility`
  alongside.
- `result` itself is property-accessed (`result.states`, `result.wells`,
  `result.time`); `result[2]` is a MethodError. Per-cell pore volume comes from
  `pore_volume(model, parameters)`.

## Plotting

Use the **native plotters** through `plot_julia` — they run on GLMakie (the
default backend) and are captured to an image automatically, headless or not:

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
interactive session a live window opens for the user by default (`window=false` to
suppress); pass `view=true` to inspect a plot yourself.

`plot_reservoir` now opens the fancy 3D viewer by default and converts displayed
values to convenient units. Pass `fancy=false` for its fixed-axis viewer or
`convert_units=false` when you need unconverted values in the plot.

**Plot from the live REPL bindings.** Your run already left `model`, `states`,
`wd` in the REPL — call the plotter straight on them. Do **not** rebuild the case
and re-run `simulate_reservoir` inside `plot_julia`; that re-simulates on every
plot. Re-run only if you changed the setup.

For a **chrome-free static artifact** (3D field + wells without `plot_reservoir`'s
GUI menus), compose the native pieces into a plain `Axis3`. Note `plot_well!` wants
the `Axis3`'s `scene` in this non-GUI path:

```julia
rmodel = reservoir_model(model)
g = physical_representation(rmodel.data_domain)
fig = Figure(size = (900, 600))
ax = Axis3(fig[1, 1]; zreversed = true, title = "Reservoir")
plt = plot_cell_data!(ax, g, collect(rmodel.data_domain[:porosity]); colormap = :viridis)
Colorbar(fig[1, 2], plt)
for (_, m) in pairs(model.models)              # overlay wells (MultiModel)
    w = physical_representation(m.data_domain)
    w isa WellDomain && plot_well!(ax.scene, g, w)
end
fig
```
