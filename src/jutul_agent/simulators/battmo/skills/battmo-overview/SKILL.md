---
name: battmo-overview
description: High-level BattMo workflow, example discovery, and output inspection
---

# BattMo orientation

## When to use

Use this skill for general BattMo setup, example discovery, and output inspection.

BattMo is a battery simulator on the Jutul AD framework. It models
lithium-ion (and other) cells with coupled electrochemistry, transport, and
optionally thermal effects.

A simulation is four parts you compose in order:

1. **Cell parameters** - physical and geometric description of the cell.
   `load_cell_parameters(; from_default_set = "chen_2020")` gives a curated
   NMC811/Graphite-SiOx cell; other default sets exist via the same loader.
2. **Cycling protocol** - driving signal.
   `load_cycling_protocol(; from_default_set = "cc_discharge")` for constant-
   current discharge; other protocols are loaded the same way.
3. **Model** - `LithiumIonBattery()` for the standard P2D system; specialised
   constructors exist for full-cell, thermal, and 3D variants.
4. **Simulation + solve** -
   `sim = Simulation(model, cell_parameters, cycling_protocol)` performs
   validation; check `sim.is_valid`. Then `sol = solve(sim)` runs it.
   `solve` is expensive (a full simulation): bind `sol = solve(sim)` **once**
   and reuse `sol` in later `run_julia` / `plot_julia` calls — the REPL keeps
   it. Do not re-run `solve` just to inspect or plot the result.

## Finding what you need

BattMo's source is read-only depot source; its path is given to you up front, so
browse it directly with the file tools (see the `workspace-and-source` skill):

```text
# BattMo source path is in your system prompt -> /.../BattMo/<hash>
glob("/.../BattMo/examples/beginner_tutorials/*.jl")    # best starting point
grep("load_cell_parameters", path="/.../BattMo/src")    # locate APIs and uses
read_file("/.../BattMo/examples/beginner_tutorials/2_run_a_simulation.jl")
```

For docstrings, stay in the REPL: `run_julia("@doc LithiumIonBattery")`.

## Result inspection

`solve` returns a `SimulationOutput`. Its `time_series` field is a
**`Dict{String, Any}` mapping each output variable to a `Vector` over the
report steps** — not a list of per-timestep state objects. Index it by
variable name; never iterate it or index it with an integer.

```julia
sol = solve(sim)               # SimulationOutput
ts  = sol.time_series          # Dict{String,Any}: variable name => Vector over steps
keys(ts)                       # discover what's available before assuming names
t = Float64.(ts["Time"])       # seconds
V = Float64.(ts["Voltage"])    # cell voltage
I = Float64.(ts["Current"])    # cell current
```

The vectors can be `Vector{Any}`, so wrap with `Float64.(…)` before plotting
or doing arithmetic. If a key you expect is missing, call `keys(ts)` and use
what's actually there. `sol` has no `keys`/`getindex` of its own — go through
`sol.time_series`.

## Plotting

Read `battmo-plotting` for native dashboards, variable-expression plots, cell
curves, and 3D viewers. Reuse the solved output; start with
`plot_dashboard(sol; new_window=false)` through `plot_julia`.
Use `plot_type="line"` when the user wants to move through the cell's states.

## Inputs from MATLAB

`load_matlab_battmo_input(filename)` reads the MATLAB BattMo `.mat` format
for cross-validation against legacy cases (see `examples/example_battery.jl`
for a reference run).