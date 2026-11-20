---
name: battmo-overview
description: Set up BattMo battery simulations and inspect their outputs.
---

# BattMo

BattMo simulates battery electrochemistry and transport, with optional thermal
physics. Choose a model and parameter set together; geometry and physical
options determine which inputs are required.

Start from the closest installed example in `examples/beginner_tutorials/`,
or a specialized example for thermal, 3D, or MATLAB cases.

## Set up and run

The workflow combines cell parameters, a cycling protocol, and a model:

```julia
using BattMo
cell_parameters = load_cell_parameters(; from_default_set="chen_2020")
cycling_protocol = load_cycling_protocol(; from_default_set="cc_discharge")
model = LithiumIonBattery()
sim = Simulation(model, cell_parameters, cycling_protocol)
@assert sim.is_valid
sol = solve(sim)
```

`LithiumIonBattery()` uses the standard P2D model. Inspect the model-selection
tutorial for thermal and 3D settings. Review validation failures before solving,
and retain `sol` for analysis and plotting instead of solving again.

## Results and input formats

`sol.time_series` maps variable names to vectors over report steps; inspect
`keys(sol.time_series)` before selecting outputs. It is not a vector of states,
and `sol` itself is a `SimulationOutput`, not a dictionary.

```julia
t = sol.time_series["Time"]       # seconds
V = sol.time_series["Voltage"]    # volts
I = sol.time_series["Current"]    # amperes
```

Vectors can have element type `Any`; use `Float64.(…)` when a custom plot
needs a concrete numeric vector. Spatial data is in `sol.states`, while
`sol.jutul_output` holds the underlying Jutul model and states.
`sol.metrics` contains cycle-based summaries; `print_info(sol)` lists the
available outputs.

For MATLAB-format input, follow `examples/example_matlab.jl` and
`load_matlab_input(filename)`. This follows a lower-level Jutul workflow;
do not assume its result has the `SimulationOutput` layout above.

Use `battmo-cycling` for inputs and `battmo-plotting` for native result views.
