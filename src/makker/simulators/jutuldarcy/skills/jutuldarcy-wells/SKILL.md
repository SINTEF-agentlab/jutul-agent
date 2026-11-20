---
name: jutuldarcy-wells
description: Construct wells, controls, and forces for JutulDarcy reservoir simulations.
---

# Wells and controls

Start from the installed wells introduction for an end-to-end case.
A well setup consists of a well object, a control, and forces.

## Wells and model

`setup_vertical_well` perforates every layer in a column. `setup_well`
accepts cell indices or logical `(i, j, k)` tuples for explicit perforations.
For a structured domain:

```julia
Prod = setup_vertical_well(domain, 1, 1; name=:Producer)
Inj = setup_well(domain, [(nx, ny, 1)]; name=:Injector)
model, parameters = setup_reservoir_model(domain, system;
    wells=[Inj, Prod], extra_out=true)
```

Wells can share submodels; use `well_symbols(model)` and named well outputs
rather than assuming each well has its own `model.models[name]`.

## Controls and forces

A control combines a target and a well type. For a two-phase system ordered
as `(LiquidPhase(), VaporPhase())`, a gas injector and BHP-controlled producer
can use:

```julia
I_ctrl = InjectorControl(TotalRateTarget(inj_rate), [0.0, 1.0]; density=rhoGS)
P_ctrl = ProducerControl(BottomHolePressureTarget(bhp))
controls = Dict(:Injector => I_ctrl, :Producer => P_ctrl)
forces = setup_reservoir_forces(model; control=controls)
```

Use SI rates and pressures. The injection mixture and initial saturations follow
the system's phase order; here `rhoGS` is the gas surface density. For water–oil,
use `(AqueousPhase(), LiquidPhase())` and adapt the mixture and density. A wrong
density changes the injected volume, and injecting the phase already filling
the reservoir does not test displacement of another phase. Phase viscosities
are stored in `parameters[:Reservoir][:PhaseViscosities]`, with one row per phase.

Other targets include `SurfaceWaterRateTarget` and `SurfaceLiquidRateTarget`;
check their installed docstrings for arguments and phase conventions.

## Simulate and inspect

One forces object applies throughout the schedule. For changing controls,
supply a vector with one forces object per entry in `dt`.
`dt` contains report-step durations in seconds.

```julia
# Initial liquid-filled reservoir for the liquid/gas example above.
state0 = setup_reservoir_state(model; Pressure=p0, Saturations=[1.0, 0.0])
wd, states, t = simulate_reservoir(state0, model, dt; parameters, forces)
```

The initial state is keyed by submodel (`state0[:Reservoir]`), whereas
`states[i]` is the flat reservoir state. Saturations have shape
`(nphases, ncells)`.
Inspect well outputs by name, for example `wd[:Producer][:bhp]`; use
`keys(wd[:Producer])` to discover available channels.
The worked reference is `examples/introduction/wells_intro.jl`.
