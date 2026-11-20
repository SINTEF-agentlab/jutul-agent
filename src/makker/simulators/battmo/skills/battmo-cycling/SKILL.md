---
name: battmo-cycling
description: Choose and customize BattMo cell parameters and cycling protocols.
---

# Cell parameters and cycling

Cell parameters describe the physical cell; the cycling protocol describes
how it is driven. Load a reference set before customizing it:

```julia
cell_parameters = load_cell_parameters(; from_default_set="chen_2020")
cycling_protocol = load_cycling_protocol(; from_default_set="cc_discharge")
```

Inspect the installed loaders and `examples/beginner_tutorials/` for available
sets, parameter names, and protocol options. Modify the inputs before building
`Simulation(model, cell_parameters, cycling_protocol)`.

Both loaders also accept `from_file_path` for a JSON file. Loaded inputs are
dictionary-like: inspect `keys(input)`, `input.all`, or
`search_parameter(input, "rate")` before editing an unfamiliar parameter.
For example, the CC-discharge protocol uses `cycling_protocol["DRate"]`.
Use `deepcopy` for independent variants so nested edits do not mutate another case.

The cell-parameter and cycling tutorials explain nested parameters and CC/CV
protocols. Use the model-selection tutorial when changing physics, and the
grid/time-resolution and solver-settings tutorials for numerical controls.
These settings have their own loaders; do not put all options in the protocol.

For a short 3D check, keep the example geometry and shorten
`Simulation(...; time_steps=...)`. Coarsening a pouch grid below its tab
dimensions can make the geometry invalid.
