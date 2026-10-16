---
name: mocca-plotting
description: Plot Mocca bed histories, spatial and stage-end profiles, flowsheet streams, and comparisons using native figures.
---

# Mocca plotting

Use native plotters through `plot_julia`, reusing the solved process and
result. Include `import Mocca` if needed.

## Bed views

Bed plotters take one bed's model and states, not the full flowsheet:

```julia
bed = process.flowsheet[:Bed]
bed_states = [s[:Bed] for s in result.states]
ncells = length(bed_states[end][:Pressure])
Mocca.plot_cell(bed_states, bed, result.timesteps, ncells)
```

This is the top/outlet history in the reference breakthrough setup; cell 1
is the bottom/feed end. For other flowsheets, inspect bed names and connected
ports before choosing the cell. Use `Mocca.plot_state(bed_states[end], bed)`
for the final spatial profile.

Pass `result.timesteps` unchanged: these are durations and the plotters
accumulate them internally. A single final state cannot show a breakthrough
history. Use `cumsum` only for a custom time axis.

## Streams and stage-end profiles

For cycle results, show native device flows and compositions, with stage shading:

```julia
stages = Mocca.stages(process)
Mocca.plot_flowsheet_streams(result.states, process.model, result.timesteps;
    stages=stages)
```

Flow is positive from each device's inlet to outlet. The native plot handles
upstream composition when flow reverses. For bed profiles at stage ends:

```julia
stage_ends = cumsum([st.duration for st in stages])
Mocca.plot_profiles(result.states, result.timesteps, bed, stage_ends;
    labels=[st.name for st in stages], unit=:Bed)
```

Profile times are relative to the start of the supplied history. The example
above selects the first cycle; for a multi-cycle `simulate` result, offset
by the preceding cycles to select the last. A steady-state result holds one
cycle. `plot_profiles` normalizes loading by saturation capacity when available;
other bed plots show loading in mol/m³ of adsorbent. Keep the native labels.

## Comparisons and optimization

For independent runs of the same bed, use
`Mocca.plot_cell_comparison(runs, bed, cell; component=i)`, where each entry
is `"label" => (bed_states, timesteps)`. For spatial comparisons, pass
`"label" => bed_state` entries to `Mocca.plot_state_comparison(states, bed;
component=i)`. Find the component index from
`Mocca.component_names(process.flowsheet.gas)`; do not assume CO₂ is row 1.

For `optimise!`, plot the recorded `.history` entries (`values`, `metrics`)
as shown in `examples/optimization.jl`. The separate
`Mocca.plot_optimization_history(dprm)` accepts Jutul `DictParameters` from
history matching, not the `optimise!` return value. Check its recorded
objectives are present and positive for its default logarithmic axis.

Native bed plots use Pa and K; convert explicitly for a requested custom bar
or Celsius curve. See `src/plotting/plot.jl` for supported options and
`examples/breakthrough.jl` / `getting_started.jl` for complete calls.
Custom figures fit requested quantities or layouts the native views lack.
Keep `slot`, `view` and `window` on `plot_julia`, and use the tool's `Makie`
namespace for annotations.
