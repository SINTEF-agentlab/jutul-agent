"""Mocca adapter."""

from __future__ import annotations

from pathlib import Path

from jutul_agent.simulators.base import SimulatorAdapter

MOCCA = SimulatorAdapter(
    name="mocca",
    display_name="Mocca",
    module_dir=Path(__file__).resolve().parent,
    package_imports=("Jutul", "Mocca"),
    primary_package="Mocca",
    # Warms the DCB quick-start simulate_process + plot_outlet, baked at init. See
    # the package's src/JutulAgentMocca.jl for the single source of the solve.
    warm_package="JutulAgentMocca",
    domain_hints=(
        "Mocca (MOdelling for Carbon Capture) simulates adsorption-based "
        "CO2 capture processes (pressure / temperature / vacuum swing "
        "adsorption) on the Jutul AD framework. Typical workflow: "
        "`Mocca.parse_input(json_filepath)` → `(constants, info)` → "
        "`Mocca.setup_mocca_case(constants, info)` → `(case, ts_config)` → "
        "`Mocca.simulate_process(case; timestep_selector_cfg=ts_config, "
        "output_substates=true)`. Built-in plotting: `Mocca.plot_outlet"
        "(case, states, timesteps)`. Examples live under `examples/` in the "
        "installed source (`pkgdir(Mocca)`): `dcb_haghpanah_2013_co2_n2.jl` "
        "(direct column breakthrough), `cyclic_vsa_haghpanah_2013_co2_n2.jl` "
        "(4-stage VSA), `custom_setup_cyclic_vsa.jl`, plus `optimization.jl` "
        "and `history_matching.jl`. Reference JSON inputs ship beside the "
        'package at `joinpath(pkgdir(Mocca), "models", "json")`.'
    ),
    review_hints=("Adsorption/separation: mole fractions, purities and recoveries lie in 0..1."),
    example_prompts=(
        "Run the direct column breakthrough example and plot the outlet composition over time.",
        "Simulate a 4-stage VSA cycle and report the CO2 purity and recovery.",
        "Walk me through setting up a custom cyclic VSA process.",
        "Compare breakthrough curves for two different feed compositions.",
    ),
)
