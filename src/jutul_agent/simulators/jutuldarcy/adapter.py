"""JutulDarcy adapter."""

from __future__ import annotations

from pathlib import Path

from jutul_agent.simulators.base import SimulatorAdapter

JUTULDARCY = SimulatorAdapter(
    name="jutuldarcy",
    display_name="JutulDarcy",
    module_dir=Path(__file__).resolve().parent,
    package_imports=("Jutul", "JutulDarcy"),
    primary_package="JutulDarcy",
    # Warms simulate_reservoir + plot_cell_data, baked GLMakie-aware at init. See
    # the package's src/JutulAgentJutulDarcy.jl for the single source of the solve.
    warm_package="JutulAgentJutulDarcy",
    domain_hints=(
        "JutulDarcy is a fully-differentiable porous-media reservoir simulator "
        "built on the Jutul framework. Core concepts: `CartesianMesh` or unstructured "
        "grid → `reservoir_domain(g; permeability, porosity)` → wells "
        "(`setup_vertical_well`, `setup_well`) → fluid system (`ImmiscibleSystem`, "
        "`BlackOilSystem`, `CompositionalSystem`) → `setup_reservoir_model(domain, "
        "sys; wells=...)` → initial state + controls + forces → `simulate_reservoir`. "
        "Units come from `si_units(:darcy, :bar, :day, ...)`. Examples live under "
        "`examples/` in the installed source (`pkgdir(JutulDarcy)`): `wells_intro.jl`, "
        "`intro_example.jl`, `compositional_5components.jl`, `data_input_file.jl`. "
        "Spatially-correlated permeability/porosity fields (optionally conditioned on "
        "well data) come from GeoStats.jl — see the `jutuldarcy-geostatistics` skill."
    ),
    review_hints=(
        "Reservoir flow: permeability in m^2 (~1e-16..1e-11; a much larger value usually "
        "means millidarcy/darcy left unconverted), porosity in 0..1, times in seconds, "
        "SI rates and pressures."
    ),
    example_prompts=(
        "Build a small 3D reservoir with one water injector and one producer, run a "
        "short immiscible simulation, and show the interactive 3D view.",
        "Investigate how porosity changes the waterflood sweep, then write me a report.",
        "Plot the well rates and bottom-hole pressures from the last run.",
        "Set up a CO2 injection case and plot the CO2 inventory over time.",
    ),
)
