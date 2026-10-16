"""Fimbul adapter."""

from __future__ import annotations

from pathlib import Path

from jutul_agent.simulators.base import SimulatorAdapter

FIMBUL = SimulatorAdapter(
    name="fimbul",
    display_name="Fimbul",
    module_dir=Path(__file__).resolve().parent,
    package_imports=("Jutul", "JutulDarcy", "Fimbul"),
    primary_package="Fimbul",
    # Warms the analytical_1d geothermal solve + plot_cell_data, baked at init. See
    # the package's src/JutulAgentFimbul.jl for the single source of the solve.
    warm_package="JutulAgentFimbul",
    domain_hints=(
        "Fimbul is a geothermal reservoir simulator built on top of JutulDarcy. "
        "It augments Darcy flow with an energy-conservation equation, "
        "transporting heat by advection and conduction. Typical workflow: a "
        "case factory (e.g. `egg_geothermal_doublet()`, `doublet_demo()`, "
        "`ates_demo()`) returns a `JutulCase` → `simulate_reservoir(case)` → "
        "inspect states (key field is `:Temperature`). Examples live under "
        "`examples/` in the installed source (`pkgdir(Fimbul)`) in three groups: "
        "`analytical/`, `production/` (doublet, EGS, AGS, coaxial BHE), and "
        "`storage/` (ATES, BTES, FTES, HTATES). Most workflows reuse "
        "JutulDarcy's grid + well constructors; its source is at "
        "`pkgdir(JutulDarcy)` and its installed examples cover those "
        "primitives."
    ),
    review_hints=(
        "Geothermal on the Darcy stack: the reservoir ranges apply, plus temperatures "
        "in Kelvin within sensible bounds."
    ),
    example_prompts=(
        "Run the geothermal doublet demo and show the temperature field around the "
        "wells over time.",
        "Plot the produced fluid temperature versus time.",
        "Set up an ATES seasonal storage case and explain the cycle.",
        "Compare the analytical 1D heat solution with the simulated result.",
    ),
)
