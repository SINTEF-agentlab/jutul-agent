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
    # Warms process simulation and native bed/stream plots, baked at init. See
    # the package's src/JutulAgentMocca.jl for the single source of the solve.
    warm_package="JutulAgentMocca",
    domain_hints=(
        "Mocca (MOdelling for Carbon Capture) simulates adsorption-based "
        "CO2 capture processes (pressure / temperature / vacuum swing "
        "adsorption) on the Jutul AD framework. Typical workflow: "
        '`Mocca.read_process("haghpanah_2013_four_stage_vsa")` → '
        "`Mocca.simulate(process; cycles=...)` or "
        "`Mocca.cyclic_steady_state(process)` → `Mocca.performance(result)`. "
        "A process contains beds and flow devices; result states are keyed by "
        "their names. Native plots cover bed histories/profiles and flowsheet "
        "streams. Read the Mocca skills for data shapes and plot calls. "
        "Installed examples include `getting_started.jl`, `breakthrough.jl`, "
        "`process_from_parts.jl`, `two_bed_vsa.jl`, `optimization.jl` and "
        "`history_matching.jl`; process and adsorbent JSON files are under "
        "`models/library/`."
    ),
    review_hints=(
        "Adsorption/separation: mole fractions and purities lie in 0..1; "
        "recovery should too at cyclic steady state. "
        "Steady-cycle claims require result.converged; transient recovery can "
        "include inventory carried from earlier cycles. Use performance(result) "
        "for stream-integrated metrics, not a cell's composition."
    ),
    example_prompts=(
        "Run the direct column breakthrough example and plot the outlet composition over time.",
        "Simulate a 4-stage VSA cycle and report the CO2 purity and recovery.",
        "Walk me through setting up a custom cyclic VSA process.",
        "Compare breakthrough curves for two different feed compositions.",
    ),
)
