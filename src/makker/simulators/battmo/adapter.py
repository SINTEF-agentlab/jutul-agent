"""BattMo adapter."""

from __future__ import annotations

from pathlib import Path

from makker.simulators.base import SimulatorAdapter

BATTMO = SimulatorAdapter(
    name="battmo",
    display_name="BattMo",
    module_dir=Path(__file__).resolve().parent,
    package_imports=("Jutul", "BattMo"),
    primary_package="BattMo",
    # Warms the chen_2020 cell + cc_discharge solve, baked at init. See the
    # package's src/MakkerBattMo.jl for the single source of the solve.
    warm_package="MakkerBattMo",
    domain_hints=(
        "BattMo is a Jutul-framework battery simulator (lithium-ion and other "
        "chemistries, coupled electrochemistry / transport / optionally thermal). "
        "Workflow: `load_cell_parameters(; from_default_set=...)` → "
        "`load_cycling_protocol(; from_default_set=...)` → `LithiumIonBattery()` "
        "→ `Simulation(model, cell_parameters, cycling_protocol)` → `solve(sim)`. "
        "Beginner walkthroughs live in `examples/beginner_tutorials/` "
        "(numbered 1-11)."
    ),
    review_hints=(
        "Batteries: cell voltage within the chemistry's window (~2.5..4.2 V for Li-ion) "
        "and current/C-rate consistent with the nominal capacity."
    ),
    example_prompts=(
        "Run a constant-current discharge on the Chen 2020 lithium-ion cell and plot "
        "the voltage curve.",
        "Compare discharge at 0.5C, 1C, and 2C, then write a report on the usable "
        "capacity at each rate.",
        "Show how cell voltage and current evolve over the discharge.",
        "Work through the first beginner tutorial and explain each step.",
    ),
)
