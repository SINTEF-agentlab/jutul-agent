"""Ensemble suite: the parallel-sweep workflow through ``run_ensemble``.

One task per simulator, each sweeping a real case the simulator ships or the
workspace provides: a waterflood, a battery discharge, a geothermal doublet, a
CO2-capture cycle. Each runs inside its simulator's workspace env, because
``run_ensemble`` ships in the JutulAgent runtime package that only exists there,
and worker processes inherit that env. The real check is the trace:
``run_ensemble`` (or plain ``pmap``) must appear in code the agent actually
executed, so a serial fallback or a textual claim cannot pass. A real
simulation cannot be shortcut by recalling a value or reimplementing it in
another language, which is what makes these robust mechanism checks (a cheap
arithmetic sweep, by contrast, the model could just compute directly).
"""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.dataset import Sample

from jutul_agent.eval.scorers import (
    julia_code_matches,
    no_interpreters_via_execute,
    numeric_answer,
    used_tools,
)
from jutul_agent.eval.solver import jutul_agent_solver, load_eval_credentials

load_eval_credentials()


# The reference case ships as a workspace fixture so the task measures the
# parallel-ensemble workflow, not waterflood construction. The grid is kept
# deliberately coarse and the horizon short: this suite grades the parallel
# mechanism (run_ensemble/pmap) plus the physical trend, so a fast case is
# enough and an exact golden would only add a recapture burden. Recovery still
# falls monotonically as a larger pore volume is swept by the same injected
# volume, which is the property the scorer checks.
_WATERFLOOD_CASE = """\
# Reference waterflood case: run_waterflood_case(porosity) -> oil recovery
# factor after ~2 years of injection at 1000 m^3/day on a coarse grid.
using JutulDarcy, Jutul

function run_waterflood_case(porosity::Real)
    Darcy, bar, kg, meter, day = si_units(:darcy, :bar, :kilogram, :meter, :day)
    g = CartesianMesh((8, 8, 3), (500.0, 500.0, 20.0))
    domain = reservoir_domain(g; permeability = 0.2 * Darcy, porosity = porosity)
    inj = setup_vertical_well(domain, 1, 1, name = :Injector)
    prd = setup_vertical_well(domain, 8, 8, name = :Producer)
    sys = ImmiscibleSystem((AqueousPhase(), LiquidPhase());
        reference_densities = [1000.0, 850.0] .* kg / meter^3)
    model, parameters = setup_reservoir_model(domain, sys; wells = [inj, prd], extra_out = true)
    parameters[:Reservoir][:PhaseViscosities][1, :] .= 1e-3   # water 1 cP
    parameters[:Reservoir][:PhaseViscosities][2, :] .= 5e-3   # oil 5 cP
    state0 = setup_reservoir_state(model, Pressure = 150 * bar, Saturations = [0.0, 1.0])
    dt = fill(60.0 * day, 12)
    I_ctrl = InjectorControl(TotalRateTarget(1000.0 * meter^3 / day), [1.0, 0.0],
        density = 1000.0 * kg / meter^3)
    P_ctrl = ProducerControl(BottomHolePressureTarget(100 * bar))
    forces = setup_reservoir_forces(model,
        control = Dict(:Injector => I_ctrl, :Producer => P_ctrl))
    wd, states, t = simulate_reservoir(state0, model, dt;
        parameters = parameters, forces = forces, info_level = -1)
    pv = pore_volume(model, parameters)
    so = states[end][:Saturations][2, :]
    return (sum(pv) - sum(pv .* so)) / sum(pv)
end
"""


@task
def ensembles_jutuldarcy() -> Task:
    sample = Sample(
        id="ens-jd-porosity-sweep",
        input=(
            "The workspace file waterflood_case.jl defines "
            "run_waterflood_case(porosity), which runs a fixed reference "
            "waterflood and returns its oil recovery factor. Using that "
            "function, run porosity = 0.15, 0.20, 0.25, 0.30 as a parallel "
            "ensemble across 4 workers, and report the four recovery factors "
            "in order of increasing porosity."
        ),
        metadata={
            "fixtures": {"waterflood_case.jl": _WATERFLOOD_CASE},
            "needs_env": True,
            "simulator": "jutuldarcy",
        },
    )
    return Task(
        dataset=[sample],
        solver=jutul_agent_solver(),
        scorer=[
            # Structural: four recovery factors, strictly decreasing with
            # porosity. The exact values are not pinned; the suite grades the
            # parallel mechanism and the physical trend, and a coarse fast case
            # makes an exact golden a recapture burden without adding signal.
            numeric_answer(0.0, 1.0, count=4, order="decreasing"),
            julia_code_matches(r"(run_ensemble|pmap)\s*\("),
            used_tools(["run_julia"]),
            no_interpreters_via_execute(),
        ],
        time_limit=2400,
        token_limit=1_000_000,
        message_limit=60,
    )


@task
def ensembles_battmo() -> Task:
    # Same workflow on the battery side, kept structural: higher discharge
    # rate delivers less capacity, so the three values must decrease.
    sample = Sample(
        id="ens-bm-crate-sweep",
        input=(
            "Using BattMo, run constant-current discharges of the chen_2020 "
            "cell at C-rates 0.5, 1, and 2 (the protocol's DRate), as a "
            "parallel ensemble across 3 workers, each case returning only "
            "its discharged capacity in Ah. Report the three capacities in "
            "that order."
        ),
        metadata={"needs_env": True, "simulator": "battmo"},
    )
    return Task(
        dataset=[sample],
        solver=jutul_agent_solver(),
        scorer=[
            numeric_answer(0.0, 10.0, count=3, order="decreasing"),
            julia_code_matches(r"(run_ensemble|pmap)\s*\("),
            used_tools(["run_julia"]),
            no_interpreters_via_execute(),
        ],
        time_limit=3000,
        token_limit=2_000_000,
        message_limit=100,
    )


@task
def ensembles_fimbul() -> Task:
    # The doublet is Fimbul's shipped geothermal_doublet factory, so the agent
    # drives it directly. A short horizon keeps the sweep fast; the suite grades
    # the parallel mechanism and plausible produced-water temperatures, not an
    # exact end-state (the cold-front trend needs a long horizon to resolve and
    # would only add a recapture burden here).
    sample = Sample(
        id="ens-fb-injtemp-sweep",
        input=(
            "Using Fimbul, run its standard geothermal doublet (the "
            "geothermal_doublet factory) for 25 years, reinjecting at 10, "
            "20, 30, and 40 deg C (geothermal_doublet's temperature_inj is in "
            "SI, so convert from Celsius). Run the four cases as a parallel "
            "ensemble across 4 workers, and report the produced-water "
            "temperature (deg C) at the end of each, in order of increasing "
            "injection temperature."
        ),
        metadata={"needs_env": True, "simulator": "fimbul"},
    )
    return Task(
        dataset=[sample],
        solver=jutul_agent_solver(),
        scorer=[
            numeric_answer(1.0, 250.0, count=4),
            julia_code_matches(r"(run_ensemble|pmap)\s*\("),
            used_tools(["run_julia"]),
            no_interpreters_via_execute(),
        ],
        time_limit=3000,
        token_limit=1_500_000,
        message_limit=80,
    )


@task
def ensembles_mocca() -> Task:
    # CO2-capture side. No captured reference for how the feed-end mole
    # fraction moves with cycle count, so the answer check is plausibility
    # (mole fractions) and the substance is the parallel trace check.
    sample = Sample(
        id="ens-mc-cycles-sweep",
        input=(
            "Using Mocca, run four_stage_vsa_process with its default settings "
            "for 1, 2, and 3 cycles from independent initial states, "
            "as a parallel ensemble across 3 workers, each case returning "
            "only the CO2 gas mole fraction at the feed end of the column "
            "in its final state. Report the three mole fractions in order "
            "of increasing cycle count."
        ),
        metadata={"needs_env": True, "simulator": "mocca"},
    )
    return Task(
        dataset=[sample],
        solver=jutul_agent_solver(),
        scorer=[
            numeric_answer(0.0, 1.0, count=3),
            julia_code_matches(r"(run_ensemble|pmap)\s*\("),
            used_tools(["run_julia"]),
            no_interpreters_via_execute(),
        ],
        time_limit=3000,
        token_limit=2_000_000,
        message_limit=100,
    )


TASKS = [
    ensembles_jutuldarcy,
    ensembles_battmo,
    ensembles_fimbul,
    ensembles_mocca,
]
