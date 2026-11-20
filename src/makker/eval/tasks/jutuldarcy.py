"""JutulDarcy suite: seed tasks for the reservoir-simulation workflows.

These instantiate the JutulDarcy workspace environment (``needs_env``), so
the first run on a cold depot takes minutes. Checks are structural
properties that must hold for any correct run; until golden references
captured from trusted baseline runs are added alongside.
"""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.dataset import Sample

from makker.eval.scorers import (
    no_interpreters_via_execute,
    no_repeated_identical_calls,
    numeric_answer,
    numeric_close,
    used_tools,
)
from makker.eval.solver import load_eval_credentials, makker_solver

load_eval_credentials()


@task
def jutuldarcy() -> Task:
    samples = [
        # Recall-and-run: a small simulation the skills cover directly.
        # The answer check is structural (a recovery factor is a fraction);
        # a golden value belongs here once captured on a trusted baseline.
        Sample(
            id="jd1-gravity-segregation",
            input=(
                "Using JutulDarcy, set up and run a small two-phase gravity "
                "segregation case (a vertical 1x1x20 column, water over gas). "
                "Report the final water saturation range across the column as "
                "two numbers between 0 and 1."
            ),
            metadata={"needs_env": True},
        ),
    ]
    return Task(
        dataset=samples,
        solver=makker_solver(simulator="jutuldarcy"),
        scorer=[
            # A saturation is a fraction; goldens replace this once captured.
            numeric_answer(0.0, 1.0, count=2),
            used_tools(["run_julia"]),
            no_interpreters_via_execute(),
        ],
        time_limit=2400,
        token_limit=2_000_000,
        message_limit=80,
    )


@task
def jutuldarcy_rate_change() -> Task:
    # Compose-and-compare: run, change one knob, rerun, compare. Halving injection
    # lowers the final average pressure, so base-then-halved must decrease.
    # The unit in the prompt is bar while Jutul works in Pa: reporting an
    # unconverted value lands far outside the plausible range and fails.
    sample = Sample(
        id="jd3-halved-injection",
        input=(
            "Using JutulDarcy, set up a small two-phase water-injection case "
            "(a simple grid of your choice with one injector and one "
            "producer), run it, then rerun the same case with the injection "
            "rate halved. Report the average reservoir pressure at the end "
            "of each run, in bar: base case first, then halved."
        ),
        metadata={"needs_env": True},
    )
    return Task(
        dataset=[sample],
        solver=makker_solver(simulator="jutuldarcy"),
        scorer=[
            numeric_answer(1.0, 1000.0, count=2, order="decreasing"),
            used_tools(["run_julia"]),
            no_interpreters_via_execute(),
            no_repeated_identical_calls(),
        ],
        time_limit=3000,
        token_limit=2_000_000,
        message_limit=100,
    )


@task
def jutuldarcy_unit_conversion() -> Task:
    # Golden-backed unit conversion: a fully specified case whose permeability is
    # given in millidarcy. Converting it to SI is the one judgement the agent must
    # get right; leaving it unconverted (100 m^2) makes the solve diverge, so it
    # cannot reach the golden pressure. The golden was captured from a trusted run of
    # this exact case in the JutulDarcy 0.3.8 env (final average reservoir pressure
    # 203.705 bar), so `numeric_close` proves the conversion landed the right physics:
    # a deterministic check with no brittle trace matching. Re-capture the golden only
    # on a deliberate JutulDarcy upgrade, never by re-running until it matches.
    sample = Sample(
        id="jd-millidarcy-conversion",
        input=(
            "Using JutulDarcy, set up and run this exact two-phase case and report "
            "the final average reservoir pressure in bar.\n"
            "- Grid: 10 x 10 x 3 cells over a 1000 x 1000 x 30 m domain.\n"
            "- Rock: uniform porosity 0.2 and uniform permeability of 100 millidarcy.\n"
            "- Wells: a vertical injector at (1, 1) and a vertical producer at "
            "(10, 10).\n"
            "- Fluids: an immiscible liquid/vapor system with reference densities "
            "1000 and 100 kg/m^3.\n"
            "- Initial state: 150 bar, fully liquid-saturated.\n"
            "- Schedule: 12 steps of 30 days; injector on a total-rate control of one "
            "pore volume over the full schedule; producer on a 50 bar bottom-hole "
            "pressure."
        ),
        metadata={
            "needs_env": True,
            # Recorded into the session trace so a later review can judge the answer
            # against ground truth (the eval cross-check in the reviewer).
            "expected": "final average reservoir pressure about 203.7 bar",
        },
    )
    return Task(
        dataset=[sample],
        solver=makker_solver(simulator="jutuldarcy"),
        scorer=[
            # Golden from a trusted 0.3.8 run; the tolerance absorbs solver noise but
            # is far tighter than the gap to any unconverted-permeability result.
            numeric_close(203.705, 8.0),
            used_tools(["run_julia"]),
            no_interpreters_via_execute(),
        ],
        time_limit=2400,
        token_limit=2_000_000,
        message_limit=120,
    )


TASKS = [jutuldarcy, jutuldarcy_rate_change, jutuldarcy_unit_conversion]
