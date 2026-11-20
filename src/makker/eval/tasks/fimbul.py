"""Fimbul suite: geothermal workflows against the real environment.

These instantiate the Fimbul workspace environment (``needs_env``), so the
first run on a cold depot takes minutes. Checks are structural until
goldens are captured from trusted baseline runs.
"""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.dataset import Sample

from makker.eval.scorers import (
    no_interpreters_via_execute,
    numeric_answer,
    used_tools,
)
from makker.eval.solver import load_eval_credentials, makker_solver

load_eval_credentials()


@task
def fimbul() -> Task:
    samples = [
        # Recall-and-run: the standard doublet case. Structural check:
        # plausible production temperatures in Celsius, cooling over the
        # simulated period (cold-water breakthrough).
        Sample(
            id="fb1-doublet-cooldown",
            input=(
                "Using Fimbul, set up and run a standard geothermal doublet "
                "case (the package's example setup is fine). Report the "
                "produced-water temperature at the start and at the end of "
                "the simulation, in degrees Celsius."
            ),
            metadata={"needs_env": True},
        ),
    ]
    return Task(
        dataset=samples,
        solver=makker_solver(simulator="fimbul"),
        scorer=[
            numeric_answer(1.0, 250.0, count=2, order="decreasing"),
            used_tools(["run_julia"]),
            no_interpreters_via_execute(),
        ],
        time_limit=2400,
        token_limit=2_000_000,
        message_limit=80,
    )
