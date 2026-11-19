"""BattMo suite: battery workflows against the real environment.

These instantiate the BattMo workspace environment (``needs_env``), so the
first run on a cold depot takes minutes.
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
def battmo() -> Task:
    samples = [
        # Recall-and-run: The prompt pins the physics (chen_2020 cell,
        # cc_discharge protocol), so the result is deterministic and golden-
        # checkable; only the path is the agent's choice. Goldens captured
        # 2026-06-10 from a direct (agent-free) run of the canonical example
        # in the instantiated env: start 4.154 V, end 2.417 V.
        Sample(
            id="bm1-chen-cc-discharge",
            input=(
                "Using BattMo, run a constant-current discharge of the "
                "chen_2020 cell with the default cc_discharge protocol. "
                "Report the cell voltage at the start and at the end of the "
                "discharge, in volts."
            ),
            metadata={"needs_env": True},
        ),
    ]
    return Task(
        dataset=samples,
        solver=makker_solver(simulator="battmo"),
        scorer=[
            numeric_close(4.154, 0.05),
            numeric_close(2.417, 0.05),
            numeric_answer(1.5, 5.0, count=2, order="decreasing"),
            used_tools(["run_julia"]),
            no_interpreters_via_execute(),
        ],
        time_limit=2400,
        token_limit=2_000_000,
        message_limit=80,
    )


@task
def battmo_sweep() -> Task:
    # Compose-and-compare: three runs, one comparison. Higher discharge rate delivers
    # less capacity, so the three values must come back in decreasing order;
    # a golden per rate can replace the structural check once captured.
    sample = Sample(
        id="bm3-crate-sweep",
        input=(
            "Using BattMo, run constant-current discharges of the chen_2020 "
            "cell at C-rates 0.5, 1, and 2 (the protocol's DRate). Report "
            "the discharged capacity for each rate in Ah, in that order."
        ),
        metadata={"needs_env": True},
    )
    return Task(
        dataset=[sample],
        solver=makker_solver(simulator="battmo"),
        scorer=[
            numeric_answer(0.0, 10.0, count=3, order="decreasing"),
            used_tools(["run_julia"]),
            no_interpreters_via_execute(),
            no_repeated_identical_calls(),
        ],
        time_limit=3000,
        token_limit=2_000_000,
        message_limit=100,
    )


TASKS = [battmo, battmo_sweep]
