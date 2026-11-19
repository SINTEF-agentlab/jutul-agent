"""Usage suite: everyday questions a simulator user actually asks.

These are the "unit tests for the agent harness": each sample pins one
ordinary interaction (look up an API in the installed package, pull a
number out of loaded parameters, browse the shipped examples, do a small
computation on workspace data) and checks that the answer is grounded in
the session rather than recalled or invented. None of them runs a long
simulation, so the suite stays cheap.
"""

from __future__ import annotations

import math

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import includes

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
def usage_jutuldarcy() -> Task:
    # API lookup grounded in the installed source: the answer is a specific
    # function name, and the prompt forbids answering from memory.
    sample = Sample(
        id="use-jd-well-api",
        input=(
            "Which JutulDarcy function creates a vertical well that "
            "perforates every layer at a given (i, j) column? Verify "
            "against the installed package, not from memory, and give "
            "the function name."
        ),
        target="setup_vertical_well",
        metadata={"needs_env": True, "simulator": "jutuldarcy"},
    )
    return Task(
        dataset=[sample],
        solver=makker_solver(),
        scorer=[includes(), no_interpreters_via_execute()],
        time_limit=1200,
        token_limit=1_000_000,
        message_limit=40,
    )


@task
def usage_battmo() -> Task:
    # Pull a physical number out of loaded parameters: requires actually
    # loading chen_2020 in the session (the structural range rejects
    # refusals and unit slips; a golden can replace it once captured).
    sample = Sample(
        id="use-bm-cell-capacity",
        input=(
            "Using BattMo, load the chen_2020 cell parameters and compute "
            "the cell's nominal capacity. Report it in Ah."
        ),
        metadata={"needs_env": True, "simulator": "battmo"},
    )
    return Task(
        dataset=[sample],
        solver=makker_solver(),
        scorer=[
            numeric_answer(0.1, 50.0, count=1),
            used_tools(["run_julia"]),
            no_interpreters_via_execute(),
            no_repeated_identical_calls(),
        ],
        time_limit=1800,
        token_limit=1_000_000,
        message_limit=50,
    )


@task
def usage_mocca() -> Task:
    # Example discovery in the installed package source: the shipped
    # example names are not guessable from memory.
    sample = Sample(
        id="use-mc-list-examples",
        input=("List the example scripts that ship with the installed Mocca package, by filename."),
        target="haghpanah",
        metadata={"needs_env": True, "simulator": "mocca"},
    )
    return Task(
        dataset=[sample],
        solver=makker_solver(),
        scorer=[includes(), no_interpreters_via_execute()],
        time_limit=1200,
        token_limit=1_000_000,
        message_limit=40,
    )


# Deterministic workspace data; the golden mean is computed from the same
# expression that generates the fixture.
_ROWS = [(i, round(2.0 * math.sin(i / 3.0) + 5.0, 5)) for i in range(40)]
_DATA_CSV = "t,y\n" + "".join(f"{t},{y}\n" for t, y in _ROWS)
_MEAN_Y = sum(y for _, y in _ROWS) / len(_ROWS)


@task
def usage_workspace() -> Task:
    # Workspace-data roundtrip through the kernel, no simulator needed.
    sample = Sample(
        id="use-csv-mean",
        input=(
            "The workspace file data.csv has columns t and y. Compute the "
            "mean of y and the number of rows, and report both (mean to "
            "four decimal places)."
        ),
        metadata={"fixtures": {"data.csv": _DATA_CSV}},
    )
    return Task(
        dataset=[sample],
        solver=makker_solver(),
        scorer=[
            numeric_close(round(_MEAN_Y, 4), 0.001),
            numeric_close(40, 0.5),
            used_tools(["run_julia"]),
            no_interpreters_via_execute(),
        ],
        time_limit=900,
        token_limit=500_000,
        message_limit=30,
    )


TASKS = [usage_jutuldarcy, usage_battmo, usage_mocca, usage_workspace]
