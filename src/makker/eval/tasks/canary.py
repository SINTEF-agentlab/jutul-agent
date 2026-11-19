"""Canary suite: the cheapest end-to-end proof that the harness works.

One sample, no simulator environment: the agent must read a workspace file
and evaluate it through the Julia kernel. If this fails, the harness is
broken, not the physics. Run it first on any model or harness change::

    uv run makker eval canary
"""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import includes

from makker.eval.scorers import used_tools
from makker.eval.solver import load_eval_credentials, makker_solver

load_eval_credentials()


@task
def canary() -> Task:
    sample = Sample(
        id="x0-sum-from-file",
        input=(
            "Read the workspace file `/data.jl` with read_file, then use "
            "run_julia to evaluate its contents as Julia code. Reply with "
            "the numeric result."
        ),
        target="105",
        metadata={"fixtures": {"data.jl": "sum([7, 14, 21, 28, 35])\n"}},
    )
    return Task(
        dataset=[sample],
        solver=makker_solver(),
        scorer=[includes(), used_tools(["read_file", "run_julia"])],
        time_limit=600,
        token_limit=200_000,
        message_limit=50,
    )
