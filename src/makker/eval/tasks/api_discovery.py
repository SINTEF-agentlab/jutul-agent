"""API-discovery suite: how efficiently does the agent learn an unfamiliar API?

Models often spend many turns probing how to call functions (`@doc`,
`methods`, trial-and-error in `run_julia`) before doing the work. That
exploration is not wrong (reading the real API beats guessing), but a good
harness makes it cheap. This suite measures the cost on a controlled API: the
synthetic ``MiniRes`` package (see ``_corpus``) ships in the workspace but is
not installed and is absent from any training data, so the agent must learn its
signatures from the source it can read, then call them correctly.

Hermetic and environment-free, so it is fast and reproducible::

    uv run makker eval api_discovery

Each task pairs a correctness check (a value only a correct call produces) with
the efficiency counters (:func:`tool_call_count`, :func:`file_op_count`,
:func:`julia_probe_count`). The discriminating signal is not whether the model
gets the answer (capable models do) but how many probes and reads it took to
get there, which is what a skill, prompt, or mount change aimed at faster API
discovery should move.
"""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.dataset import Sample

from makker.eval.scorers import (
    file_op_count,
    julia_probe_count,
    no_interpreters_via_execute,
    no_unresolvable_path_in_julia,
    numeric_close,
    tool_call_count,
    used_tools,
)
from makker.eval.solver import load_eval_credentials, makker_solver
from makker.eval.tasks._corpus import ROOT, corpus_fixtures

load_eval_credentials()

_LIMITS = {"time_limit": 900, "token_limit": 500_000, "message_limit": 60}

_PREAMBLE = (
    f"The Julia package under `{ROOT}` is in the workspace but is not installed, "
    "so load its source directly rather than `using` it. "
)


def _efficiency() -> list:
    """Efficiency counters: this suite is about the cost, not the answer."""
    return [tool_call_count(), file_op_count(), julia_probe_count()]


def _sample(sample_id: str, prompt: str, target: str) -> Sample:
    return Sample(
        id=sample_id,
        input=prompt,
        target=target,
        metadata={"fixtures": corpus_fixtures()},
    )


@task
def api_discovery_solver() -> Task:
    """Discover a function with a keyword argument, then run it.

    The agent must learn ``build_grid(nx, ny)`` and
    ``solve_newton(grid, perm; iters)`` from the source, then run the four-step
    iteration, which it cannot do in its head, so the value (0.0625) is
    evidence it discovered the signature and actually called it.
    """
    sample = _sample(
        "api1-newton-residual",
        _PREAMBLE + "Using its Newton solver, run 4 iterations on a 20x20 grid "
        "with permeability 500.0, and report the residual it returns.",
        target="0.0625",
    )
    return Task(
        dataset=[sample],
        solver=makker_solver(),
        scorer=[
            numeric_close(0.0625, 0.001),
            used_tools(["run_julia"]),
            no_unresolvable_path_in_julia(),
            no_interpreters_via_execute(),
            *_efficiency(),
        ],
        **_LIMITS,
    )


@task
def api_discovery_internal() -> Task:
    """Discover and call a non-exported (internal) function.

    The Darcy-flux kernel is not exported, so the agent cannot reach it by name
    after a plain load. It has to read the source to find it and its argument
    order, then call it (result -25000.0 for the given inputs).
    """
    sample = _sample(
        "api2-internal-darcy",
        _PREAMBLE + "It has an internal (not exported) Darcy-flux function. "
        "Call it with permeability 0.5, pressure drop 100.0, and dx 2.0, and "
        "report the value it returns.",
        target="-25000",
    )
    return Task(
        dataset=[sample],
        solver=makker_solver(),
        scorer=[
            numeric_close(-25000.0, 1.0),
            used_tools(["run_julia"]),
            no_unresolvable_path_in_julia(),
            no_interpreters_via_execute(),
            *_efficiency(),
        ],
        **_LIMITS,
    )


TASKS = [api_discovery_solver, api_discovery_internal]
