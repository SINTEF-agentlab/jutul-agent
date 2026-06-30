"""Geostatistics suite: the ``jutuldarcy-geostatistics`` skill.

``GeoStats`` ships in the JutulDarcy env (``needs_env``), so both tasks run
there. Geostatistics is stochastic, so each task grades a property that holds
regardless of the seed:

- ``geostatistics`` (golden): kriging is a linear solve with no RNG, so a
  fully specified problem has one answer per cell.
- ``geostatistics_conditioning`` (structural): conditioning collapses the
  ensemble std to ~0 at a well and leaves it near the sill far away, for any
  seed or ensemble size; the task grades that shape, not a value.
"""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.dataset import Sample

from jutul_agent.eval.scorers import (
    julia_code_matches,
    no_interpreters_via_execute,
    numeric_answer,
    numeric_close,
    used_tools,
)
from jutul_agent.eval.solver import jutul_agent_solver, load_eval_credentials

load_eval_credentials()


# Five wells on a 50x50 unit-cell grid, shared by both tasks; porosity values
# spread over a realistic 0.21..0.30.
_WELLS = (
    "- cell (10, 10): porosity 0.28\n"
    "- cell (40, 12): porosity 0.21\n"
    "- cell (25, 38): porosity 0.30\n"
    "- cell (40, 40): porosity 0.24\n"
    "- cell (12, 40): porosity 0.26"
)


@task
def geostatistics() -> Task:
    # Golden ordinary kriging: fully specified, so the estimate at a cell is one
    # deterministic number, captured from this case in the JutulDarcy 0.3.8 /
    # GeoStats 0.84.5 env. The wrong moves all miss the tolerance -- a transposed
    # read is ~0.011 off, a wrong variogram range >= 0.006, an echoed well value
    # ~0.009 -- while the overall sill cannot move the estimate. Re-capture only
    # on a deliberate GeoStats upgrade, never by re-running until it matches.
    sample = Sample(
        id="geo-kriging-estimate",
        input=(
            "Using GeoStats.jl, build an ordinary-kriging estimate of porosity "
            "on a 50 x 50 unit-cell Cartesian grid, conditioned on five wells "
            "given as logical grid cells (i, j) with their measured porosity:\n"
            f"{_WELLS}\n"
            "Use a spherical variogram with range 15.0 and sill 0.06^2 (no "
            "nugget). Report the estimated porosity at the unsampled grid cell "
            "(25, 30)."
        ),
        metadata={
            "needs_env": True,
            "expected": "ordinary-kriging porosity estimate at cell (25, 30) is about 0.2694",
        },
    )
    return Task(
        dataset=[sample],
        solver=jutul_agent_solver(simulator="jutuldarcy"),
        scorer=[
            # Tolerance absorbs reported rounding, far inside the gap to any
            # wrong move.
            numeric_close(0.2694, 0.003),
            used_tools(["run_julia"]),
            no_interpreters_via_execute(),
        ],
        time_limit=2400,
        token_limit=2_000_000,
        message_limit=60,
    )


@task
def geostatistics_conditioning() -> Task:
    # Structural: conditioning honors the data exactly, so the per-cell std is
    # ~0 at a well and near the sill (~0.06) far from data, for any seed. No
    # golden needed -- the scorer grades that shape and the trace must show the
    # conditioning call; dropping data= leaves std near the sill everywhere.
    sample = Sample(
        id="geo-conditioning-std",
        input=(
            "Using GeoStats.jl, model porosity as a Gaussian process on a "
            "50 x 50 unit-cell Cartesian grid and generate a conditional "
            "simulation ensemble of 100 realizations that honor five wells "
            "given as logical grid cells (i, j) with their measured porosity:\n"
            f"{_WELLS}\n"
            "Use a spherical variogram with range 15.0 and sill 0.06^2, and the "
            "mean of the well data as the process mean. Across the ensemble, "
            "compute the per-cell standard deviation of porosity. Report two "
            "numbers: first the ensemble standard deviation at the conditioned "
            "well cell (10, 10), then the ensemble standard deviation at the "
            "unsampled cell (25, 25)."
        ),
        metadata={
            "needs_env": True,
            "expected": (
                "std ~0 at the conditioned well (10, 10) and ~0.06 (the sill) at the "
                "unsampled cell (25, 25)"
            ),
        },
    )
    return Task(
        dataset=[sample],
        solver=jutul_agent_solver(simulator="jutuldarcy"),
        scorer=[
            # Two small std values, well below far: the signature of
            # conditioning. Exact values are seed-dependent, so not pinned.
            numeric_answer(0.0, 0.1, count=2, order="increasing"),
            # The substance: rand(proc, grid, N; data = data) -- the data=
            # keyword conditions the realizations, so a claim cannot pass.
            julia_code_matches(r"rand\([^\n]*data"),
            used_tools(["run_julia"]),
            no_interpreters_via_execute(),
        ],
        time_limit=2400,
        token_limit=2_000_000,
        message_limit=60,
    )


TASKS = [geostatistics, geostatistics_conditioning]
