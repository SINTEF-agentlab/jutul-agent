"""Plotting suite: a figure must actually exist, not just be described.

Needs the simulator environment (Makie) and a display, so the solver
instantiates the workspace env and starts a virtual display on headless
Linux. First run on a cold Julia depot is slow; later runs reuse the
shared precompile cache.
"""

from __future__ import annotations

import random

from inspect_ai import Task, task
from inspect_ai.dataset import Sample

from makker.eval.scorers import (
    artifact_produced,
    no_interpreters_via_execute,
    reads_digit,
    tool_call_matches,
    used_tools,
)
from makker.eval.solver import load_eval_credentials, makker_solver

load_eval_credentials()


@task
def plotting() -> Task:
    sample = Sample(
        id="x5-headless-plot",
        input=(
            "Plot sin(x) for x in [0, 2pi] and save it as a PNG in the "
            "workspace. Reply with the file path."
        ),
        metadata={"needs_env": True, "needs_display": True},
    )
    return Task(
        dataset=[sample],
        solver=makker_solver(),
        scorer=[used_tools(["plot_julia"]), artifact_produced(".png")],
        time_limit=1800,
        token_limit=2_000_000,
        message_limit=50,
    )


# A single value, encoded as the height of a bar of points. The answer is read
# off the rendered figure's y-axis, which every model does reliably -- unlike
# OCR of a numeral drawn from scatter points, which small models misread even
# when the render is clean. Two stray points sit higher than the bar, so the
# raw-data maximum is NOT the answer: the bar's height is only apparent in the
# plot, keeping the check a genuine read of the figure rather than of the
# numbers. A fixed seed keeps the golden deterministic; the bar stays in 2..7 so
# it reads on clear gridlines well below the strays.
_VALUE = str(random.Random(20260615).randint(2, 7))
_STRAYS = ((2, 9), (8, 8))  # high, off to the sides; above any bar height


def _bar_csv(height: str) -> str:
    """``x,y`` rows: a vertical bar of points at x=5 up to ``height``, plus two
    stray points higher up, so the bar height must be read off the figure rather
    than taken as the maximum of the column."""
    rows = ["x,y", *(f"5,{y}" for y in range(int(height) + 1))]
    rows += [f"{x},{y}" for x, y in _STRAYS]
    return "\n".join(rows) + "\n"


_POINTS_CSV = _bar_csv(_VALUE)


@task
def plotting_vision() -> Task:
    # A genuine vision check that stays robust: the value is the height of a bar,
    # read off the y-axis of the agent's own plot (view=true is scored from the
    # trace). Two stray points sit above the bar, so the answer is not the
    # maximum of the raw coordinates -- it has to come from the rendered figure.
    sample = Sample(
        id="x6-read-the-bar",
        input=(
            "The workspace file points.csv has two columns, x and y. Plot the "
            "points as a scatter of y against x with GLMakie, with the y-axis "
            "fixed from 0 to 10 and integer gridlines, in a figure about 600 by "
            "600 pixels. Most of the points form a single vertical bar; two "
            "stray points sit higher up, off to the sides. View the figure so "
            "the image comes back to you, then read off the y-axis the height "
            "the bar reaches (ignore the two stray points) and reply with only "
            "that integer."
        ),
        metadata={
            "fixtures": {"points.csv": _POINTS_CSV},
            "needs_env": True,
            "needs_display": True,
        },
    )
    return Task(
        dataset=[sample],
        solver=makker_solver(),
        scorer=[
            reads_digit(_VALUE),
            tool_call_matches("plot_julia", r'"view":\s*true'),
            artifact_produced(".png"),
            no_interpreters_via_execute(),
        ],
        time_limit=1800,
        token_limit=2_000_000,
        message_limit=50,
    )


TASKS = [plotting, plotting_vision]
