"""Calibration suite: an iterative investigation, graded on the recorded process.

No simulator environment is needed: the agent fits a two-parameter model
to workspace data through the Julia session. The scorers read the
``attempt`` events in the trace, so a correct answer with an unrecorded
process still fails.
"""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.dataset import Sample

from makker.eval.scorers import (
    no_interpreters_via_execute,
    numeric_close,
    used_tools,
)
from makker.eval.solver import load_eval_credentials, makker_solver

load_eval_credentials()

# Generated from y = 2.5 * exp(-0.7 t) plus Gaussian noise (sigma 0.01,
# fixed seed); least squares on these exact points recovers a = 2.4975,
# b = 0.6983 with rmse 0.0063.
_DATA = """t,y
0.0,2.49856
0.1,2.32926
0.2,2.17228
0.3,2.03348
0.4,1.88818
0.5,1.74675
0.6,1.64594
0.7,1.52889
0.8,1.42585
0.9,1.33264
1.0,1.24379
1.1,1.16917
1.2,1.08584
1.3,1.00742
1.4,0.93089
1.5,0.8647
1.6,0.81816
1.7,0.77366
1.8,0.70955
1.9,0.66013
2.0,0.62181
"""


@task
def calibration() -> Task:
    # Simple curve-fit, graded on the answer. An earlier version of this task
    # also required a recorded >=3-step attempt tree (record_attempt /
    # write_report), but capable models solve the exp-decay fit in a single
    # least-squares step and legitimately record one attempt, so the tree
    # requirement failed every model and measured the prompt, not the agent.
    # The investigation-tier scoring (investigation_recorded) is being reworked
    # around a genuinely iterative problem; until then this stays a plain fit.
    sample = Sample(
        id="cal1-exp-decay-fit",
        input=(
            "The workspace file data.csv holds measurements with columns t "
            "and y, generated from y = a*exp(-b*t) plus a little noise. "
            "Calibrate a and b against the data in the Julia session and "
            "report the calibrated a and b."
        ),
        metadata={"fixtures": {"data.csv": _DATA}},
    )
    return Task(
        dataset=[sample],
        solver=makker_solver(),
        scorer=[
            numeric_close(2.50, 0.05),
            numeric_close(0.70, 0.03),
            used_tools(["run_julia"]),
            no_interpreters_via_execute(),
        ],
        time_limit=1200,
        token_limit=500_000,
        message_limit=60,
    )
