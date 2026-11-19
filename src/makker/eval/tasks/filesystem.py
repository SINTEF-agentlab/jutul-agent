"""Filesystem suite: resolving workspace paths the agent keeps tripping on.

The file tools (`read_file`, `write_file`, `edit_file`, `glob`, `ls`),
`run_julia`, `plot_julia`, and `execute` all share the workspace as their
working directory and resolve the same real paths: a workspace file is a
relative path (``model.jl``) or its absolute path, and a bare leading slash
(``/model.jl``) is the machine root, not the workspace. Writing a workspace
file with a leading slash, saving outside the workspace, or losing track of
where a file went is the recurring failure this suite measures.

Every sample is fixture-driven and needs no simulator environment, so the
suite is fast, hermetic, and does not chase upstream package changes. It runs
on any model, like the canary::

    uv run makker eval filesystem

Most capable models *can* do these. The point is doing them cleanly, in few
steps. Alongside the correctness checks, every task carries efficiency counters
(:func:`tool_call_count`, :func:`file_op_count`, :func:`julia_probe_count`):
re-reading, re-listing, and retrying writes is the visible cost of path
confusion, so a harness change that keeps the suite green while lowering the
file-op count is what "less confusing paths" looks like in numbers.
"""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import includes

from makker.eval.scorers import (
    file_op_count,
    julia_probe_count,
    no_interpreters_via_execute,
    no_repeated_identical_calls,
    no_unresolvable_path_in_julia,
    numeric_close,
    tool_call_count,
    used_any_tool,
    used_tools,
    workspace_file_exists,
)
from makker.eval.solver import load_eval_credentials, makker_solver

load_eval_credentials()

# A simulator with no environment instantiated still loads that simulator's
# prompt and skills, and the trivial Julia below (a sum, an include) runs on
# base Julia, so the path model is exercised per simulator at canary speed.
_OTHER_SIMULATORS = ("battmo", "fimbul", "mocca")

_BUGGY_SOLVE = """\
# Returns the wrong value on purpose; edit answer() to return 100.
function answer()
    return 6 * 7
end

println(answer())
"""

_RAW_NUMBERS = "12\n34\n56\n78\n"

_LIMITS = {"time_limit": 900, "token_limit": 400_000, "message_limit": 60}


def _efficiency() -> list:
    """Efficiency counters attached to every task (read with the correctness scorers)."""
    return [tool_call_count(), file_op_count(), julia_probe_count()]


@task
def filesystem() -> Task:
    """Write a workspace file, then load it in the REPL and report the value.

    The core handoff: a file created with the file tools must be loaded with a
    workspace-relative ``include`` (not a leading-slash ``/compute.jl``). The sum is
    deterministic (1+...+100 = 5050) and not mentally computable, so the value
    can only come from the REPL having run the file. Repeated once per
    simulator (no environment) to check the path guidance survives each skill
    bundle.
    """
    prompt = (
        "Create a file named `compute.jl` in the workspace containing Julia "
        "code that assigns `total = sum(1:100)`. Then load that file into the "
        "Julia session with `include` and reply with the value of `total`."
    )
    samples = [Sample(id="fs1-write-and-include", input=prompt, target="5050")]
    samples += [
        Sample(
            id=f"fs1-write-and-include-{sim}",
            input=prompt,
            target="5050",
            metadata={"simulator": sim},
        )
        for sim in _OTHER_SIMULATORS
    ]
    return Task(
        dataset=samples,
        solver=makker_solver(),
        scorer=[
            includes(),
            used_tools(["write_file", "run_julia"]),
            workspace_file_exists("compute.jl"),
            no_unresolvable_path_in_julia(),
            no_interpreters_via_execute(),
            *_efficiency(),
        ],
        **_LIMITS,
    )


@task
def filesystem_nested() -> Task:
    """A nested path must round-trip through both filesystems.

    The file is written with the file tools at ``scripts/stats.jl`` and loaded
    relative in the REPL. The value (30/5 = 6.0) confirms the include resolved
    the same file the write created.
    """
    sample = Sample(
        id="fs2-nested-write-and-include",
        input=(
            "Create the file `scripts/stats.jl` in the workspace (note the "
            "subdirectory) containing Julia code that assigns "
            "`m = sum([2, 4, 6, 8, 10]) / 5`. Load it into the Julia session "
            "with `include` and reply with the value of `m`."
        ),
    )
    return Task(
        dataset=[sample],
        solver=makker_solver(),
        scorer=[
            numeric_close(6.0, 0.01),
            used_tools(["write_file", "run_julia"]),
            workspace_file_exists("scripts/stats.jl"),
            no_unresolvable_path_in_julia(),
            no_interpreters_via_execute(),
            *_efficiency(),
        ],
        **_LIMITS,
    )


@task
def filesystem_edit() -> Task:
    """Edit an existing workspace file and re-run it.

    The fix requires an ``edit_file`` followed by a re-``include``; re-running
    the identical include without editing is a stuck loop, so
    ``no_repeated_identical_calls`` guards the trajectory while the value (100)
    confirms the edited file actually ran.
    """
    sample = Sample(
        id="fs3-edit-and-rerun",
        input=(
            "The workspace file `solve.jl` defines `answer()` and prints its "
            "result. It currently prints 42, but it should print 100. Edit the "
            "file so `answer()` returns 100, run it again with `include`, and "
            "report the value it prints."
        ),
        target="100",
        metadata={"fixtures": {"solve.jl": _BUGGY_SOLVE}},
    )
    return Task(
        dataset=[sample],
        solver=makker_solver(),
        scorer=[
            numeric_close(100.0, 0.5),
            used_tools(["edit_file", "run_julia"]),
            no_repeated_identical_calls(),
            no_unresolvable_path_in_julia(),
            no_interpreters_via_execute(),
            *_efficiency(),
        ],
        **_LIMITS,
    )


@task
def filesystem_save() -> Task:
    """Produce a real artifact in the workspace at a nested path, and report it.

    Tests "where do I save things": the file must land at
    ``results/cubes.txt`` in the workspace, and if it is written from Julia
    (``open("results/cubes.txt", ...)``) the path must be relative, not a
    leading-slash ``/results/cubes.txt``.
    """
    sample = Sample(
        id="fs4-save-output-file",
        input=(
            "Compute the cubes of 1 through 5 in the Julia session, then save "
            "them one per line to a file `results/cubes.txt` in the workspace. "
            "Reply with the largest cube."
        ),
        target="125",
    )
    return Task(
        dataset=[sample],
        solver=makker_solver(),
        scorer=[
            numeric_close(125.0, 0.5),
            workspace_file_exists("results/cubes.txt"),
            used_any_tool(["write_file", "run_julia"]),
            no_unresolvable_path_in_julia(),
            no_interpreters_via_execute(),
            *_efficiency(),
        ],
        **_LIMITS,
    )


@task
def filesystem_project() -> Task:
    """Build a small multi-file project and run it (nested relative include).

    Two files the agent writes itself, where ``main.jl`` includes
    ``src/geom.jl``: the include must resolve against the workspace, across
    files, so a path slip costs extra iterations. The printed value
    (round(pi*25, 4) = 78.5398) confirms the whole include chain ran.
    """
    sample = Sample(
        id="fs5-multi-file-project",
        input=(
            "Build a small Julia project in the workspace: a file `src/geom.jl` "
            "defining `circle_area(r) = pi * r^2`, and a file `main.jl` that "
            "includes `src/geom.jl` and prints `round(circle_area(5); "
            "digits=4)`. Run `main.jl` in the Julia session with `include` and "
            "report the printed value."
        ),
        target="78.5398",
    )
    return Task(
        dataset=[sample],
        solver=makker_solver(),
        scorer=[
            numeric_close(78.5398, 0.01),
            used_tools(["write_file", "run_julia"]),
            workspace_file_exists("src/geom.jl"),
            workspace_file_exists("main.jl"),
            no_unresolvable_path_in_julia(),
            no_repeated_identical_calls(),
            no_interpreters_via_execute(),
            *_efficiency(),
        ],
        **_LIMITS,
    )


@task
def filesystem_transform() -> Task:
    """Read a nested input file, compute, and write a nested output file.

    Reading and writing files is where the path confusion bites hardest. The
    integers sum to 180, and the output must land at ``data/total.txt`` in the
    workspace (not a leading-slash ``/data/total.txt``), and the read must
    resolve the fixture at ``data/raw.txt``.
    """
    sample = Sample(
        id="fs6-read-transform-write",
        input=(
            "The workspace file `data/raw.txt` has one integer per line. Read "
            "it, sum the integers, write the total to `data/total.txt` in the "
            "workspace, and report the total."
        ),
        target="180",
        metadata={"fixtures": {"data/raw.txt": _RAW_NUMBERS}},
    )
    return Task(
        dataset=[sample],
        solver=makker_solver(),
        scorer=[
            numeric_close(180.0, 0.5),
            workspace_file_exists("data/total.txt"),
            used_any_tool(["read_file", "run_julia"]),
            no_unresolvable_path_in_julia(),
            no_interpreters_via_execute(),
            *_efficiency(),
        ],
        **_LIMITS,
    )


TASKS = [
    filesystem,
    filesystem_nested,
    filesystem_edit,
    filesystem_save,
    filesystem_project,
    filesystem_transform,
]
