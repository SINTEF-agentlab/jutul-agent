"""Per-simulator filesystem round-trip against the real installed env.

The :mod:`~jutul_agent.eval.tasks.filesystem` suite is hermetic (no simulator
environment); this is its realism counterpart, the way
:mod:`~jutul_agent.eval.tasks.search_source` is to
:mod:`~jutul_agent.eval.tasks.search`. Each sample instantiates the simulator's
Julia environment (``needs_env``), so the file tools, the read-only depot
guard, and the loaded simulator stack are all present, and checks that the path
model still holds with them in play: inspect the installed package source at its
real path, then write a workspace file by a relative path. The first
run on a cold depot is slow, so this is the opt-in realism layer rather than
part of the fast default::

    uv run jutul-agent eval filesystem_source

Grading is structural and robust to upstream churn: the workspace output file
must exist and the answer must name the package whose source path was found
(``pkgdir`` reports it for whatever version is installed). Exact paths and
filenames inside the depot are deliberately not pinned, since they move with
each release and the Julia depot hashes them.
"""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import includes

from jutul_agent.eval.scorers import (
    no_interpreters_via_execute,
    no_unresolvable_path_in_julia,
    used_any_tool,
    workspace_file_exists,
)
from jutul_agent.eval.solver import jutul_agent_solver, load_eval_credentials
from jutul_agent.simulators import registry

load_eval_credentials()

_SIMULATORS = ("jutuldarcy", "battmo", "fimbul", "mocca")


@task
def filesystem_source() -> Task:
    """Read installed source at its real path, then write a workspace file, per simulator."""
    samples = []
    for sim in _SIMULATORS:
        package = registry.get(sim).primary_package
        samples.append(
            Sample(
                id=f"fss-{sim}-source-to-workspace",
                input=(
                    f"The {package} package is installed in this environment. Find its "
                    "source directory using the path supplied in the session context, then "
                    f"write that path to a new workspace file `notes/{package}_source.txt`. "
                    "Reply with the path you wrote."
                ),
                # The reported source path always contains the package name, so a
                # correct round-trip names it. The path itself is not pinned.
                target=package,
                metadata={"needs_env": True, "simulator": sim},
            )
        )
    return Task(
        dataset=samples,
        solver=jutul_agent_solver(),
        scorer=[
            includes(),
            workspace_file_exists("notes/*_source.txt"),
            used_any_tool(["read_file", "grep", "glob", "ls"]),
            no_unresolvable_path_in_julia(),
            no_interpreters_via_execute(),
        ],
        time_limit=2400,
        token_limit=2_000_000,
        message_limit=60,
    )


TASKS = [filesystem_source]
