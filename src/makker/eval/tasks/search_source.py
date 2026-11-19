"""Per-simulator search against real installed source on disk.

The :mod:`~makker.eval.tasks.search` suite measures retrieval on a fixed
synthetic corpus; this one measures it on the genuine article: each
simulator's installed package source, found at its real path (``pkgdir`` in the
REPL) and read with the file tools. It instantiates the simulator's Julia
environment (``needs_env``), so the first run on a cold depot is slow and this
suite is the opt-in realism layer rather than part of the fast default::

    uv run makker eval search_source

To stay robust against upstream churn, grading is structural and pinned to the
one fact that holds for every Julia package across every release: its source
tree contains a main module file ``src/<Package>.jl``. The agent must locate the
source (via ``pkgdir``) and report that path. Exact line numbers, example
filenames, and symbol locations are deliberately not graded; those move with
each release, and chasing them would make the suite flaky.
"""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import includes

from makker.eval.scorers import no_interpreters_via_execute, used_any_tool
from makker.eval.solver import load_eval_credentials, makker_solver
from makker.simulators import registry

load_eval_credentials()

_SIMULATORS = ("jutuldarcy", "battmo", "fimbul", "mocca")


@task
def search_source() -> Task:
    """Find a package's main module file in its installed source, per simulator."""
    samples = []
    for sim in _SIMULATORS:
        package = registry.get(sim).primary_package
        samples.append(
            Sample(
                id=f"src-{sim}-locate-module",
                input=(
                    f"The {package} package is installed in this environment. Find "
                    "its source on disk (its path is what `pkgdir` returns in "
                    "`run_julia`), then use the file-search tools to locate its main "
                    f"module file, the one named `{package}.jl` under `src/`, and "
                    "reply with the path to it."
                ),
                # The module file is named <Package>.jl on every release, so the
                # filename is a stable target in whatever real path the agent reports.
                target=f"{package}.jl",
                metadata={"needs_env": True, "simulator": sim},
            )
        )
    return Task(
        dataset=samples,
        solver=makker_solver(),
        scorer=[
            includes(),
            used_any_tool(["grep", "glob", "ls", "read_file"]),
            no_interpreters_via_execute(),
        ],
        time_limit=2400,
        token_limit=2_000_000,
        message_limit=60,
    )


TASKS = [search_source]
