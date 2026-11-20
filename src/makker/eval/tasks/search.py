"""Search suite: can the agent find things with the file-search tools?

Each sample drops a small, fixed Julia package (``pkg/MiniRes``, see
``_corpus``) into the workspace and asks a retrieval question: where is a
function defined, which example uses it, which files call it, how many files of
a kind exist, what value a constant takes, and a multi-hop chase across files.
The corpus is synthetic and hermetic, so the suite is fast, runs on any model
with no simulator environment, and gives the same answers on every machine::

    uv run makker eval search

Because the corpus and its ground truth are fixed, this suite is also the
instrument for the open question of whether plain ``grep``/``glob`` retrieval
is enough or whether something heavier (BM25, embeddings) is worth adding:
swap the retrieval mechanism, rerun, compare. Each task pairs an answer check
(did it find the right thing) with efficiency counters (:func:`tool_call_count`,
:func:`file_op_count`): a model that answers by reading every file is
distinguishable from one that searched, and a retrieval change that keeps the
answers right while lowering the file-op count is a measurable win.
"""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import includes

from makker.eval.scorers import (
    answer_cites,
    file_op_count,
    no_interpreters_via_execute,
    numeric_close,
    tool_call_count,
    used_any_tool,
)
from makker.eval.solver import load_eval_credentials, makker_solver
from makker.eval.tasks._corpus import GRAVITY, JL_FILE_COUNT, ROOT, corpus_fixtures

load_eval_credentials()

_OTHER_SIMULATORS = ("battmo", "fimbul", "mocca")
_LIMITS = {"time_limit": 900, "token_limit": 400_000, "message_limit": 60}


def _efficiency() -> list:
    """Efficiency counters attached to every task (read with the answer scorer)."""
    return [tool_call_count(), file_op_count()]


def _sample(sample_id: str, prompt: str, target: str = "", simulator: str | None = None) -> Sample:
    """A search sample carrying a fresh copy of the corpus as fixtures."""
    metadata: dict = {"fixtures": corpus_fixtures()}
    if simulator:
        metadata["simulator"] = simulator
    return Sample(id=sample_id, input=prompt, target=target, metadata=metadata)


_LOCATE_DEF = (
    f"A small Julia package is in the workspace under `{ROOT}`. Find which file "
    "defines the function `darcy_flux` and reply with just that file's name "
    "(for example, `foo.jl`)."
)


@task
def search() -> Task:
    """Locate a definition / an example by name (recursion into subdirectories).

    ``darcy_flux`` is defined two directories deep and ``solve_newton``'s only
    example is in a nested ``examples/advanced/`` folder, so a non-recursive
    search misses both. The headline locate-the-definition question is repeated
    per simulator (no environment) to confirm the search guidance holds under
    each skill bundle.
    """
    samples = [
        _sample("se1-locate-definition", _LOCATE_DEF, target="darcy.jl"),
        _sample(
            "se2-locate-example",
            f"A small Julia package is in the workspace under `{ROOT}`. Among the "
            f"example scripts under `{ROOT}/examples`, which one calls "
            "`solve_newton`? Reply with just that file's name.",
            target="sweep.jl",
        ),
    ]
    samples += [
        _sample(f"se1-locate-definition-{sim}", _LOCATE_DEF, target="darcy.jl", simulator=sim)
        for sim in _OTHER_SIMULATORS
    ]
    return Task(
        dataset=samples,
        solver=makker_solver(),
        scorer=[
            includes(),
            used_any_tool(["grep", "glob"]),
            no_interpreters_via_execute(),
            *_efficiency(),
        ],
        **_LIMITS,
    )


@task
def search_callers() -> Task:
    """Find the call sites of a symbol spread across the tree.

    ``darcy_flux`` is defined in ``src/physics/darcy.jl`` and called in
    ``wells.jl`` and ``newton.jl``. A correct answer names both callers, which
    needs a recursive grep across the subdirectories, not just the top of
    ``src``. The grade requires both caller filenames but does not forbid a
    mention of the definition file. A complete answer naturally points out where
    ``darcy_flux`` is defined while listing its callers, and banning that
    substring would penalize a correct, thorough answer.
    """
    sample = _sample(
        "se3-find-call-sites",
        f"A small Julia package is in the workspace under `{ROOT}`. The function "
        "`darcy_flux` is defined in one file and called in others. Name the "
        f"files under `{ROOT}/src` that *call* `darcy_flux` (not the one that "
        "defines it).",
    )
    return Task(
        dataset=[sample],
        solver=makker_solver(),
        scorer=[
            answer_cites(required=("wells.jl", "newton.jl")),
            used_any_tool(["grep"]),
            no_interpreters_via_execute(),
            *_efficiency(),
        ],
        **_LIMITS,
    )


@task
def search_multihop() -> Task:
    """A two-hop chase: example -> the function it calls -> what that calls.

    ``examples/advanced/sweep.jl`` calls ``solve_newton``, which (in
    ``src/solver/newton.jl``) internally calls ``darcy_flux``, defined under
    ``src/physics/``. Answering means following that call chain across files,
    not running a single grep. It is the kind of code navigation a real session
    does.
    """
    sample = _sample(
        "se6-call-chain",
        f"In the package under `{ROOT}`, the example "
        "`examples/advanced/sweep.jl` calls a function. That function "
        f"internally calls another function that is defined under `{ROOT}/src/"
        "physics/`. Name that physics function.",
        target="darcy_flux",
    )
    return Task(
        dataset=[sample],
        solver=makker_solver(),
        scorer=[
            includes(),
            used_any_tool(["grep", "read_file"]),
            no_interpreters_via_execute(),
            *_efficiency(),
        ],
        **_LIMITS,
    )


@task
def search_count() -> Task:
    """Count files of a kind across the whole tree (recursive glob).

    The right answer needs every ``.jl`` file, including the nested ones; a
    flat listing of the top directory under-counts.
    """
    sample = _sample(
        "se4-count-jl-files",
        f"Count the `.jl` files under `{ROOT}`, including those in "
        "subdirectories, and reply with just the number.",
    )
    return Task(
        dataset=[sample],
        solver=makker_solver(),
        scorer=[
            numeric_close(float(JL_FILE_COUNT), 0.5),
            used_any_tool(["glob", "grep", "ls"]),
            no_interpreters_via_execute(),
            *_efficiency(),
        ],
        **_LIMITS,
    )


@task
def search_constant() -> Task:
    """Retrieve a specific value from the source (grep to a number)."""
    sample = _sample(
        "se5-find-constant",
        f"In the `{ROOT}` source, find the line that sets the `GRAVITY` constant "
        "and reply with its numeric value.",
    )
    return Task(
        dataset=[sample],
        solver=makker_solver(),
        scorer=[
            numeric_close(GRAVITY, 0.001),
            used_any_tool(["grep", "read_file"]),
            no_interpreters_via_execute(),
            *_efficiency(),
        ],
        **_LIMITS,
    )


TASKS = [search, search_callers, search_multihop, search_count, search_constant]
