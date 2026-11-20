"""A fixed synthetic source tree for the search suite: the `MiniRes` package.

`MiniRes` is a tiny, made-up reservoir-simulation package kept as real files
under ``MiniRes/`` next to this module. The search tasks copy it into the
workspace as fixtures and ask retrieval questions about it. A fixed synthetic
package (rather than real installed source) keeps the tasks fast, hermetic, and
reproducible: the corpus is identical on every machine and every run, its
ground truth is known, and it never shifts under an upstream release. That is
what makes it a fair instrument for comparing retrieval strategies, plain
``grep``/``glob`` today and BM25 or embeddings later.

The layout exercises what trips a flat search up. A definition is buried two
directories deep (``src/physics/darcy.jl``), one symbol is defined in one file
and called in two others, and a nested example (``examples/advanced/sweep.jl``)
is reachable only by a recursive glob.

The tasks and tests rely on this ground truth:

- ``darcy_flux`` is defined only in ``src/physics/darcy.jl``.
- ``darcy_flux`` is called in ``src/physics/wells.jl`` and
  ``src/solver/newton.jl``, and nowhere else.
- ``solve_newton`` is defined in ``src/solver/newton.jl``, and the only example
  that calls it is ``examples/advanced/sweep.jl``.
- ``GRAVITY = 9.81`` is set once, in ``src/physics/darcy.jl``.
- The tree holds ``JL_FILE_COUNT`` ``.jl`` files.
"""

from __future__ import annotations

from pathlib import Path

# Where the corpus lands in the eval workspace, and where its real files live.
ROOT = "pkg/MiniRes"
_SOURCE = Path(__file__).parent / "MiniRes"

# The value the GRAVITY search task expects to find in the source.
GRAVITY = 9.81


def corpus_fixtures() -> dict[str, str]:
    """The corpus as a fresh ``{workspace-relative path: content}`` mapping.

    Reads the real ``MiniRes/`` files, so the fixtures written into the eval
    workspace can never drift from the package checked into the repo.
    """
    return {
        f"{ROOT}/{path.relative_to(_SOURCE).as_posix()}": path.read_text(encoding="utf-8")
        for path in sorted(_SOURCE.rglob("*"))
        if path.is_file()
    }


# Derived from the real tree so the count stays in step with the files.
JL_FILE_COUNT = sum(1 for path in _SOURCE.rglob("*.jl"))
