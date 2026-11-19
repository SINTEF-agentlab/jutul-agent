"""The eval harness: run makker under Inspect AI and grade the trace.

This package is the harness: the solver, the scorers, and RunConfig.
makker-bench is the benchmark built on it; its public suites live in
:mod:`makker.eval.tasks` and run via ``makker eval``::

    uv run makker eval canary

The agent under test runs unchanged (its own tools, skills, prompt, and
trace) inside Inspect's agent bridge, which routes every model call to the
eval's ``--model``. Scorers grade the final answer plus the session trace
and workspace, so a pass means the agent did the work, not just said so.

Requires the ``eval`` extra: ``uv sync --extra eval``.
"""

from __future__ import annotations

try:
    import inspect_ai  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError("makker.eval requires the 'eval' extra (uv sync --extra eval).") from exc
