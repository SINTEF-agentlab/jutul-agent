"""Link an eval run's scores back onto the sessions it produced.

An eval session already carries the expected answer (recorded by the solver); the
pass/fail verdict, though, is computed by the scorers afterwards and lives in the
inspect log, not the session. This closes that gap: after an eval run, write each
sample's verdict onto its session trace as an ``eval_result`` event, so a later
review can say "the agent's answer looked fine but the run scored FAIL" and the
dashboard can badge it.
"""

from __future__ import annotations

from typing import Any

# The store key the eval solver stamps on each sample. Mirrored here as a literal
# so importing this module (e.g. for `jutul-agent review`) does not pull in the
# eval stack, which needs the optional ``[eval]`` extra. Kept in sync with
# ``jutul_agent.eval.solver.STORE_SESSION_ID``.
STORE_SESSION_ID = "jutul/session_id"


def _passed(scores: dict | None) -> bool:
    """True when every *grading* scorer marked the sample correct.

    Only pass/fail scorers (value ``CORRECT``/``INCORRECT``/``PARTIAL``) count toward
    the verdict. Diagnostic metric scorers — e.g. ``tool_call_count`` — report a
    number, not a grade, so they are ignored; otherwise any task that carries
    efficiency metrics would read as FAIL even when its answer is correct.
    """
    from inspect_ai.scorer import CORRECT, INCORRECT, PARTIAL

    values = [getattr(s, "value", s) for s in (scores or {}).values()]
    grades = [v for v in values if v in (CORRECT, INCORRECT, PARTIAL)]
    return bool(grades) and all(v == CORRECT for v in grades)


def _samples(log: Any) -> list:
    samples = getattr(log, "samples", None)
    if samples:
        return samples
    location = getattr(log, "location", None)
    if not location:
        return []
    try:
        from inspect_ai.log import read_eval_log

        return read_eval_log(location).samples or []
    except Exception:
        return []


def link_eval_results(logs: Any) -> int:
    """Record each scored sample's verdict on its session trace. Returns the count."""
    from jutul_agent.review.discovery import find_session
    from jutul_agent.trace import TraceLog
    from jutul_agent.trace.schema import EVAL_RESULT

    linked = 0
    for log in logs:
        task = getattr(getattr(log, "eval", None), "task", "") or ""
        for sample in _samples(log):
            sid = (getattr(sample, "store", None) or {}).get(STORE_SESSION_ID)
            if not sid:
                continue
            session = find_session(sid)
            if session is None:
                continue
            scores = getattr(sample, "scores", None)
            with TraceLog(session.trace_path) as session_log:
                session_log.append(
                    EVAL_RESULT,
                    {
                        "passed": _passed(scores),
                        "task": task,
                        "scores": {
                            k: str(getattr(v, "value", v)) for k, v in (scores or {}).items()
                        },
                    },
                )
            linked += 1
    return linked
