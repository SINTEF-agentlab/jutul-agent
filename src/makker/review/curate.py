"""Fold a review's findings into the curated issue store.

An agent decides whether each new finding is the *same* underlying issue as one
already tracked (merge) or a new one (create) — matching on root cause, not
wording. The store bookkeeping (counts, sessions, severity, examples) is then
applied deterministically in :mod:`makker.review.issues`. This is the
"intelligently update the log" step: many raw findings collapse into a few
high-signal, growing issues.
"""

from __future__ import annotations

from makker.credentials import missing_credential
from makker.review.findings import Finding, ReviewReport
from makker.review.issues import (
    Issue,
    load_issues,
    merge_finding,
    new_issue,
    save_issues,
)
from makker.review.reviewer import _coerce_text, _extract_json

_SYSTEM = """\
You maintain a deduplicated list of recurring issues found while reviewing an AI \
agent that drives Julia scientific simulators. For each NEW finding, decide \
whether it is the SAME underlying issue as one already tracked (merge) or a new, \
distinct issue (create). Match on root cause, not wording — e.g. "permeability in \
millidarcy" and "permeability not converted to SI" are the same issue. Reply with \
STRICT JSON only:
{"decisions": [{"finding": <index>, "issue": "<existing-id>" or "new", \
"title": "<concise title, required when issue is new>"}]}
Include one decision per finding, in order."""


def _decisions_prompt(findings: list[Finding], issues: dict[str, Issue]) -> str:
    if issues:
        existing = "\n".join(f"  [{i.id}] ({i.category}) {i.title}" for i in issues.values())
    else:
        existing = "  (none yet)"
    new = "\n".join(
        f"  [{n}] ({f.category}) {f.title} — evidence: {f.evidence[:200]}"
        for n, f in enumerate(findings)
    )
    return f"EXISTING ISSUES:\n{existing}\n\nNEW FINDINGS:\n{new}"


def _fallback_decisions(findings: list[Finding], issues: dict[str, Issue]) -> list[dict]:
    """Deterministic match by exact title (case-insensitive); used if the LLM fails."""
    by_title = {i.title.strip().lower(): i.id for i in issues.values()}
    out = []
    for f in findings:
        out.append({"issue": by_title.get(f.title.strip().lower()), "title": f.title})
    return out


async def _match_findings(
    findings: list[Finding], issues: dict[str, Issue], *, model_id: str
) -> list[dict]:
    """Per-finding decisions aligned with ``findings``: ``{issue: id|None, title}``.

    The matcher is an LLM call, but the offline (coding-agent) path promises no API
    cost — so when the reviewer model's provider key is absent, fall back to
    deterministic exact-title matching instead of erroring. Same-root-cause findings
    that share a title still merge; only fuzzy, reworded duplicates are missed, and a
    later ``review curate`` with a key re-clusters them.
    """
    if not findings:
        return []
    if not issues:
        return [{"issue": None, "title": f.title} for f in findings]

    if missing_credential(model_id) is not None:
        return _fallback_decisions(findings, issues)

    try:
        from langchain.chat_models import init_chat_model
        from langchain_core.messages import HumanMessage, SystemMessage

        reply = await init_chat_model(model_id).ainvoke(
            [
                SystemMessage(content=_SYSTEM),
                HumanMessage(content=_decisions_prompt(findings, issues)),
            ]
        )
        data = _extract_json(_coerce_text(reply.content))
    except Exception:
        return _fallback_decisions(findings, issues)

    out: list[dict] = [{"issue": None, "title": f.title} for f in findings]
    for d in data.get("decisions", []):
        if not isinstance(d, dict):
            continue
        idx = d.get("finding")
        if not isinstance(idx, int) or not (0 <= idx < len(findings)):
            continue
        target = d.get("issue")
        out[idx] = {
            "issue": target if isinstance(target, str) and target in issues else None,
            "title": (d.get("title") or findings[idx].title),
        }
    return out


async def curate_report(report: ReviewReport, *, model_id: str) -> dict[str, Issue]:
    """Merge ``report``'s findings into the issue store and persist it."""
    issues = load_issues()
    if not report.findings:
        return issues

    decisions = await _match_findings(report.findings, issues, model_id=model_id)
    for finding, decision in zip(report.findings, decisions, strict=False):
        target = decision.get("issue")
        if target and target in issues:
            merge_finding(issues[target], finding, report)
        else:
            issue = new_issue(finding, report, title=decision.get("title"), existing=issues)
            issues[issue.id] = issue
    save_issues(issues)
    return issues


async def curate_log(*, model_id: str, rebuild: bool = False) -> dict[str, Issue]:
    """(Re)build the issue store from the raw findings log.

    With ``rebuild`` it starts from an empty store, so the whole history is
    re-clustered; otherwise it folds every logged report into the current store.
    """
    from makker.review.findings import load_reports

    if rebuild:
        save_issues({})
    for report in load_reports():
        await curate_report(report, model_id=model_id)
    return load_issues()
