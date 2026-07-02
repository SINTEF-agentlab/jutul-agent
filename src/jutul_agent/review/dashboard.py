"""Assemble the review data into the dashboard page model and render the page.

The page is plain HTML/CSS/JS with the data embedded as JSON. The live dashboard
(served by :mod:`jutul_agent.review.server`) renders transcripts on demand; the
export (:mod:`jutul_agent.review.export`) embeds them inline for a single shareable
file. Both share the one page shell here.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from jutul_agent.review.discovery import (
    discover_sessions,
    session_eval_result,
    session_simulator,
)
from jutul_agent.review.findings import load_reports
from jutul_agent.review.issues import is_stale, load_issues

_SEV_WEIGHT = {"high": 3, "medium": 2, "low": 1}


def _days_since(iso: str, now: datetime) -> float:
    try:
        seen = datetime.fromisoformat(iso)
    except ValueError:
        return 365.0
    if seen.tzinfo is None:
        seen = seen.replace(tzinfo=UTC)
    return max(0.0, (now - seen).total_seconds() / 86400.0)


def _real_started(trace_path) -> str:
    """The session's first-event timestamp (``''`` if unreadable) — more reliable than
    ``DiscoveredSession.started``, which falls back to the directory mtime."""
    import sqlite3

    from jutul_agent.trace import TraceLog

    try:
        with TraceLog.open_readonly(trace_path) as log:
            return log.first_timestamp() or ""
    except sqlite3.Error:
        return ""


def _priority(severity: str, count: int, last_seen: str, now: datetime) -> float:
    """Rank an issue by severity, how often it recurs, and how recently.

    Severity weight times recurrence, decayed by age so a stale one-off sinks below
    a fresh recurring problem.
    """
    recency = 1.0 / (1.0 + _days_since(last_seen, now) / 14.0)
    return round(_SEV_WEIGHT.get(severity, 2) * count * recency, 2)


def build_data() -> dict:
    """The JSON-serialisable page model: stats, issues, and per-session reviews."""
    from jutul_agent.review.reviewer import app_version

    now = datetime.now(UTC)
    current_version = app_version()
    issues = list(load_issues().values())
    reports = load_reports()
    by_id = {s.session_id: s for s in discover_sessions()}

    def report_simulator(r) -> str:
        if r.simulator:
            return r.simulator
        s = by_id.get(r.session_id)
        return (s and session_simulator(s.trace_path)) or ""

    def report_eval(r) -> str:
        # PASS / FAIL if this session was an eval run whose verdict was linked back.
        s = by_id.get(r.session_id)
        result = session_eval_result(s.trace_path) if s else None
        if result is None:
            return ""
        return "PASS" if result.get("passed") else "FAIL"

    # Real session-start dates, read once per referenced session.
    referenced = {sid for i in issues for sid in i.sessions} | {
        r.session_id for r in reports if r.session_id
    }
    started_by_id: dict[str, str] = {}
    for sid in referenced:
        s = by_id.get(sid)
        if s is None:
            continue
        started_by_id[sid] = _real_started(s.trace_path) or (
            s.started.isoformat() if s.started else ""
        )

    def session_meta(sid: str) -> dict:
        s = by_id.get(sid)
        return {
            "id": sid,
            "title": (s.title if s and s.title else ""),
            "started": started_by_id.get(sid, ""),
            "has_transcript": s is not None,
        }

    issue_rows = []
    for i in issues:
        sess = [session_meta(sid) for sid in i.sessions]
        sdates = sorted(m["started"] for m in sess if m["started"])
        last_session = sdates[-1] if sdates else ""
        issue_rows.append(
            {
                "id": i.id,
                "title": i.title,
                "category": i.category,
                "fix_target": i.fix_target,
                "severity": i.severity,
                "status": i.status,
                "count": i.count,
                "first_seen": i.first_seen,
                "last_seen": i.last_seen,
                # Date range of the underlying sessions, not the review timestamps.
                "first_session": sdates[0] if sdates else "",
                "last_session": last_session,
                "last_version": i.last_version,
                "stale": is_stale(i, current_version),
                # Recency decay keys off the latest session date.
                "priority": _priority(i.severity, i.count, last_session or i.last_seen, now),
                "sessions": sess,
                "examples": i.examples,
            }
        )
    issue_rows.sort(key=lambda r: r["priority"], reverse=True)

    review_rows = [
        {
            "session": session_meta(r.session_id),
            "title": r.title,
            "model": r.model,
            "created_at": r.created_at,
            "simulator": report_simulator(r),
            "eval": report_eval(r),
            "app_version": r.app_version,
            "summary": r.summary,
            "findings": [f.to_dict() for f in r.findings],
        }
        for r in reversed(reports)  # newest first
    ]

    by_sim: dict[str, int] = {}
    for r in review_rows:
        if r["simulator"]:
            by_sim[r["simulator"]] = by_sim.get(r["simulator"], 0) + len(r["findings"])

    open_issues = [r for r in issue_rows if r["status"] == "open"]
    reviewed = {r.session_id for r in reports if r.session_id}
    stats = {
        "issues_total": len(issue_rows),
        "issues_open": len(open_issues),
        "issues_stale": sum(1 for r in open_issues if r["stale"]),
        "current_version": current_version,
        "findings_total": sum(len(r["findings"]) for r in review_rows),
        "reviews_total": len(review_rows),
        "sessions_total": len(by_id),
        "sessions_reviewed": len(reviewed & set(by_id)),
        "by_severity": _tally(open_issues, "severity", ("high", "medium", "low")),
        "by_fix_target": _tally(
            open_issues,
            "fix_target",
            ("case-validation", "skill", "prompt", "eval", "code", "other"),
        ),
        "by_category": _tally(
            open_issues,
            "category",
            (
                "validation-gap",
                "agent-error",
                "plausible-but-wrong",
                "silent-failure",
                "tooling-gap",
                "other",
            ),
        ),
        "by_simulator": [
            {"name": k, "count": v} for k, v in sorted(by_sim.items(), key=lambda kv: -kv[1])
        ],
    }
    return {"stats": stats, "issues": issue_rows, "reviews": review_rows}


def _tally(rows: list[dict], key: str, order: tuple[str, ...]) -> list[dict]:
    counts: dict[str, int] = {}
    for r in rows:
        counts[r[key]] = counts.get(r[key], 0) + 1
    known = [{"name": name, "count": counts.pop(name, 0)} for name in order]
    extra = [{"name": k, "count": v} for k, v in sorted(counts.items())]
    return [row for row in known + extra if row["count"]]


def _template() -> str:
    return (Path(__file__).with_name("dashboard_template.html")).read_text(encoding="utf-8")


def _embed(value: object) -> str:
    """JSON for embedding in a <script> tag (guard against a literal closing tag)."""
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")


def render_page(data: dict | None = None, transcripts: dict[str, str] | None = None) -> str:
    """The full dashboard HTML. With ``transcripts`` it is a single shareable file."""
    data = build_data() if data is None else data
    return (
        _template()
        .replace("__DATA__", _embed(data))
        .replace("__TRANSCRIPTS__", _embed(transcripts or {}))
    )
