"""The curated issue store: recurring findings merged into distinct issues.

The findings log (``findings.jsonl``) is the raw, append-only record — one entry
per review, duplicates and all. This store sits on top of it: an evolving set of
*distinct* issues, each accumulating how often it has been seen, in which sessions,
and a few representative examples. New findings are folded in by
:mod:`makker.review.curate` (an agent decides what is the same issue); the
bookkeeping here — counts, severity, examples — is deterministic.

It is the developer's high-signal view: "this unit-conversion gap has shown up 7
times across 5 sessions" instead of 7 separate log lines.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from makker.review.findings import Finding, ReviewReport, now_iso, review_dir

_SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2}
_STATUSES = ("open", "fixed", "dismissed")
_MAX_EXAMPLES = 5


@dataclass
class Issue:
    """A distinct, recurring problem, accumulated across reviews.

    ``last_version`` is the makker version of the most recent review that
    touched this issue. Compared against the current version it gives a staleness
    hint: an open issue last seen on an older version, not re-flagged since, has
    likely already been fixed.
    """

    id: str
    title: str
    category: str
    fix_target: str
    severity: str
    status: str
    count: int
    first_seen: str
    last_seen: str
    sessions: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    last_version: str = ""

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, data: dict) -> Issue:
        return cls(
            id=str(data.get("id", "")),
            title=str(data.get("title", "")),
            category=str(data.get("category", "other")),
            fix_target=str(data.get("fix_target", "other")),
            severity=str(data.get("severity", "medium")),
            status=str(data.get("status", "open")),
            count=int(data.get("count", 0)),
            first_seen=str(data.get("first_seen", "")),
            last_seen=str(data.get("last_seen", "")),
            sessions=list(data.get("sessions", [])),
            examples=list(data.get("examples", [])),
            last_version=str(data.get("last_version", "")),
        )


def issues_path():
    return review_dir() / "issues.json"


def load_issues() -> dict[str, Issue]:
    """The curated issues keyed by id; empty if none have been recorded."""
    path = issues_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {k: Issue.from_dict(v) for k, v in data.items() if isinstance(v, dict)}


def save_issues(issues: dict[str, Issue]) -> None:
    path = issues_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v.to_dict() for k, v in issues.items()}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _slug(title: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return base[:48] or "issue"


def unique_id(title: str, existing: dict[str, Issue]) -> str:
    base = _slug(title)
    if base not in existing:
        return base
    n = 2
    while f"{base}-{n}" in existing:
        n += 1
    return f"{base}-{n}"


def _max_severity(a: str, b: str) -> str:
    return a if _SEVERITY_RANK.get(a, 1) >= _SEVERITY_RANK.get(b, 1) else b


def new_issue(
    finding: Finding, report: ReviewReport, *, title: str | None, existing: dict[str, Issue]
) -> Issue:
    """Create a fresh issue from a finding."""
    use_title = (title or finding.title).strip() or finding.category
    return Issue(
        id=unique_id(use_title, existing),
        title=use_title,
        category=finding.category,
        fix_target=finding.fix_target,
        severity=finding.severity,
        status="open",
        count=1,
        first_seen=report.created_at or now_iso(),
        last_seen=report.created_at or now_iso(),
        sessions=[report.session_id] if report.session_id else [],
        examples=[finding.evidence] if finding.evidence else [],
        last_version=report.app_version,
    )


def merge_finding(issue: Issue, finding: Finding, report: ReviewReport) -> None:
    """Fold a finding into an existing issue (deterministic bookkeeping)."""
    issue.count += 1
    issue.last_seen = report.created_at or now_iso()
    issue.last_version = report.app_version or issue.last_version
    issue.severity = _max_severity(issue.severity, finding.severity)
    if report.session_id and report.session_id not in issue.sessions:
        issue.sessions.append(report.session_id)
    if finding.evidence and finding.evidence not in issue.examples:
        issue.examples = [*issue.examples, finding.evidence][-_MAX_EXAMPLES:]
    # A re-opened issue that recurs is worth surfacing again.
    if issue.status == "fixed":
        issue.status = "open"


def set_status(issue_id: str, status: str) -> bool:
    """Mark an issue fixed/dismissed/open. Returns False if the id is unknown."""
    if status not in _STATUSES:
        raise ValueError(f"status must be one of {_STATUSES}")
    issues = load_issues()
    if issue_id not in issues:
        return False
    issues[issue_id].status = status
    save_issues(issues)
    return True


def delete_issue(issue_id: str) -> bool:
    """Remove an issue from the store entirely. Returns False if the id is unknown.

    Unlike ``dismiss`` (which keeps the record, hidden), delete is for issues that
    no longer make sense to track at all — e.g. an obsolete one after a refactor.
    The raw findings log is untouched, so a later ``curate --rebuild`` can recreate
    it if the underlying findings still warrant it.
    """
    issues = load_issues()
    if issue_id not in issues:
        return False
    del issues[issue_id]
    save_issues(issues)
    return True


def is_stale(issue: Issue, current_version: str) -> bool:
    """True when an open issue was last seen on a different (older) app version.

    A cheap "probably fixed" hint: we keep fixing things as they surface, so an
    open issue that hasn't recurred since the version changed is a candidate to
    resolve. Only meaningful once reports carry a version, so a blank version
    (older logs) is never called stale.
    """
    return (
        issue.status == "open"
        and bool(issue.last_version)
        and bool(current_version)
        and issue.last_version != current_version
    )
