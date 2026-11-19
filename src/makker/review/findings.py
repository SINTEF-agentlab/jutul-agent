"""Findings a session review produces, and the developer-only log they append to.

A :class:`ReviewReport` is one critic pass over one session: a short summary plus
a list of :class:`Finding`. Reports are appended as JSON lines to a single log
under the state home so the developer can browse what the agent (or its in-Julia
validation) missed across many runs — this is a developer tool, never shown to the
end user.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from makker.paths import state_home

# Open vocabularies — the critic is asked to use these, but parsing tolerates
# anything so an off-list value is kept rather than dropped.
CATEGORIES = (
    "validation-gap",  # a bad input/result the agent or its validation let through
    "agent-error",  # wrong API, gave up, misread output, hallucinated
    "plausible-but-wrong",  # ran fine but the result is physically or numerically off
    "silent-failure",  # no error surfaced but something is broken
    "tooling-gap",  # a missing tool/skill/check would have prevented this
    "other",
)
SEVERITIES = ("low", "medium", "high")
FIX_TARGETS = (
    "case-validation",  # extend the active simulator's input/case validation
    "skill",  # clarify/extend a skill
    "prompt",  # adjust the system prompt
    "eval",  # add a regression case to the eval suite
    "code",  # a makker code change
    "other",
)


@dataclass(frozen=True)
class Finding:
    """One thing the reviewer flagged about a session.

    ``category`` and ``fix_target`` use the open vocabularies above as *guidance*:
    parsing keeps any off-list value the reviewer chooses rather than dropping it,
    so the labels never force a finding into a box that doesn't fit. ``detail`` is
    free-form room for nuance that ``evidence``/``suggestion`` don't capture (why it
    matters, what the agent did well, broader context).
    """

    category: str
    severity: str
    title: str
    evidence: str
    suggestion: str
    fix_target: str
    detail: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> Finding:
        """Build from a critic-produced dict, tolerating missing/extra keys."""

        def s(key: str) -> str:
            value = data.get(key, "")
            return value.strip() if isinstance(value, str) else str(value)

        return cls(
            category=s("category") or "other",
            severity=s("severity") or "medium",
            title=s("title"),
            evidence=s("evidence"),
            suggestion=s("suggestion"),
            fix_target=s("fix_target") or "other",
            detail=s("detail"),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "category": self.category,
            "severity": self.severity,
            "title": self.title,
            "evidence": self.evidence,
            "suggestion": self.suggestion,
            "fix_target": self.fix_target,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class ReviewReport:
    """One critic pass over one session.

    ``simulator`` and ``app_version`` (the makker version present when the
    review ran) are recorded so findings can be aged out: an issue last seen on an
    old version, with no recurrence since, is probably already fixed.
    """

    session_id: str
    title: str
    model: str
    created_at: str
    summary: str
    findings: list[Finding]
    simulator: str = ""
    app_version: str = ""

    @property
    def ok(self) -> bool:
        """True when the reviewer flagged nothing."""
        return not self.findings

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "title": self.title,
            "model": self.model,
            "created_at": self.created_at,
            "summary": self.summary,
            "simulator": self.simulator,
            "app_version": self.app_version,
            "findings": [f.to_dict() for f in self.findings],
        }

    @classmethod
    def from_dict(cls, data: dict) -> ReviewReport:
        raw = data.get("findings") or []
        findings = [Finding.from_dict(f) for f in raw if isinstance(f, dict)]
        return cls(
            session_id=str(data.get("session_id", "")),
            title=str(data.get("title", "")),
            model=str(data.get("model", "")),
            created_at=str(data.get("created_at", "")),
            summary=str(data.get("summary", "")),
            findings=findings,
            simulator=str(data.get("simulator", "")),
            app_version=str(data.get("app_version", "")),
        )


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def review_dir() -> Path:
    return state_home() / "review"


def review_log_path() -> Path:
    """The append-only JSONL log of every review (one report per line)."""
    return review_dir() / "findings.jsonl"


def append_report(report: ReviewReport) -> Path:
    """Append ``report`` to the review log (created on first use)."""
    path = review_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(report.to_dict(), ensure_ascii=False) + "\n")
    return path


def load_reports() -> list[ReviewReport]:
    """Every review in the log, oldest first; malformed lines are skipped."""
    path = review_log_path()
    if not path.exists():
        return []
    reports: list[ReviewReport] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            reports.append(ReviewReport.from_dict(json.loads(line)))
        except (json.JSONDecodeError, AttributeError, TypeError):
            continue
    return reports
