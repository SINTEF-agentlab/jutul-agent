"""Developer-facing session review: an autonomous critic that flags what the agent
(or its in-Julia validation) missed, and logs it for the developer to act on.

See ``docs/review.md``. Off unless ``MAKKER_REVIEW`` is set.
"""

from __future__ import annotations

from makker.review.curate import curate_log, curate_report
from makker.review.discovery import DiscoveredSession, discover_sessions, find_session
from makker.review.findings import Finding, ReviewReport, load_reports, review_log_path
from makker.review.issues import Issue, load_issues, set_status
from makker.review.reviewer import (
    ingest_findings,
    maybe_review_session,
    review_session,
    review_transcript,
)
from makker.review.settings import review_enabled, review_model

__all__ = [
    "DiscoveredSession",
    "Finding",
    "Issue",
    "ReviewReport",
    "curate_log",
    "curate_report",
    "discover_sessions",
    "find_session",
    "ingest_findings",
    "load_issues",
    "load_reports",
    "maybe_review_session",
    "review_enabled",
    "review_log_path",
    "review_model",
    "review_session",
    "review_transcript",
    "set_status",
]
