"""Run the critic over a session and turn its reply into a stored report.

``review_transcript`` is the core: it sends the rendered session to the reviewer
model and parses the JSON it returns. ``review_session`` wraps it for the live
agent (render the session's own trace, then store the report), and is best-effort
so a review never disturbs the run it is reviewing.
"""

from __future__ import annotations

import json
import re
from typing import Any

from makker.review.findings import Finding, ReviewReport, append_report, now_iso
from makker.review.prompt import SYSTEM, build_user_message

# Keep the transcript we send bounded, since a runaway solve can print megabytes,
# and the tail (results, the agent's conclusion) matters more than a flood of progress.
_TRANSCRIPT_CAP = 120_000


def _coerce_text(content: Any) -> str:
    """Flatten a chat message's content (str, or a list of parts) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return str(content)


def _extract_json(text: str) -> dict:
    """Pull the report object out of a model reply that may wrap it in prose/fences."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in reviewer response")
    return json.loads(text[start : end + 1])


def _clip(transcript: str) -> str:
    if len(transcript) <= _TRANSCRIPT_CAP:
        return transcript
    head = _TRANSCRIPT_CAP // 4
    tail = _TRANSCRIPT_CAP - head
    return f"{transcript[:head]}\n\n…(transcript trimmed)…\n\n{transcript[-tail:]}"


async def review_transcript(
    transcript: str,
    *,
    session_id: str,
    title: str,
    model_id: str,
    simulator: str | None = None,
    ground_truth: str | None = None,
) -> ReviewReport:
    """Send a rendered session to the reviewer model and parse its findings."""

    from langchain.chat_models import init_chat_model
    from langchain_core.messages import HumanMessage, SystemMessage

    model = init_chat_model(model_id)
    user = build_user_message(_clip(transcript), simulator=simulator, ground_truth=ground_truth)
    reply = await model.ainvoke([SystemMessage(content=SYSTEM), HumanMessage(content=user)])
    data = _extract_json(_coerce_text(reply.content))
    findings = [Finding.from_dict(f) for f in (data.get("findings") or []) if isinstance(f, dict)]
    return ReviewReport(
        session_id=session_id,
        title=title,
        model=model_id,
        created_at=now_iso(),
        summary=str(data.get("summary", "")),
        findings=findings,
        simulator=simulator or "",
        app_version=app_version(),
    )


def app_version() -> str:
    """The makker version present when this review ran (for staleness)."""
    try:
        from makker import __version__

        return str(__version__)
    except Exception:
        return ""


def render_session(session: Any) -> str:
    """Markdown of the live session's whole trace, the rendering the CLI produces."""
    from makker.transcript import render_markdown

    return render_markdown(session.trace.iter_events())


def session_simulator(session: Any) -> str | None:
    return getattr(getattr(session, "simulator", None), "name", None)


async def review_session(
    session: Any,
    *,
    model_id: str,
    store: bool = True,
    curate: bool = True,
) -> ReviewReport | None:
    """Review the whole live ``session`` once and store the result.

    Appends the raw report to the findings log and, when ``curate``, folds its
    findings into the curated issue store. Best-effort: any failure (no API key, a
    model error, an empty session) is swallowed and returns ``None`` so reviewing
    never breaks the run.
    """

    try:
        transcript = render_session(session)
        if not transcript.strip():
            return None
        report = await review_transcript(
            transcript,
            session_id=session.session_id,
            title=getattr(session, "title", "") or "",
            model_id=model_id,
            simulator=session_simulator(session),
        )
        return await store_report(report, store=store, curate=curate, model_id=model_id)
    except Exception:
        return None


async def store_report(
    report: ReviewReport, *, store: bool = True, curate: bool = True, model_id: str
) -> ReviewReport:
    """Append a report to the log and fold it into the curated issues."""
    if store:
        append_report(report)
    if curate and report.findings:
        from makker.review.curate import curate_report

        await curate_report(report, model_id=model_id)
    return report


async def maybe_review_session(session: Any) -> ReviewReport | None:
    """Review the session iff review mode is enabled (the end-of-session hook)."""
    from makker.review.settings import review_enabled, review_model

    if not review_enabled():
        return None
    return await review_session(session, model_id=review_model())


async def ingest_findings(
    data: dict,
    *,
    session_id: str,
    title: str = "",
    model_id: str,
    source: str = "external",
    simulator: str | None = None,
    store: bool = True,
    curate: bool = True,
) -> ReviewReport:
    """Build a report from findings produced *outside* the API path and store it.

    This is the coding-agent path: a tool like Claude Code reads the transcript
    (see ``makker review prompt``), produces the same ``{summary, findings}``
    JSON the critic would, and feeds it back here, so the expensive read costs no
    API. ``source`` records who produced it; curation still runs (cheaply) unless
    disabled.
    """
    from makker.review.findings import Finding, now_iso

    findings = [Finding.from_dict(f) for f in (data.get("findings") or []) if isinstance(f, dict)]
    report = ReviewReport(
        session_id=session_id,
        title=title,
        model=source,
        created_at=now_iso(),
        summary=str(data.get("summary", "")),
        findings=findings,
        simulator=simulator or "",
        app_version=app_version(),
    )
    return await store_report(report, store=store, curate=curate, model_id=model_id)
