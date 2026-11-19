"""Export the review for offline, private sharing.

Two formats, neither of which needs a server or touches anything public:

- ``html``: one self-contained file with every reviewed session's transcript
  embedded, so a single attachment carries the whole picture.
- ``md``: a plain-text digest of the open issues, ranked, for pasting into a doc
  or message.
"""

from __future__ import annotations

from pathlib import Path

from makker.review.dashboard import build_data, render_page
from makker.review.discovery import find_session, render_trace_html
from makker.review.findings import review_dir


def _reviewed_session_ids(data: dict) -> list[str]:
    seen: dict[str, None] = {}
    for review in data["reviews"]:
        sid = review["session"]["id"]
        if review["session"]["has_transcript"]:
            seen.setdefault(sid, None)
    return list(seen)


def export_html(out_path: Path | None = None) -> Path:
    """Write a single self-contained HTML file with transcripts embedded."""
    out = out_path or (review_dir() / "review-dashboard.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    data = build_data()
    transcripts: dict[str, str] = {}
    for sid in _reviewed_session_ids(data):
        session = find_session(sid)
        if session is None:
            continue
        try:
            transcripts[sid] = render_trace_html(session.trace_path)
        except Exception:
            continue
    out.write_text(render_page(data, transcripts=transcripts), encoding="utf-8")
    return out


def export_markdown(out_path: Path | None = None) -> Path:
    """Write a ranked digest of the open issues as Markdown."""
    out = out_path or (review_dir() / "review-digest.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    data = build_data()
    st = data["stats"]
    lines = [
        "# Makker session review",
        "",
        f"{st['sessions_reviewed']} of {st['sessions_total']} sessions reviewed, "
        f"{st['issues_open']} open issues "
        f"({st.get('current_version') and 'Makker v' + st['current_version']}).",
        "",
    ]
    open_issues = [i for i in data["issues"] if i["status"] == "open"]
    for i in open_issues:
        stale = " _(possibly fixed)_" if i["stale"] else ""
        lines += [
            f"## [{i['severity']}] {i['title']}{stale}",
            "",
            f"- seen {i['count']}x in {len(i['sessions'])} session(s) "
            f"| {i['category']} | fix: {i['fix_target']} | priority {i['priority']}",
        ]
        if i["examples"]:
            lines += ["", "Evidence:", "", f"> {i['examples'][-1]}"]
        lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    return out
