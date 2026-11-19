"""``makker review`` — the developer's improvement queue.

Subcommands (the first word is a verb; anything else is treated as a session id):

    makker review                  curated issues, most-seen first (default)
    makker review log              the raw per-review findings log
    makker review pending          sessions on this machine not yet reviewed
    makker review mine [--limit N] review pending sessions in bulk (API)
    makker review dashboard        open the interactive dashboard (serves locally)
    makker review export [--format html|md]   write a shareable file
    makker review curate [--rebuild]   (re)cluster the log into issues
    makker review resolve <id>     mark an issue fixed
    makker review dismiss <id>     mark an issue dismissed
    makker review delete <id>      remove an issue from the store entirely
    makker review prune --stale    resolve every issue badged "possibly fixed"
    makker review fix <id>         emit a fix-this-issue prompt for a coding agent
    makker review <session-id>     review a past session now (API), log + curate it
    makker review prompt <session> emit the critic prompt for a coding agent
    makker review ingest <session> ingest findings JSON a coding agent produced

Add --json to `review` (issues) for the curated store, e.g. so a coding agent can
reuse existing issue titles when it curates offline.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from makker.interfaces.cli._helpers import add_workspace_flags

_VERBS = {
    "issues",
    "log",
    "pending",
    "mine",
    "dashboard",
    "export",
    "curate",
    "resolve",
    "dismiss",
    "delete",
    "prune",
    "fix",
    "prompt",
    "ingest",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="makker review")
    parser.add_argument(
        "command",
        nargs="?",
        help="issues (default) | log | curate | resolve <id> | dismiss <id> | "
        "prompt <session> | ingest <session> | <session-id>",
    )
    parser.add_argument("arg", nargs="?", help="issue id, or session id for prompt/ingest")
    parser.add_argument(
        "--model",
        default=None,
        help="Reviewer model (default: $MAKKER_REVIEW_MODEL or the built-in default).",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="With `curate`, re-cluster the whole log from an empty store.",
    )
    parser.add_argument(
        "--no-store", action="store_true", help="Print a review without logging it."
    )
    parser.add_argument(
        "--from",
        dest="from_file",
        default=None,
        help="With `ingest`, read the findings JSON from this file (default: stdin).",
    )
    parser.add_argument(
        "--no-curate", action="store_true", help="With `ingest`, log findings without curating."
    )
    parser.add_argument("--limit", type=int, default=30, help="Max rows to list (default 30).")
    parser.add_argument(
        "--all",
        dest="include_reviewed",
        action="store_true",
        help="With `pending`/`mine`, include sessions already reviewed.",
    )
    parser.add_argument(
        "--json", action="store_true", help="With `pending`, emit the list as JSON."
    )
    parser.add_argument("-o", "--output", default=None, help="With `export`, the file to write.")
    parser.add_argument(
        "--format",
        dest="fmt",
        choices=["html", "md"],
        default="html",
        help="With `export`: a single self-contained html file, or a markdown digest.",
    )
    parser.add_argument(
        "--no-open", action="store_true", help="With `dashboard`, don't open a browser."
    )
    parser.add_argument(
        "--port", type=int, default=8765, help="Port for the dashboard server (default 8765)."
    )
    parser.add_argument(
        "--stale", action="store_true", help="With `prune`, resolve every 'possibly fixed' issue."
    )
    add_workspace_flags(parser)
    return parser


def run(args: argparse.Namespace) -> int:
    cmd = args.command
    if cmd is None or cmd == "issues":
        return _list_issues(args)
    if cmd == "log":
        return _list_log(args)
    if cmd == "pending":
        return _list_pending(args)
    if cmd == "mine":
        return _mine(args)
    if cmd == "dashboard":
        return _dashboard(args)
    if cmd == "export":
        return _export(args)
    if cmd == "curate":
        return _curate(args)
    if cmd in ("resolve", "dismiss"):
        return _set_status(args.arg, "fixed" if cmd == "resolve" else "dismissed")
    if cmd == "delete":
        return _delete(args.arg)
    if cmd == "prune":
        return _prune(args)
    if cmd == "fix":
        return _emit_fix(args.arg)
    if cmd == "prompt":
        return _emit_prompt(args)
    if cmd == "ingest":
        return _ingest(args)
    if cmd in _VERBS:  # an unhandled verb
        print(f"error: unknown review command {cmd!r}.", file=sys.stderr)
        return 2
    return _review_one(cmd, args)


# ---- coding-agent path (no API for the review itself) ----------------------


def _emit_prompt(args: argparse.Namespace) -> int:
    """Print the full critic prompt + transcript for a coding agent to review."""
    from makker.review.discovery import (
        eval_review_context,
        find_session,
        render_trace,
        session_simulator,
    )
    from makker.review.prompt import full_prompt

    session = find_session(args.arg or "")
    if session is None:
        print(
            f"error: `review prompt` needs a known session id (got {args.arg!r}).", file=sys.stderr
        )
        return 2
    print(
        full_prompt(
            render_trace(session.trace_path),
            simulator=session_simulator(session.trace_path),
            ground_truth=eval_review_context(session.trace_path),
        )
    )
    return 0


def _ingest(args: argparse.Namespace) -> int:
    """Ingest findings JSON a coding agent produced for a session."""
    import asyncio
    import json

    from makker.review.discovery import find_session, session_simulator
    from makker.review.reviewer import ingest_findings
    from makker.review.settings import review_model

    found = find_session(args.arg or "")
    session_id = found.session_id if found else (args.arg or "")
    title = found.title or "" if found else ""
    sim = session_simulator(found.trace_path) if found else None
    if not session_id:
        print("error: `review ingest` needs a session id.", file=sys.stderr)
        return 2
    raw = Path(args.from_file).read_text(encoding="utf-8") if args.from_file else sys.stdin.read()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"error: findings input is not valid JSON: {exc}", file=sys.stderr)
        return 2

    report = asyncio.run(
        ingest_findings(
            data,
            session_id=session_id,
            title=title,
            model_id=args.model or review_model(),
            source="coding-agent",
            simulator=sim,
            curate=not args.no_curate,
        )
    )
    print(f"Ingested {len(report.findings)} finding(s) for {session_id}.")
    return 0


# ---- mining sessions across the machine ------------------------------------


def _list_pending(args: argparse.Namespace) -> int:
    """List sessions on this machine, newest first, flagging the unreviewed ones."""
    import json

    from makker.review.discovery import discover_sessions

    sessions = discover_sessions(pending_only=not args.include_reviewed, limit=args.limit)
    if args.json:
        print(
            json.dumps(
                [
                    {
                        "session_id": s.session_id,
                        "title": s.title or "",
                        "workspace": s.workspace,
                        "started": s.started.isoformat(timespec="seconds"),
                        "reviewed": s.reviewed,
                        "trace": str(s.trace_path),
                    }
                    for s in sessions
                ],
                indent=2,
            )
        )
        return 0
    if not sessions:
        scope = "" if args.include_reviewed else "unreviewed "
        print(f"No {scope}sessions found.", file=sys.stderr)
        return 0
    label = "session(s)" if args.include_reviewed else "unreviewed session(s)"
    print(f"{len(sessions)} {label}, newest first.\n")
    for s in sessions:
        mark = "✓" if s.reviewed else "○"
        title = s.title or "(untitled)"
        print(f"{mark} {s.session_id}  {title}")
        print(f"    {s.started:%Y-%m-%d %H:%M} · workspace {s.workspace}")
    print("\nReview one with `makker review <id>` (API) or the coding-agent loop.")
    return 0


def _mine(args: argparse.Namespace) -> int:
    """Review pending sessions in bulk via the API, logging + curating each."""
    from makker.review.discovery import (
        discover_sessions,
        eval_review_context,
        render_trace,
        session_simulator,
    )
    from makker.review.reviewer import review_transcript, store_report
    from makker.review.settings import review_model

    sessions = discover_sessions(pending_only=not args.include_reviewed, limit=args.limit)
    if not sessions:
        print("Nothing to mine: no pending sessions.", file=sys.stderr)
        return 0

    model_id = args.model or review_model()
    print(
        f"Mining {len(sessions)} session(s) with {model_id}. "
        "Ctrl-C to stop; progress is saved per session.",
        file=sys.stderr,
    )

    async def run() -> int:
        flagged = 0
        for n, s in enumerate(sessions, start=1):
            try:
                transcript = render_trace(s.trace_path)
            except Exception as exc:
                print(f"  [{n}/{len(sessions)}] {s.session_id}: skip ({exc})", file=sys.stderr)
                continue
            if not transcript.strip():
                print(f"  [{n}/{len(sessions)}] {s.session_id}: skip (empty)", file=sys.stderr)
                continue
            try:
                report = await review_transcript(
                    transcript,
                    session_id=s.session_id,
                    title=s.title or "",
                    model_id=model_id,
                    simulator=session_simulator(s.trace_path),
                    ground_truth=eval_review_context(s.trace_path),
                )
                await store_report(report, store=True, curate=True, model_id=model_id)
            except Exception as exc:
                print(f"  [{n}/{len(sessions)}] {s.session_id}: error ({exc})", file=sys.stderr)
                continue
            k = len(report.findings)
            flagged += 1 if k else 0
            print(
                f"  [{n}/{len(sessions)}] {s.session_id}: {k} finding(s)",
                file=sys.stderr,
            )
        return flagged

    flagged = asyncio.run(run())
    print(f"\nDone. {flagged}/{len(sessions)} session(s) had findings. See `makker review`.")
    return 0


def _dashboard(args: argparse.Namespace) -> int:
    """Serve the interactive dashboard until interrupted."""
    from makker.review.server import serve_dashboard

    serve_dashboard(port=args.port, open_browser=not args.no_open)
    return 0


def _export(args: argparse.Namespace) -> int:
    """Write a shareable file: a single self-contained html, or a markdown digest."""
    from makker.review.export import export_html, export_markdown

    out = Path(args.output) if args.output else None
    path = export_markdown(out) if args.fmt == "md" else export_html(out)
    print(f"Wrote {path}")
    return 0


# ---- curated issues (default view) -----------------------------------------

_SEV_RANK = {"high": 2, "medium": 1, "low": 0}


def _list_issues(args: argparse.Namespace) -> int:
    import json

    from makker.review.issues import is_stale, issues_path, load_issues
    from makker.review.reviewer import app_version

    all_issues = list(load_issues().values())
    if args.json:
        # Everything (including dismissed) so a curating agent can reuse exact titles.
        print(json.dumps([i.to_dict() for i in all_issues], indent=2))
        return 0

    issues = [i for i in all_issues if i.status != "dismissed"]
    if not issues:
        print(
            "No issues yet. Run sessions with MAKKER_REVIEW=1, or `makker review\n"
            "curate` to cluster an existing findings log.",
            file=sys.stderr,
        )
        return 0
    current = app_version()
    issues.sort(key=lambda i: (_SEV_RANK.get(i.severity, 1), i.count), reverse=True)
    print(f"{issues_path()}")
    print(f"{len(issues)} open issue(s), most significant first.\n")
    for i in issues[: args.limit]:
        tag = "" if i.status == "open" else f" [{i.status}]"
        stale = "  ⚠ possibly fixed" if is_stale(i, current) else ""
        print(f"● {i.id}{tag}{stale}")
        print(f"    {i.title}")
        print(
            f"    seen {i.count}x in {len(i.sessions)} session(s) · {i.severity}/{i.category}"
            f" · fix: {i.fix_target} · last {i.last_seen[:10]}"
        )
        if i.examples:
            print(f"    e.g. {i.examples[-1][:160]}")
        print()
    return 0


def _set_status(issue_id: str | None, status: str) -> int:
    from makker.review.issues import set_status

    if not issue_id:
        print(
            f"error: `review {('resolve' if status == 'fixed' else 'dismiss')}` needs an issue id.",
            file=sys.stderr,
        )
        return 2
    if not set_status(issue_id, status):
        print(f"error: no issue with id {issue_id!r}.", file=sys.stderr)
        return 2
    print(f"{issue_id} → {status}")
    return 0


def _delete(issue_id: str | None) -> int:
    from makker.review.issues import delete_issue

    if not issue_id:
        print("error: `review delete` needs an issue id.", file=sys.stderr)
        return 2
    if not delete_issue(issue_id):
        print(f"error: no issue with id {issue_id!r}.", file=sys.stderr)
        return 2
    print(f"{issue_id} deleted")
    return 0


def _prune(args: argparse.Namespace) -> int:
    """Bulk-resolve every open issue badged 'possibly fixed'."""
    from makker.review.issues import is_stale, load_issues, set_status
    from makker.review.reviewer import app_version

    if not args.stale:
        print("error: `review prune` needs --stale.", file=sys.stderr)
        return 2
    current = app_version()
    stale = [i for i in load_issues().values() if is_stale(i, current)]
    for issue in stale:
        set_status(issue.id, "fixed")
    print(f"Resolved {len(stale)} possibly-fixed issue(s).")
    return 0


def _emit_fix(issue_id: str | None) -> int:
    """Print a fix-this-issue brief for a coding agent, with transcript pointers."""
    from makker.review.discovery import find_session
    from makker.review.issues import load_issues
    from makker.review.prompt import fix_prompt

    if not issue_id:
        print("error: `review fix` needs an issue id.", file=sys.stderr)
        return 2
    issue = load_issues().get(issue_id)
    if issue is None:
        print(f"error: no issue with id {issue_id!r}.", file=sys.stderr)
        return 2
    paths = [str(s.trace_path) for sid in issue.sessions if (s := find_session(sid))]
    print(fix_prompt(issue, transcript_paths=paths or None))
    return 0


def _curate(args: argparse.Namespace) -> int:
    from makker.review.curate import curate_log
    from makker.review.settings import review_model

    model_id = args.model or review_model()
    verb = "Rebuilding" if args.rebuild else "Updating"
    print(f"{verb} issues from the findings log with {model_id}…", file=sys.stderr)
    issues = asyncio.run(curate_log(model_id=model_id, rebuild=args.rebuild))
    print(f"{len(issues)} issue(s) tracked. See `makker review`.")
    return 0


# ---- raw findings log -------------------------------------------------------


def _list_log(args: argparse.Namespace) -> int:
    from makker.review.findings import load_reports, review_log_path

    reports = load_reports()
    if not reports:
        print("No reviews logged yet.", file=sys.stderr)
        return 0
    flagged = [r for r in reports if not r.ok]
    print(f"{review_log_path()}")
    print(f"{len(reports)} reviews logged, {len(flagged)} with findings.\n")
    for report in reports[-args.limit :]:
        head = f"● {report.session_id}"
        if report.title:
            head += f" — {report.title}"
        print(f"{head}  [{report.model}, {report.created_at}]")
        if report.summary:
            print(f"  {report.summary}")
        for f in sorted(report.findings, key=lambda x: _SEV_RANK.get(x.severity, 1), reverse=True):
            print(f"  [{f.severity}/{f.category}] {f.title}  → fix: {f.fix_target}")
        print()
    return 0


# ---- review a session on demand --------------------------------------------


def _review_one(session_arg: str, args: argparse.Namespace) -> int:
    from makker.review.discovery import (
        eval_review_context,
        find_session,
        render_trace,
        session_simulator,
    )
    from makker.review.findings import append_report
    from makker.review.reviewer import review_transcript
    from makker.review.settings import review_model

    session = find_session(session_arg)
    if session is None:
        print(f"error: no session matches {session_arg!r}.", file=sys.stderr)
        return 2
    session_id = session.session_id

    try:
        transcript = render_trace(session.trace_path)
    except Exception as exc:
        print(f"error: could not read session {session_id}: {exc}", file=sys.stderr)
        return 1
    if not transcript.strip():
        print(f"error: session {session_id} has no recorded events.", file=sys.stderr)
        return 1

    model_id = args.model or review_model()
    print(f"Reviewing {session_id} with {model_id}…", file=sys.stderr)
    report = asyncio.run(
        review_transcript(
            transcript,
            session_id=session_id,
            title=session.title or "",
            model_id=model_id,
            simulator=session_simulator(session.trace_path),
            ground_truth=eval_review_context(session.trace_path),
        )
    )
    if not args.no_store:
        append_report(report)
        if report.findings:
            from makker.review.curate import curate_report

            asyncio.run(curate_report(report, model_id=model_id))

    print(f"\n● {session_id}  ({len(report.findings)} finding(s))")
    if report.summary:
        print(f"  {report.summary}")
    for f in sorted(report.findings, key=lambda x: _SEV_RANK.get(x.severity, 1), reverse=True):
        print(f"  [{f.severity}/{f.category}] {f.title}  → fix: {f.fix_target}")
        if f.evidence:
            print(f"      evidence:   {f.evidence}")
        if f.suggestion:
            print(f"      suggestion: {f.suggestion}")
    return 0
