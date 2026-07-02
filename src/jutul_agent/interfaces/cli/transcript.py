"""``jutul-agent transcript`` subcommand."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from jutul_agent.interfaces.cli._helpers import add_workspace_flags
from jutul_agent.paths import workspace_state_dir
from jutul_agent.session import existing_output_dir, read_last_session, session_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jutul-agent transcript")
    parser.add_argument(
        "session_id",
        nargs="?",
        help="Session ID under the workspace's state dir. Omit to use the last session.",
    )
    parser.add_argument(
        "--format",
        choices=("html", "markdown"),
        default="html",
        help="Output format (default: html).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help=(
            "Write transcript to this path. Default: <session>/transcript.<ext>. "
            "Use '-' for stdout."
        ),
    )
    parser.add_argument(
        "--bundle",
        action="store_true",
        help=(
            "Also write a zip bundle containing transcript.html and artifacts/. "
            "Default path: <session>/transcript-bundle.zip"
        ),
    )
    add_workspace_flags(parser)
    return parser


def run(args: argparse.Namespace) -> int:
    from jutul_agent.trace import TraceLog
    from jutul_agent.transcript import bundle_transcript, render_html, render_markdown

    session_id = args.session_id or read_last_session()
    if session_id is None:
        print(
            f"error: no session id given and no last-session marker under {workspace_state_dir()}.",
            file=sys.stderr,
        )
        return 2

    db = session_dir(session_id) / "trace.sqlite"
    if not db.exists():
        print(f"No trace at {db}", file=sys.stderr)
        return 1
    # Artifacts live in the session's visible output dir under the workspace;
    # the state dir is the fallback for sessions where output fell back there.
    artifacts_root = existing_output_dir(session_id) or session_dir(session_id)
    log = TraceLog.open_readonly(db)
    try:
        events = log.iter_events()
        if args.format == "html":
            content = render_html(
                events, artifact_dirs=[artifacts_root, artifacts_root / "artifacts"]
            )
            ext = "html"
        else:
            content = render_markdown(events)
            ext = "md"
    finally:
        log.close()

    output = args.output
    if output is None:
        output = str(session_dir(session_id) / f"transcript.{ext}")

    if output == "-":
        sys.stdout.write(content)
        return 0

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    print(out_path)

    if args.format == "html" and args.bundle:
        session_path = session_dir(session_id)
        bundle_path = session_path / "transcript-bundle.zip"
        if out_path != session_path / "transcript.html":
            bundle_path = out_path.with_suffix(".zip")
        bundle_transcript(artifacts_root, out_path, bundle_path)
        print(bundle_path)

    return 0
