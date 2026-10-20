"""``jutul-agent sessions`` subcommand: list resumable sessions."""

from __future__ import annotations

import argparse

from jutul_agent.interfaces.cli._helpers import add_workspace_flags
from jutul_agent.session import SessionInfo, list_sessions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jutul-agent sessions",
        description=(
            "List this workspace's sessions, newest first. Resume one from the web "
            "UI's sidebar, or with `jutul-agent tui --resume <id>` (or `--continue` "
            "for the latest)."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Most sessions to show (default: 20; 0 shows all).",
    )
    add_workspace_flags(parser)
    return parser


def format_session_line(info: SessionInfo) -> str:
    title = info.title or "(untitled)"
    stamp = info.started.strftime("%Y-%m-%d %H:%M")
    return f"{stamp}  {info.session_id}  {title}"


def run(args: argparse.Namespace) -> int:
    sessions = list_sessions()
    if args.limit > 0:
        sessions = sessions[: args.limit]
    if not sessions:
        print("No sessions in this workspace yet.")
        return 0
    for info in sessions:
        print(format_session_line(info))
    return 0
