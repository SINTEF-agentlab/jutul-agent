"""Tests for --continue/--resume resolution and the sessions listing."""

from __future__ import annotations


def test_run_parser_accepts_continue_and_resume() -> None:
    from makker.interfaces.cli.run import build_parser

    parser = build_parser()
    args = parser.parse_args(["--continue"])
    assert args.continue_last is True and args.resume is None
    args = parser.parse_args(["--resume"])
    assert args.resume == ""
    args = parser.parse_args(["--resume", "2026-06-12"])
    assert args.resume == "2026-06-12"


def _seed_session(sid: str, title: str | None = None) -> None:
    from makker.paths import workspace_state_dir
    from makker.session import write_last_session
    from makker.trace import TraceLog

    d = workspace_state_dir() / "sessions" / sid
    d.mkdir(parents=True, exist_ok=True)
    TraceLog(d / "trace.sqlite").close()
    if title:
        (d / "title").write_text(title, encoding="utf-8")
    write_last_session(sid)


def test_resolve_resume_id_continue_and_prefix(tmp_path) -> None:
    import pytest

    from makker.interfaces.cli.run import _resolve_resume_id, _ResumeError, build_parser

    parser = build_parser()

    # Nothing on disk yet: --continue is a clean error, a fresh run is None.
    with pytest.raises(_ResumeError):
        _resolve_resume_id(parser.parse_args(["--continue"]))
    assert _resolve_resume_id(parser.parse_args([])) is None

    _seed_session("2026-06-12-2300-bbbb", "newest")
    assert _resolve_resume_id(parser.parse_args(["--continue"])) == "2026-06-12-2300-bbbb"
    assert _resolve_resume_id(parser.parse_args(["--resume", "2026-06-12"])) == (
        "2026-06-12-2300-bbbb"
    )
    with pytest.raises(_ResumeError):
        _resolve_resume_id(parser.parse_args(["--resume", "1999"]))
    with pytest.raises(_ResumeError):
        _resolve_resume_id(parser.parse_args(["--continue", "--resume", "x"]))


def test_sessions_cli_lists_sessions(tmp_path, capsys) -> None:
    from makker.interfaces.cli import sessions as sessions_cmd

    _seed_session("2026-06-12-2300-bbbb", "newest work")
    _seed_session("2026-06-10-0900-aaaa", "older work")
    rc = sessions_cmd.run(sessions_cmd.build_parser().parse_args([]))
    out = capsys.readouterr().out
    assert rc == 0
    lines = [line for line in out.splitlines() if line.strip()]
    assert "newest work" in lines[0]
    assert "older work" in lines[1]
