"""Unit tests for the trace log."""

from __future__ import annotations

from pathlib import Path

from makker.trace import TraceLog


def test_append_and_iter(tmp_path: Path) -> None:
    log = TraceLog(tmp_path / "trace.sqlite")
    try:
        log.append("session_start", {"session_id": "abc"})
        log.append("message_user", {"content": "hi"})
        log.append("tool_call", {"name": "run_julia", "args": {"code": "1+1"}})
        log.append("tool_result", {"name": "run_julia", "content": "2"})
        log.append("session_end", {"session_id": "abc"})
        events = log.iter_events()
    finally:
        log.close()

    kinds = [e.kind for e in events]
    assert kinds == [
        "session_start",
        "message_user",
        "tool_call",
        "tool_result",
        "session_end",
    ]
    assert events[2].payload["args"] == {"code": "1+1"}
    assert all(e.timestamp for e in events)
    assert [e.id for e in events] == sorted(e.id for e in events)


def test_creates_parent_dir(tmp_path: Path) -> None:
    db = tmp_path / "nested" / "deeper" / "trace.sqlite"
    log = TraceLog(db)
    log.append("session_start", {"session_id": "x"})
    log.close()
    assert db.exists()


def test_events_after_and_max_id(tmp_path: Path) -> None:
    # Incremental helpers used by the server's side-output flush and high-water mark:
    # events_after(id) returns only newer events; max_id() is the latest id (0 empty).
    log = TraceLog(tmp_path / "trace.sqlite")
    try:
        assert log.max_id() == 0  # empty trace
        assert log.events_after(0) == []
        log.append("session_start", {"session_id": "abc"})
        log.append("message_user", {"content": "hi"})
        first_two = log.iter_events()
        mark = first_two[-1].id
        log.append("message_assistant", {"content": "yo"})

        assert log.max_id() == mark + 1
        after = log.events_after(mark)
        assert [e.kind for e in after] == ["message_assistant"]  # only the new one
        assert log.events_after(log.max_id()) == []  # nothing past the latest
    finally:
        log.close()
