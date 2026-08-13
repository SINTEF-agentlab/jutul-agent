"""Per-session SQLite trace log.

Append-only event store at ``<state_dir>/trace.sqlite``. Renderers
(see ``core.transcript``) consume it to produce transcripts and other
artifacts.

Common event kinds include ``session_start``, ``session_end``,
``message_user``, ``message_assistant``, ``message_reasoning``,
``tool_call``, ``tool_result``, ``hitl_request``, ``hitl_response``,
and ``artifact``. An ``artifact`` payload looks like::

    {
        "path": "artifacts/plot-<id>.png",
        "mime": "image/png",
        "caption": "optional label",
        "tool_call_id": "<id or null>",
        "format": "png",
        "size_px": [800, 500],
        "dpi": null,
        "slot": null,
        "source_code": "<julia snippet>",
    }
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp    TEXT NOT NULL,
    kind         TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_kind_idx ON events(kind);
"""


@dataclass(frozen=True)
class Event:
    id: int
    timestamp: str
    kind: str
    payload: dict[str, Any]


class TraceLog:
    """Append-only event log for a single session."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._readonly = False
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)

    @classmethod
    def open_readonly(cls, path: Path) -> TraceLog:
        """Open an existing trace for reading without touching it.

        The writing constructor creates the file, ensures the schema, and
        flips the journal mode; none of that belongs when inspecting another
        session's trace (possibly one being written right now). Raises
        ``sqlite3.OperationalError`` when the file does not exist.
        """
        self = cls.__new__(cls)
        self.path = path
        self._readonly = True
        self._conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        return self

    def append(self, kind: str, payload: dict[str, Any]) -> None:
        if self._readonly:
            raise RuntimeError(f"trace {self.path} is open read-only")
        ts = datetime.now(UTC).isoformat()
        self._conn.execute(
            "INSERT INTO events (timestamp, kind, payload_json) VALUES (?, ?, ?)",
            (ts, kind, json.dumps(payload, default=str)),
        )

    def iter_events(self) -> list[Event]:
        rows = self._conn.execute(
            "SELECT id, timestamp, kind, payload_json FROM events ORDER BY id"
        ).fetchall()
        return [Event(id=r[0], timestamp=r[1], kind=r[2], payload=json.loads(r[3])) for r in rows]

    def events_after(self, event_id: int) -> list[Event]:
        """Events newer than ``event_id`` (ordered by id).

        For incremental polling — e.g. flushing the side outputs produced since the
        last flush — so a long trace is not re-read and re-decoded each time."""
        rows = self._conn.execute(
            "SELECT id, timestamp, kind, payload_json FROM events WHERE id > ? ORDER BY id",
            (event_id,),
        ).fetchall()
        return [Event(id=r[0], timestamp=r[1], kind=r[2], payload=json.loads(r[3])) for r in rows]

    def max_id(self) -> int:
        """The id of the most recent event, or 0 if the trace is empty (cheap, by PK)."""
        row = self._conn.execute("SELECT id FROM events ORDER BY id DESC LIMIT 1").fetchone()
        return row[0] if row else 0

    def first_payload(self, kind: str) -> dict[str, Any] | None:
        """The payload of the earliest event of ``kind`` (uses the kind index), or None."""
        row = self._conn.execute(
            "SELECT payload_json FROM events WHERE kind = ? ORDER BY id LIMIT 1", (kind,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def first_timestamp(self) -> str | None:
        """ISO timestamp of the earliest event — when the session really started."""
        row = self._conn.execute("SELECT timestamp FROM events ORDER BY id LIMIT 1").fetchone()
        return row[0] if row else None

    def last_timestamp(self) -> str | None:
        """ISO timestamp of the most recent event — the session's last activity, or None.

        A cheap ``ORDER BY id DESC LIMIT 1`` (the primary key), so it stays fast on a
        long trace and is a reliable "last used" signal where a file mtime is not
        (WAL appends don't touch the main db file's mtime)."""
        row = self._conn.execute("SELECT timestamp FROM events ORDER BY id DESC LIMIT 1").fetchone()
        return row[0] if row else None

    def last_timestamp_of(self, kinds: Sequence[str]) -> str | None:
        """ISO timestamp of the most recent event of any of ``kinds``, or None.

        For "when was this last worked on", where reopening a session should not
        count: a resume writes lifecycle events, which ``last_timestamp`` sees.
        Uses the kind index.
        """
        if not kinds:
            return None
        holes = ",".join("?" * len(kinds))
        row = self._conn.execute(
            f"SELECT timestamp FROM events WHERE kind IN ({holes}) ORDER BY id DESC LIMIT 1",
            tuple(kinds),
        ).fetchone()
        return row[0] if row else None

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> TraceLog:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
