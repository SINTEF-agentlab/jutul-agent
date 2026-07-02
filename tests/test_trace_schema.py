"""The trace event vocabulary as a contract: constants, docs, read-only access."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from jutul_agent.trace import TraceLog, schema

DOCS_TRACE = Path(__file__).parent.parent / "docs" / "trace.md"


def test_every_kind_is_documented() -> None:
    """docs/trace.md is the payload contract; a new kind must land there too."""
    doc = DOCS_TRACE.read_text(encoding="utf-8")
    missing = [kind for kind in sorted(schema.ALL_KINDS) if f"`{kind}`" not in doc]
    assert not missing, f"kinds missing from docs/trace.md: {missing}"


def test_open_readonly_reads_but_rejects_writes(tmp_path: Path) -> None:
    path = tmp_path / "trace.sqlite"
    with TraceLog(path) as log:
        log.append(schema.MESSAGE_USER, {"content": "hello"})

    ro = TraceLog.open_readonly(path)
    try:
        events = ro.iter_events()
        assert [e.kind for e in events] == [schema.MESSAGE_USER]
        assert ro.first_timestamp() == events[0].timestamp
        with pytest.raises(RuntimeError):
            ro.append(schema.MESSAGE_USER, {"content": "nope"})
    finally:
        ro.close()


def test_open_readonly_does_not_create_a_missing_trace(tmp_path: Path) -> None:
    path = tmp_path / "absent" / "trace.sqlite"
    with pytest.raises(sqlite3.OperationalError):
        TraceLog.open_readonly(path)
    assert not path.exists()


def test_compaction_payload_shape_matches_renderers() -> None:
    """Writers and renderers meet at this payload; keep the keys in one place."""
    payload = schema.compaction_payload(summarized=21, kept=9, offloaded=True, manual=True)
    assert payload == {"summarized": 21, "kept": 9, "offloaded": True, "manual": True}


def test_artifact_payload_carries_the_full_field_set() -> None:
    payload = schema.artifact_payload(path="artifacts/p.png", mime="image/png", caption="p")
    for key in ("tool_call_id", "format", "kind", "size_px", "dpi", "slot", "poster", "live_url"):
        assert key in payload
