"""Tests for session path helpers."""

from __future__ import annotations

from pathlib import Path

from jutul_agent.paths import set_state_home, set_workspace_root, workspace_state_dir
from jutul_agent.session import Session, read_last_session, sessions_root, write_last_session


def test_last_session_round_trip_via_state_root(tmp_path: Path) -> None:
    write_last_session("session-abc", state_root=tmp_path)
    assert read_last_session(state_root=tmp_path) == "session-abc"


def test_last_session_round_trip_via_workspace_state_dir(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    state = tmp_path / "state"
    set_workspace_root(ws)
    set_state_home(state)

    write_last_session("session-xyz")
    assert read_last_session() == "session-xyz"
    assert (workspace_state_dir() / "last-session").exists()


def test_sessions_root_honours_explicit_state_root(tmp_path: Path) -> None:
    assert sessions_root(tmp_path) == tmp_path / "sessions"


def test_default_session_id_is_dated_and_sortable() -> None:
    from datetime import datetime

    from jutul_agent.paths import is_dated_session_id
    from jutul_agent.session import default_session_id

    earlier = default_session_id(datetime(2026, 6, 12, 9, 5))
    later = default_session_id(datetime(2026, 6, 12, 23, 15))
    assert is_dated_session_id(earlier)
    assert is_dated_session_id(later)
    assert earlier.startswith("2026-06-12-0905-")
    assert sorted([later, earlier]) == [earlier, later]


def test_session_output_dir_uses_dated_id_directly(tmp_path: Path) -> None:
    from jutul_agent.paths import session_output_dir

    dated = session_output_dir("2026-06-12-2315-3f2a", workspace=tmp_path)
    assert dated.name == "2026-06-12-2315-3f2a"
    legacy = session_output_dir("0a1b2c3d-aaaa-bbbb-cccc-0123456789ab", workspace=tmp_path)
    assert legacy.name.endswith("-0a1b2c3d")  # legacy UUIDs keep the date prefix


def test_derive_session_title_and_slug() -> None:
    from jutul_agent.session import _slugify_title, derive_session_title

    title = derive_session_title("  Set up a constant-current discharge\nplot it too")
    assert title == "Set up a constant-current discharge"
    assert _slugify_title(title) == "set-up-a-constant-current"

    long = derive_session_title("word " * 40)
    assert len(long) <= 81  # 80 chars + ellipsis
    assert long.endswith("…")
    assert derive_session_title("  \n\t ") == ""


def _title_session(tmp_path: Path):
    from jutul_agent.trace import TraceLog

    state = tmp_path / "state-dir"
    state.mkdir(parents=True)
    out = tmp_path / "out" / "2026-06-12-0101-abcd"
    out.mkdir(parents=True)
    trace = TraceLog(state / "trace.sqlite")
    return Session(
        julia=None,
        state_dir=state,
        output_dir=out,
        trace=trace,
        simulator=None,
        session_id="2026-06-12-0101-abcd",
    )


def test_adopt_title_renames_output_and_records(tmp_path: Path) -> None:
    from jutul_agent.session import read_session_title

    session = _title_session(tmp_path)
    session.adopt_title("Run a chen_2020 discharge and plot voltage")

    assert session.session_id == "2026-06-12-0101-abcd"  # immutable
    assert session.output_dir.name == "2026-06-12-0101-abcd-run-a-chen-2020-discharge-and"
    assert session.output_dir.exists()
    assert read_session_title(session.state_dir) == "Run a chen_2020 discharge and plot voltage"
    kinds = [event.kind for event in session.trace.iter_events()]
    assert "session_title" in kinds

    # Idempotent: the second prompt of the session never re-titles.
    before = session.output_dir
    session.adopt_title("Another prompt entirely")
    assert session.output_dir == before
    session.trace.close()


def test_existing_output_dir_prefers_the_folder_with_artifacts(tmp_path: Path) -> None:
    # A stray empty ``<sid>/`` next to the real ``<sid>-<slug>/`` (the title slug)
    # must not shadow it on resume, or the resumed session's artifacts 404. Prefer
    # whichever folder actually holds the ``artifacts/``.
    from jutul_agent.paths import session_output_dir
    from jutul_agent.session import existing_output_dir

    sid = "2026-06-12-0101-abcd"
    bare = session_output_dir(sid)  # workspace is the autouse tmp_path
    slugged = bare.with_name(bare.name + "-do-a-thing")
    (bare / "artifacts").mkdir(parents=True)  # stray, empty
    (slugged / "artifacts").mkdir(parents=True)
    (slugged / "artifacts" / "plot.png").write_bytes(b"x")  # the real output

    assert existing_output_dir(sid) == slugged


def test_existing_output_dir_prefers_nonempty_dir_for_report_only_session(tmp_path: Path) -> None:
    # A chat/report-only session keeps ``report.html`` at the dir root, not under
    # ``artifacts/``. With a stray empty ``<sid>/`` beside the real ``<sid>-<slug>/``,
    # the artifacts check matches neither, so it must fall back to the NON-EMPTY dir
    # (the slug one holding the report), not the empty ``<sid>/`` that sorts first.
    from jutul_agent.paths import session_output_dir
    from jutul_agent.session import existing_output_dir

    sid = "2026-06-12-0202-bbbb"
    bare = session_output_dir(sid)
    slugged = bare.with_name(bare.name + "-report-only")
    bare.mkdir(parents=True)  # stray, empty
    slugged.mkdir(parents=True)
    (slugged / "report.html").write_text("<html></html>", encoding="utf-8")  # the real output

    assert existing_output_dir(sid) == slugged


def test_existing_output_dir_ignores_stray_with_empty_artifacts_subdir(tmp_path: Path) -> None:
    # A stray ``<sid>/`` whose only content is an empty ``artifacts/`` subdir must
    # not count as having output and win over the real ``<sid>-<slug>/`` that holds
    # report.html at its root.
    from jutul_agent.paths import session_output_dir
    from jutul_agent.session import existing_output_dir

    sid = "2026-06-12-0303-cccc"
    bare = session_output_dir(sid)
    slugged = bare.with_name(bare.name + "-report")
    (bare / "artifacts").mkdir(parents=True)  # stray: only an EMPTY artifacts/ subdir
    slugged.mkdir(parents=True)
    (slugged / "report.html").write_text("<html></html>", encoding="utf-8")

    assert existing_output_dir(sid) == slugged


def test_retitle_overwrites_title_but_keeps_output_dir(tmp_path: Path) -> None:
    from jutul_agent.session import read_session_title

    session = _title_session(tmp_path)
    session.adopt_title("Run a chen_2020 discharge and plot voltage")
    out_after_adopt = session.output_dir  # carries the first-prompt slug

    # An LLM title replaces the displayed/stored name without renaming the folder
    # (no open handles disturbed) — only the title file and a trace event change.
    session.retitle("Chen2020 CC Discharge Voltage")
    assert session.title == "Chen2020 CC Discharge Voltage"
    assert read_session_title(session.state_dir) == "Chen2020 CC Discharge Voltage"
    assert session.output_dir == out_after_adopt  # folder slug unchanged
    assert [e.kind for e in session.trace.iter_events()].count("session_title") == 2

    session.retitle("   ")  # blank is a no-op
    assert session.title == "Chen2020 CC Discharge Voltage"
    session.trace.close()


def test_adopt_title_without_usable_slug_keeps_dir(tmp_path: Path) -> None:
    session = _title_session(tmp_path)
    before = session.output_dir
    session.adopt_title("???!!!")
    assert session.output_dir == before  # title with no slug → no rename
    session.trace.close()


def test_adopt_title_skips_rename_when_output_is_state_dir(tmp_path: Path) -> None:
    from jutul_agent.session import read_session_title
    from jutul_agent.trace import TraceLog

    state = tmp_path / "solo"
    state.mkdir()
    trace = TraceLog(state / "trace.sqlite")
    session = Session(
        julia=None,
        state_dir=state,
        output_dir=state,
        trace=trace,
        simulator=None,
        session_id="2026-06-12-0101-eeee",
    )
    session.adopt_title("Fallback layout")
    assert session.output_dir == state  # never rename the live state dir
    assert read_session_title(state) == "Fallback layout"
    trace.close()


def test_list_sessions_sorts_newest_first_with_titles(tmp_path: Path) -> None:
    from jutul_agent.session import list_sessions
    from jutul_agent.trace import TraceLog

    root = tmp_path / "sessions"
    for sid, title in [
        ("2026-06-10-0900-aaaa", "older work"),
        ("2026-06-12-2300-bbbb", "newest work"),
    ]:
        d = root / sid
        d.mkdir(parents=True)
        TraceLog(d / "trace.sqlite").close()
        (d / "title").write_text(title + "\n", encoding="utf-8")
    (root / "not-a-session").mkdir()  # no trace.sqlite → ignored

    infos = list_sessions(tmp_path)
    assert [info.session_id for info in infos] == [
        "2026-06-12-2300-bbbb",
        "2026-06-10-0900-aaaa",
    ]
    assert infos[0].title == "newest work"
    assert infos[0].started.strftime("%H%M") == "2300"


def test_resolve_session_id_exact_and_prefix(tmp_path: Path) -> None:
    from jutul_agent.session import resolve_session_id
    from jutul_agent.trace import TraceLog

    root = tmp_path / "sessions"
    for sid in ("2026-06-10-0900-aaaa", "2026-06-10-0900-abcd", "2026-06-12-2300-bbbb"):
        d = root / sid
        d.mkdir(parents=True)
        TraceLog(d / "trace.sqlite").close()

    assert resolve_session_id("2026-06-12-2300-bbbb", state_root=tmp_path) is not None
    assert resolve_session_id("2026-06-12", state_root=tmp_path) == "2026-06-12-2300-bbbb"
    assert resolve_session_id("2026-06-10-0900-a", state_root=tmp_path) is None  # ambiguous
    assert resolve_session_id("nope", state_root=tmp_path) is None
    assert resolve_session_id("", state_root=tmp_path) is None


def test_session_resume_reopens_state_and_finds_titled_output(
    tmp_path: Path, fake_julia, fake_adapter
) -> None:
    original = Session.create(julia=fake_julia, state_root=tmp_path, simulator=fake_adapter)
    original.adopt_title("Resume me later please")
    titled_output = original.output_dir
    original.finalize()

    resumed = Session.resume(
        julia=fake_julia,
        simulator=fake_adapter,
        session_id=original.session_id,
        state_root=tmp_path,
    )
    assert resumed.resumed is True
    assert resumed.session_id == original.session_id
    assert resumed.state_dir == original.state_dir
    assert resumed.output_dir == titled_output  # rediscovered the slugged folder
    assert resumed.title == "Resume me later please"
    kinds = [event.kind for event in resumed.trace.iter_events()]
    assert kinds.count("session_start") == 1
    assert kinds[-1] == "session_resume"
    resumed.finalize()


def test_session_resume_requires_existing_trace(tmp_path: Path, fake_julia, fake_adapter) -> None:
    import pytest

    with pytest.raises(FileNotFoundError):
        Session.resume(
            julia=fake_julia,
            simulator=fake_adapter,
            session_id="2026-01-01-0000-dead",
            state_root=tmp_path,
        )


def test_session_create_defaults_surface_to_tui(tmp_path: Path, fake_julia, fake_adapter) -> None:
    session = Session.create(julia=fake_julia, state_root=tmp_path, simulator=fake_adapter)
    assert session.surface == "tui"
    session.finalize()


def test_session_surface_round_trips_through_create_and_resume(
    tmp_path: Path, fake_julia, fake_adapter
) -> None:
    original = Session.create(
        julia=fake_julia, state_root=tmp_path, simulator=fake_adapter, surface="web"
    )
    assert original.surface == "web"
    original.finalize()

    resumed = Session.resume(
        julia=fake_julia,
        simulator=fake_adapter,
        session_id=original.session_id,
        state_root=tmp_path,
        surface="web",
    )
    assert resumed.surface == "web"
    resumed.finalize()
