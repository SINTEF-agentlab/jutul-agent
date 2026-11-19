"""Tests for workspace-scoped agent memory."""

from __future__ import annotations

from pathlib import Path

from makker.agent.builder import build_backend
from makker.agent.memory import (
    MEMORY_INDEX_FILENAME,
    ensure_memory_dir,
    make_remember_tool,
)
from makker.paths import workspace_memory_dir, workspace_state_dir


def test_workspace_memory_dir_is_under_workspace_state(tmp_path: Path, monkeypatch) -> None:
    from makker.paths import set_state_home, set_workspace_root

    set_workspace_root(tmp_path)
    set_state_home(tmp_path / "state")

    memory_dir = workspace_memory_dir()
    assert memory_dir == workspace_state_dir() / "memory"
    assert "workspaces" in memory_dir.parts


def test_ensure_memory_dir_seeds_index_once(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    ensure_memory_dir(memory_dir)
    index = memory_dir / MEMORY_INDEX_FILENAME
    assert index.exists()
    first = index.read_text(encoding="utf-8")
    assert "Memory index" in first

    index.write_text("# customized\n", encoding="utf-8")
    ensure_memory_dir(memory_dir)
    assert index.read_text(encoding="utf-8") == "# customized\n"


def test_memory_is_read_and_written_at_its_real_path(tmp_path: Path) -> None:
    # Memory is real files the agent reads and edits at their real path
    # (and the user can open them directly).
    memory_dir = ensure_memory_dir(tmp_path / "memory")
    backend = build_backend(workspace=tmp_path)

    index = str(memory_dir / MEMORY_INDEX_FILENAME)
    assert "Memory index" in (backend.read(index).file_data or {}).get("content", "")
    assert backend.write(str(memory_dir / "note.md"), "hi").error is None
    assert (memory_dir / "note.md").read_text(encoding="utf-8") == "hi"


def test_memory_middleware_builds_over_the_real_dir(tmp_path: Path) -> None:
    from deepagents.middleware.memory import MemoryMiddleware

    from makker.agent.memory import build_memory_middleware

    memory_dir = ensure_memory_dir(tmp_path / "memory")
    backend = build_backend(workspace=tmp_path)
    assert isinstance(build_memory_middleware(backend, memory_dir), MemoryMiddleware)


async def test_remember_writes_note_and_indexes_it(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    remember = make_remember_tool(memory_dir)

    result = await remember.ainvoke(
        {
            "content": "The user is Jakob at SINTEF; focus is BattMo battery sims.",
            "title": "User role",
            "kind": "user",
        }
    )

    note = memory_dir / "user-role.md"
    assert note.exists()
    note_text = note.read_text(encoding="utf-8")
    assert "type: user" in note_text
    assert "SINTEF" in note_text

    index = (memory_dir / MEMORY_INDEX_FILENAME).read_text(encoding="utf-8")
    assert "(file: `user-role.md`)" in index
    assert "user-role.md" in result


async def test_remember_updates_existing_note_without_duplicating_index(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    remember = make_remember_tool(memory_dir)

    await remember.ainvoke({"content": "first version", "title": "Quirk X"})
    await remember.ainvoke({"content": "second version", "title": "Quirk X"})

    note_text = (memory_dir / "quirk-x.md").read_text(encoding="utf-8")
    assert "second version" in note_text
    assert "first version" not in note_text

    index = (memory_dir / MEMORY_INDEX_FILENAME).read_text(encoding="utf-8")
    assert index.count("(file: `quirk-x.md`)") == 1


async def test_remember_falls_back_to_project_for_unknown_kind(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    remember = make_remember_tool(memory_dir)

    await remember.ainvoke({"content": "fact", "title": "Thing", "kind": "nonsense"})
    assert "type: project" in (memory_dir / "thing.md").read_text(encoding="utf-8")


def test_memory_note_path_sanitizes_and_suffixes(tmp_path) -> None:
    from makker.agent.memory import memory_note_path

    assert memory_note_path(tmp_path, "user-workflow") == tmp_path / "user-workflow.md"
    assert memory_note_path(tmp_path, "user-workflow.md") == tmp_path / "user-workflow.md"
    assert memory_note_path(tmp_path, "../../etc/passwd") == tmp_path / "passwd.md"


def test_render_memory_overview_lists_notes(tmp_path) -> None:
    from makker.agent.memory import ensure_memory_dir, render_memory_overview

    memory_dir = ensure_memory_dir(tmp_path / "memory")
    (memory_dir / "quirks.md").write_text("BattMo output API note\n", encoding="utf-8")

    body = render_memory_overview(memory_dir)
    assert "Memory index" in body
    assert "`quirks.md`" in body
    assert "/memory edit" in body

    empty = render_memory_overview(ensure_memory_dir(tmp_path / "fresh"))
    assert "No notes yet" in empty
