"""Tests for adding extra working directories (``/add-dir``).

The agent's filesystem uses real paths, so an added folder is read and written
at its real absolute path by the normal file tools. ``add_dir`` validates and
records the folder so the session can list it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from makker.agent.added_dirs import AddDirError, add_dir, added_dirs
from makker.agent.builder import build_backend


@pytest.fixture
def backend(tmp_path: Path):
    """A backend rooted at an empty workspace."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return build_backend(workspace=workspace)


@pytest.fixture
def extra(tmp_path: Path) -> Path:
    folder = tmp_path / "data"
    folder.mkdir()
    (folder / "notes.txt").write_text("hello from outside\n", encoding="utf-8")
    return folder


def test_added_dir_is_read_and_written_at_its_real_path(backend, extra: Path) -> None:
    entry = add_dir(backend, extra, workspace=backend.default.cwd)
    assert entry.path == extra.resolve()
    # The real-path backend reads and writes the added folder at its real path.
    content = (backend.read(str(extra / "notes.txt")).file_data or {}).get("content", "")
    assert "hello from outside" in content
    assert backend.write(str(extra / "new.txt"), "added").error is None
    assert (extra / "new.txt").read_text(encoding="utf-8") == "added"


def test_add_dir_is_idempotent(backend, extra: Path) -> None:
    first = add_dir(backend, extra, workspace=backend.default.cwd)
    second = add_dir(backend, extra, workspace=backend.default.cwd)
    assert first == second
    assert added_dirs(backend) == [first]


def test_add_dir_disambiguates_same_basename(backend, tmp_path: Path) -> None:
    a = tmp_path / "a" / "shared"
    b = tmp_path / "b" / "shared"
    a.mkdir(parents=True)
    b.mkdir(parents=True)

    first = add_dir(backend, a, workspace=backend.default.cwd)
    second = add_dir(backend, b, workspace=backend.default.cwd)

    assert first.name == "shared"
    assert second.name == "shared-2"
    assert {entry.path for entry in added_dirs(backend)} == {a.resolve(), b.resolve()}


def test_add_dir_resolves_relative_to_workspace(backend) -> None:
    workspace = backend.default.cwd
    nested = Path(workspace) / "sub" / "child"
    nested.mkdir(parents=True)
    entry = add_dir(backend, "sub/child", workspace=workspace)
    assert entry.path == nested.resolve()


def test_add_dir_rejects_missing(backend, tmp_path: Path) -> None:
    with pytest.raises(AddDirError, match="no such directory"):
        add_dir(backend, tmp_path / "ghost", workspace=backend.default.cwd)


def test_add_dir_rejects_file(backend, tmp_path: Path) -> None:
    f = tmp_path / "file.txt"
    f.write_text("x", encoding="utf-8")
    with pytest.raises(AddDirError, match="not a directory"):
        add_dir(backend, f, workspace=backend.default.cwd)


def test_add_dir_rejects_workspace_itself(backend) -> None:
    with pytest.raises(AddDirError, match="already the working directory"):
        add_dir(backend, backend.default.cwd, workspace=backend.default.cwd)


def test_added_dirs_lists_paths(backend, extra: Path) -> None:
    assert added_dirs(backend) == []
    add_dir(backend, extra, workspace=backend.default.cwd)
    listed = added_dirs(backend)
    assert len(listed) == 1
    assert listed[0].path == extra.resolve()


def test_build_backend_records_extra_dirs(tmp_path: Path, extra: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    backend = build_backend(workspace=workspace, added_dirs=[extra])
    assert [entry.path for entry in added_dirs(backend)] == [extra.resolve()]
