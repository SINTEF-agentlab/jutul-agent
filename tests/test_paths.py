"""Tests for path helpers."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from makker.paths import (
    resolve_in_workspace,
    set_state_home,
    set_workspace_root,
    state_home,
    workspace_hash,
    workspace_state_dir,
)


def test_workspace_hash_is_deterministic(tmp_path: Path) -> None:
    ws = tmp_path / "project"
    ws.mkdir()
    expected = hashlib.sha256(str(ws.resolve()).encode("utf-8")).hexdigest()[:12]
    assert workspace_hash(ws) == expected


def test_workspace_hash_differs_for_different_paths(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    assert workspace_hash(a) != workspace_hash(b)


def test_state_home_honours_override(tmp_path: Path) -> None:
    custom = tmp_path / "custom-state"
    set_state_home(custom)
    assert state_home() == custom.resolve()


def test_state_home_defaults_to_xdg(monkeypatch, tmp_path: Path) -> None:
    xdg = tmp_path / "xdg"
    monkeypatch.setenv("XDG_DATA_HOME", str(xdg))
    set_state_home(None)
    assert state_home() == (xdg / "makker").resolve()


def test_workspace_state_dir_under_state_home(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    state = tmp_path / "state"
    set_workspace_root(ws)
    set_state_home(state)
    assert workspace_state_dir() == state / "workspaces" / workspace_hash(ws)


def test_resolve_in_workspace_handles_virtual_paths(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    set_workspace_root(ws)

    # Relative paths stay relative to the workspace.
    assert resolve_in_workspace("experiments/report.html") == ws / "experiments/report.html"

    # Leading-slash virtual paths (from the agent's view) map to workspace root.
    assert resolve_in_workspace("/experiments/report.html") == ws / "experiments/report.html"
    assert (
        resolve_in_workspace("/workspace/experiments/report.html") == ws / "experiments/report.html"
    )

    # Real absolute paths inside the workspace are preserved.
    inside = ws / "x" / "y.html"
    assert resolve_in_workspace(str(inside)) == inside


def test_resolve_in_workspace_rejects_outside_paths(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    set_workspace_root(ws)

    # Real host paths outside the workspace are rejected, matching the file
    # tools' rule, rather than silently rerooted under the workspace. The
    # /tmp literal is host-absolute only on POSIX; on Windows a leading-slash
    # string has no drive letter and is a virtual path by design.
    if os.name == "posix":
        assert resolve_in_workspace("/tmp/random.html") is None
    assert resolve_in_workspace(str(tmp_path / "elsewhere.txt")) is None

    # `..` escapes are rejected whatever their form.
    assert resolve_in_workspace("../outside/secret.txt") is None
    assert resolve_in_workspace("/../outside/secret.txt") is None

    # Empty input resolves to nothing rather than the workspace root.
    assert resolve_in_workspace("") is None
