"""Tests for ephemeral session memory."""

from __future__ import annotations

from pathlib import Path

from jutul_agent.agent.memory import MEMORY_INDEX_FILENAME, ensure_memory_dir
from jutul_agent.lab.fakes import FakeJulia, make_fake_adapter
from jutul_agent.paths import workspace_memory_dir
from jutul_agent.session import Session


def test_ephemeral_memory_uses_temp_dir(tmp_path: Path, monkeypatch) -> None:
    from jutul_agent.paths import set_state_home, set_workspace_root

    set_workspace_root(tmp_path / "workspace")
    set_state_home(tmp_path / "state")
    (tmp_path / "workspace").mkdir()

    adapter = make_fake_adapter(tmp_path)
    session = Session.create(
        julia=FakeJulia(),  # type: ignore[arg-type]
        simulator=adapter,
        state_root=tmp_path / "state" / "workspaces" / "x",
        ephemeral_memory=True,
    )

    mem = session.memory_dir(workspace_memory=workspace_memory_dir())
    assert mem.exists()
    assert "jutul-agent-ephemeral-" in str(mem)
    # The agent builder seeds the index when it mounts the dir.
    assert (ensure_memory_dir(mem) / MEMORY_INDEX_FILENAME).exists()

    session.finalize()
    assert not mem.exists()


def test_persistent_memory_uses_workspace_dir(tmp_path: Path, monkeypatch) -> None:
    from jutul_agent.paths import set_state_home, set_workspace_root

    set_workspace_root(tmp_path / "workspace")
    set_state_home(tmp_path / "state")
    (tmp_path / "workspace").mkdir()

    adapter = make_fake_adapter(tmp_path)
    session = Session.create(
        julia=FakeJulia(),  # type: ignore[arg-type]
        simulator=adapter,
        state_root=tmp_path / "state" / "workspaces" / "x",
        ephemeral_memory=False,
    )

    mem = session.memory_dir(workspace_memory=workspace_memory_dir())
    assert mem == workspace_memory_dir()
    session.finalize()
    # Memory dir is created lazily by build_agent; path should still resolve.
    assert mem == workspace_memory_dir()
