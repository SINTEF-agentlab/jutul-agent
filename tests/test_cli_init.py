"""Tests for the init subcommand."""

from __future__ import annotations

from pathlib import Path

import pytest

from jutul_agent.interfaces.cli import main
from jutul_agent.workspace import workspace_config_path, workspace_julia_env


@pytest.fixture
def _julia_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend Julia 1.12 is installed so init's ``require_julia()`` check passes.

    These tests exercise the bootstrap (template copy + config), not the Julia
    toolchain, so they must not depend on a real Julia being on PATH (the cross-OS
    CI lane has none).
    """
    import shutil
    import subprocess

    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/julia" if name == "julia" else None)

    class _Version:
        returncode = 0
        stdout = "julia version 1.12.0\n"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Version())


def test_init_bootstraps_workspace_and_writes_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], _julia_on_path: None
) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    state = tmp_path / "state"

    code = main(
        [
            "init",
            "--sim",
            "jutuldarcy",
            "--no-precompile",  # bootstrap only; precompile is exercised separately
            "--workspace",
            str(ws),
            "--state-home",
            str(state),
        ]
    )
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert workspace_config_path(ws).exists()
    assert (workspace_julia_env(ws) / "Project.toml").exists()
    assert "Workspace ready" in captured.out
    assert "skipped (--no-precompile)" in captured.out


def test_setup_alias_bootstraps_workspace(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], _julia_on_path: None
) -> None:
    ws = tmp_path / "ws2"
    ws.mkdir()

    code = main(["setup", "--sim", "jutuldarcy", "--no-precompile", "--workspace", str(ws)])
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert (workspace_julia_env(ws) / "Project.toml").exists()


def test_instantiate_alias_runs_precompile_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    ws = tmp_path / "ws3"
    ws.mkdir()
    captured_cmds: list[list[str]] = []

    class _Result:
        returncode = 0
        # init now runs `require_julia()` first, which parses `julia --version`.
        stdout = "julia version 1.12.0\n"

    def _fake_run(argv, check=False, **kwargs):
        captured_cmds.append(argv)
        return _Result()

    import jutul_agent.simulators.env_setup as env_setup

    monkeypatch.setattr(env_setup.shutil, "which", lambda _: "/usr/bin/julia")
    monkeypatch.setattr(env_setup.subprocess, "run", _fake_run)

    code = main(
        [
            "init",
            "--sim",
            "jutuldarcy",
            "--instantiate",
            "--workspace",
            str(ws),
        ]
    )
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert "precompile:    done" in captured.out
    assert len(captured_cmds) >= 2
    assert any("Pkg.instantiate()" in cmd[-1] for cmd in captured_cmds)


def test_init_precompiles_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # No flags: the env (plotting backends included) is baked so the first
    # session in any interface is fast out of the box.
    ws = tmp_path / "ws-default"
    ws.mkdir()

    class _Result:
        returncode = 0
        stdout = "julia version 1.12.0\n"

    import jutul_agent.simulators.env_setup as env_setup

    monkeypatch.setattr(env_setup.shutil, "which", lambda _: "/usr/bin/julia")
    monkeypatch.setattr(env_setup.subprocess, "run", lambda *a, **k: _Result())

    code = main(["init", "--sim", "jutuldarcy", "--workspace", str(ws)])
    out = capsys.readouterr().out
    assert code == 0
    assert "precompile:    done" in out


def test_init_no_precompile_skips_the_bake(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    ws = tmp_path / "ws-skip"
    ws.mkdir()

    import jutul_agent.simulators.env_setup as env_setup

    monkeypatch.setattr(env_setup.shutil, "which", lambda _: "/usr/bin/julia")

    class _Version:
        returncode = 0
        stdout = "julia version 1.12.0\n"

    monkeypatch.setattr(env_setup.subprocess, "run", lambda *a, **k: _Version())

    code = main(["init", "--sim", "jutuldarcy", "--no-precompile", "--workspace", str(ws)])
    out = capsys.readouterr().out
    assert code == 0
    assert "skipped (--no-precompile)" in out
