"""Tests for the ``jutul-agent upgrade`` subcommand."""

from __future__ import annotations

import subprocess

import pytest

from jutul_agent.interfaces.cli import upgrade as upgrade_cmd
from jutul_agent.update_check import InstallInfo


@pytest.fixture(autouse=True)
def _not_a_workspace(tmp_path, monkeypatch) -> None:
    """Run from an empty folder, so the environment step has nothing to update."""
    monkeypatch.setattr(upgrade_cmd, "workspace_root", lambda: tmp_path)


def _args(**overrides):
    args = upgrade_cmd.build_parser().parse_args([])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_check_reports_newer(monkeypatch, capsys) -> None:
    monkeypatch.setattr(upgrade_cmd, "install_info", lambda: InstallInfo("registry"))
    monkeypatch.setattr(upgrade_cmd, "refresh_cache", lambda force=False: "9.9.9")

    code = upgrade_cmd.run(_args(check=True))
    out = capsys.readouterr().out
    assert code == 0
    assert "Latest:    9.9.9" in out
    assert "newer version is available" in out


def test_check_reports_up_to_date(monkeypatch, capsys) -> None:
    monkeypatch.setattr(upgrade_cmd, "install_info", lambda: InstallInfo("registry"))
    monkeypatch.setattr(upgrade_cmd, "refresh_cache", lambda force=False: "0.0.0")

    code = upgrade_cmd.run(_args(check=True))
    assert code == 0
    assert "latest version" in capsys.readouterr().out


def test_editable_prints_git_pull(monkeypatch, capsys) -> None:
    from pathlib import Path

    monkeypatch.setattr(upgrade_cmd, "install_info", lambda: InstallInfo("editable", Path("/repo")))
    monkeypatch.setattr(upgrade_cmd, "refresh_cache", lambda force=False: "9.9.9")

    code = upgrade_cmd.run(_args(check=False))
    out = capsys.readouterr().out
    assert code == 0
    assert "git pull && uv sync" in out


def test_registry_runs_uv_tool_upgrade(monkeypatch, capsys) -> None:
    monkeypatch.setattr(upgrade_cmd, "install_info", lambda: InstallInfo("registry"))
    monkeypatch.setattr(upgrade_cmd, "refresh_cache", lambda force=False: "9.9.9")
    monkeypatch.setattr(upgrade_cmd.shutil, "which", lambda _name: "/usr/bin/uv")
    monkeypatch.setattr(upgrade_cmd.os, "name", "posix")  # exercise the inline path

    calls: list[list[str]] = []

    def _fake_run(argv, check=False):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(upgrade_cmd.subprocess, "run", _fake_run)

    code = upgrade_cmd.run(_args(check=False))
    assert code == 0
    # The new install, not this process, updates the environment: only it has
    # the new simulator templates.
    assert calls[0] == ["uv", "tool", "upgrade", "jutul-agent"]
    assert calls[1][1:3] == ["upgrade", "--env-only"]


def test_registry_windows_runs_detached(monkeypatch, capsys) -> None:
    # On Windows the running launcher is locked, so uv must run in a separate
    # process while jutul-agent exits — not inline (which would fail the copy).
    monkeypatch.setattr(upgrade_cmd, "install_info", lambda: InstallInfo("registry"))
    monkeypatch.setattr(upgrade_cmd, "refresh_cache", lambda force=False: "9.9.9")
    monkeypatch.setattr(upgrade_cmd.shutil, "which", lambda _name: "C:/uv.exe")
    monkeypatch.setattr(upgrade_cmd.os, "name", "nt")

    popened: list[list[str]] = []

    def _fake_popen(argv, creationflags=0, close_fds=True):
        popened.append(argv)
        return object()

    def _no_run(*a, **k):
        raise AssertionError("must not run uv inline on Windows (locked exe)")

    monkeypatch.setattr(upgrade_cmd.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(upgrade_cmd.subprocess, "run", _no_run)

    code = upgrade_cmd.run(_args(check=False))
    assert code == 0
    assert popened == [["uv", "tool", "upgrade", "jutul-agent"]]
    assert "new window" in capsys.readouterr().out


def test_registry_errors_without_uv(monkeypatch, capsys) -> None:
    monkeypatch.setattr(upgrade_cmd, "install_info", lambda: InstallInfo("registry"))
    monkeypatch.setattr(upgrade_cmd, "refresh_cache", lambda force=False: "9.9.9")
    monkeypatch.setattr(upgrade_cmd.shutil, "which", lambda _name: None)

    code = upgrade_cmd.run(_args(check=False))
    assert code == 1
    assert "uv` is not on PATH" in capsys.readouterr().err


def test_unknown_install_still_updates_the_environment(monkeypatch, capsys) -> None:
    monkeypatch.setattr(upgrade_cmd, "install_info", lambda: InstallInfo("unknown"))
    monkeypatch.setattr(upgrade_cmd, "refresh_cache", lambda force=False: "9.9.9")
    steps = []
    monkeypatch.setattr(upgrade_cmd, "update_workspace_env", lambda ws: steps.append(ws) or 0)

    code = upgrade_cmd.run(_args(check=False))
    assert code == 0
    assert "isn't installed as a managed package" in capsys.readouterr().err
    assert len(steps) == 1


def test_up_to_date_install_updates_the_environment(monkeypatch, capsys) -> None:
    monkeypatch.setattr(upgrade_cmd, "install_info", lambda: InstallInfo("registry"))
    monkeypatch.setattr(upgrade_cmd, "refresh_cache", lambda force=False: "0.0.0")
    steps = []
    monkeypatch.setattr(upgrade_cmd, "update_workspace_env", lambda ws: steps.append(ws) or 0)

    assert upgrade_cmd.run(_args(check=False)) == 0
    assert "latest version" in capsys.readouterr().out
    assert len(steps) == 1


def test_env_only_skips_the_install(monkeypatch, capsys) -> None:
    monkeypatch.setattr(upgrade_cmd, "refresh_cache", lambda **_k: pytest.fail("no network"))
    assert upgrade_cmd.run(_args(env_only=True)) == 0
    assert "nothing else to update" in capsys.readouterr().out


def _fake_status(monkeypatch, outdated):
    adapter = type("A", (), {"display_name": "JutulDarcy"})()
    monkeypatch.setattr(
        upgrade_cmd,
        "workspace_env_status",
        lambda ws: (adapter, ws / "env", "jutuldarcy", outdated),
    )


def test_env_step_updates_an_outdated_environment(monkeypatch, capsys) -> None:
    _fake_status(monkeypatch, [("JutulDarcy", "0.3.12", "0.4")])
    upgrades = []
    monkeypatch.setattr(
        upgrade_cmd, "upgrade_workspace", lambda *a, sysimage: upgrades.append(sysimage) or True
    )

    assert upgrade_cmd.run(_args(env_only=True)) == 0
    assert "has JutulDarcy 0.3.12; updating it to JutulDarcy 0.4" in capsys.readouterr().out
    assert upgrades == [False]  # this folder does not use a system image


def test_env_step_leaves_a_current_environment(monkeypatch, capsys) -> None:
    _fake_status(monkeypatch, [])
    monkeypatch.setattr(upgrade_cmd, "upgrade_workspace", lambda *a, **k: pytest.fail("no update"))
    assert upgrade_cmd.run(_args(env_only=True)) == 0
    assert "up to date" in capsys.readouterr().out


def test_check_reports_an_outdated_environment(monkeypatch, capsys) -> None:
    monkeypatch.setattr(upgrade_cmd, "install_info", lambda: InstallInfo("registry"))
    monkeypatch.setattr(upgrade_cmd, "refresh_cache", lambda force=False: "0.0.0")
    _fake_status(monkeypatch, [("JutulDarcy", "0.3.12", "0.4")])
    assert upgrade_cmd.run(_args(check=True)) == 0
    out = capsys.readouterr().out
    assert "has JutulDarcy 0.3.12; this version needs JutulDarcy 0.4" in out


@pytest.mark.parametrize(
    ("interactive", "answer", "rebuild", "still_on"),
    [(False, None, True, True), (True, "", True, True), (True, "n", False, False)],
)
def test_a_system_image_is_rebuilt_or_switched_off(
    tmp_path, monkeypatch, interactive, answer, rebuild, still_on
) -> None:
    """An update makes the image stale, so the user picks: rebuild it or stop using it."""
    from jutul_agent.workspace import (
        WorkspaceConfig,
        load_workspace_config,
        write_workspace_config,
    )

    write_workspace_config(
        WorkspaceConfig(simulator="jutuldarcy", sysimage=True), workspace=tmp_path
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: answer)

    assert upgrade_cmd.decide_sysimage(tmp_path, interactive=interactive) is rebuild
    assert load_workspace_config(tmp_path).sysimage is still_on
