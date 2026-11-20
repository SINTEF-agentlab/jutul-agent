"""The last jutul-agent release, which installs Makker and starts it (``bridge/``)."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

BRIDGE = Path(__file__).resolve().parents[1] / "bridge" / "jutul-agent"


@pytest.fixture(params=[False, True], ids=["posix", "windows"])
def bridge(request, monkeypatch):
    """The bridge, set to start Makker the way each kind of system does.

    Both ways run on every system: only the bridge's own switch is set, never
    ``os.name``, which the standard library reads too.
    """
    spec = importlib.util.spec_from_file_location(
        "jutul_agent_bridge", BRIDGE / "src" / "jutul_agent" / "__main__.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "WINDOWS", request.param)
    return module


class Machine:
    """A stand-in for uv and the tools folder: records what the bridge runs."""

    def __init__(self, root: Path, monkeypatch, bridge, *, uv_tool: bool, has_uv: bool = True):
        self.bin = root / "bin"
        self.bin.mkdir()
        self.tool = root / "tools" / "jutul-agent"
        self.tool.mkdir(parents=True)
        if uv_tool:
            (self.tool / "uv-receipt.toml").write_text("[tool]\n", encoding="utf-8")
        self.command = "makker.exe" if bridge.WINDOWS else "makker"
        self.own_bin = self.tool / ("Scripts" if bridge.WINDOWS else "bin")
        self.install_works = True
        self.uv_calls: list[list[str]] = []
        self.started: list[str] | None = None
        monkeypatch.setattr(sys, "prefix", str(self.tool))
        monkeypatch.setattr(sys, "argv", [str(self.bin / "jutul-agent"), "--port", "8800"])
        monkeypatch.setattr(bridge.shutil, "which", lambda name: "/usr/bin/uv" if has_uv else None)
        monkeypatch.setattr(bridge.subprocess, "run", self._uv)
        monkeypatch.setattr(bridge.os, "execv", self._exec)
        monkeypatch.setattr(bridge.subprocess, "Popen", self._child)
        monkeypatch.setattr(bridge.os, "environ", {"JUTUL_AGENT_RESTARTED_AFTER_UPDATE": "1"})
        self.environ = bridge.os.environ

    def _uv(self, argv, **kwargs):
        self.uv_calls.append(argv[1:])
        if argv[1:4] == ["tool", "dir", "--bin"]:
            return subprocess.CompletedProcess(argv, 0, f"{self.bin}\n", "")
        if argv[1:3] == ["tool", "install"] and self.install_works:
            (self.bin / self.command).write_text("#!/bin/sh\n", encoding="utf-8")
            return subprocess.CompletedProcess(argv, 0, "", "")
        return subprocess.CompletedProcess(argv, 1, "", "")

    def _exec(self, path, argv):
        self.started = argv
        raise SystemExit(0)

    def _child(self, argv, **kwargs):
        self.started = argv
        return type("Child", (), {"wait": lambda self: 3})()

    def start(self, bridge) -> list[str]:
        if bridge.WINDOWS:
            assert bridge.main() == 3  # Makker ran as a child; its exit code is passed on
        else:
            with pytest.raises(SystemExit):  # Makker replaced the process
                bridge.main()
        assert self.started is not None
        return self.started


def test_the_first_start_installs_makker_as_its_own_tool_and_runs_it(
    tmp_path, monkeypatch, bridge, capsys
) -> None:
    machine = Machine(tmp_path, monkeypatch, bridge, uv_tool=True)

    started = machine.start(bridge)

    python = f"{sys.version_info.major}.{sys.version_info.minor}"
    assert ["tool", "install", "makker", "--python", python] in machine.uv_calls
    assert started == [str(machine.bin / machine.command), "--port", "8800"]
    assert "jutul-agent is now Makker" in capsys.readouterr().err
    # What the old version set for the restarted server reaches Makker under its name.
    assert machine.environ["MAKKER_RESTARTED_AFTER_UPDATE"] == "1"
    assert machine.environ[bridge.VIA_OLD_COMMAND] == "1"


def test_a_makker_that_is_already_installed_is_just_started(tmp_path, monkeypatch, bridge) -> None:
    machine = Machine(tmp_path, monkeypatch, bridge, uv_tool=True)
    (machine.bin / machine.command).write_text("#!/bin/sh\n", encoding="utf-8")

    started = machine.start(bridge)

    assert started[0] == str(machine.bin / machine.command)
    assert not [call for call in machine.uv_calls if call[:2] == ["tool", "install"]]


@pytest.mark.parametrize("why", ["install fails", "no uv", "not a uv tool"])
def test_makker_still_runs_in_place_when_it_cannot_get_its_own_install(
    tmp_path, monkeypatch, bridge, capsys, why
) -> None:
    machine = Machine(
        tmp_path, monkeypatch, bridge, uv_tool=why != "not a uv tool", has_uv=why != "no uv"
    )
    machine.install_works = False

    started = machine.start(bridge)

    assert started == [sys.executable, "-m", "makker", "--port", "8800"]
    assert "Start it with `makker`" not in capsys.readouterr().err  # there is no such command yet
    if why == "not a uv tool":
        assert machine.uv_calls == []


def test_the_dependency_inside_its_own_environment_is_not_mistaken_for_an_install(
    tmp_path, monkeypatch, bridge
) -> None:
    machine = Machine(tmp_path, monkeypatch, bridge, uv_tool=True)
    machine.install_works = False
    machine.own_bin.mkdir()
    (machine.own_bin / machine.command).write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [str(machine.own_bin / "jutul-agent")])

    assert machine.start(bridge) == [sys.executable, "-m", "makker"]


def test_settings_keep_a_value_already_given_under_the_new_name(bridge) -> None:
    environ = {"JUTUL_AGENT_MODEL": "old", "MAKKER_MODEL": "new", "JUTUL_AGENT_NO_OPEN": "1"}
    bridge.translate_settings(environ)
    assert environ["MAKKER_MODEL"] == "new"
    assert environ["MAKKER_NO_OPEN"] == "1"


def test_the_bridge_depends_on_makker_alone_and_offers_only_the_old_command() -> None:
    import tomllib

    project = tomllib.loads((BRIDGE / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["name"] == "jutul-agent"
    assert len(project["dependencies"]) == 1 and project["dependencies"][0].startswith("makker>=")
    # A `makker` command here would block `uv tool install makker`.
    assert list(project["scripts"]) == ["jutul-agent"]
