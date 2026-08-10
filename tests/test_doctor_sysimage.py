"""Tests for `jutul-agent doctor`'s system-image line.

The check is deliberately quiet for the workspaces that never build an image, so
the tests pin both halves: silence when there is nothing to say, and a FAIL that
names the cause when a folder is set to use an image it cannot.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from jutul_agent import sysimage
from jutul_agent.interfaces.cli import doctor
from jutul_agent.interfaces.cli.doctor import FAIL, PASS, _check_sysimage, _Report
from jutul_agent.workspace import WorkspaceConfig, write_workspace_config


def _install_image(ws: Path) -> Path:
    path = sysimage.sysimage_path(ws)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not a real system image")
    return path


def test_no_line_at_all_when_nothing_uses_an_image(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report = _Report()
    _check_sysimage(report, workspace, workspace, WorkspaceConfig())
    assert report.worst == PASS
    assert capsys.readouterr().out == ""


def test_an_image_the_folder_ignores_is_reported_but_not_a_problem(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _install_image(workspace)
    report = _Report()
    _check_sysimage(report, workspace, workspace, WorkspaceConfig())
    assert report.worst == PASS
    assert "not set to use it" in capsys.readouterr().out


def test_a_folder_that_wants_a_missing_image_fails_with_the_reason(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report = _Report()
    _check_sysimage(report, workspace, workspace, WorkspaceConfig(sysimage=True))
    assert report.worst == FAIL
    out = capsys.readouterr().out
    assert "no system image has been built" in out
    assert "jutul-agent sysimage build" in out


def test_a_usable_image_reports_when_it_was_built(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _install_image(workspace)
    sysimage.write_stamp(workspace, workspace, cpu_target="native", build_seconds=1.0)
    report = _Report()
    _check_sysimage(report, workspace, workspace, WorkspaceConfig(sysimage=True))
    assert report.worst == PASS
    assert "built 20" in capsys.readouterr().out


def test_doctor_actually_runs_the_check(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A check nobody calls reports nothing, and passes every test of itself."""
    write_workspace_config(
        WorkspaceConfig(simulator="jutuldarcy", sysimage=True), workspace=workspace
    )
    doctor.run(argparse.Namespace(sim=None, workspace=workspace, state_home=None))
    assert "System image" in capsys.readouterr().out
