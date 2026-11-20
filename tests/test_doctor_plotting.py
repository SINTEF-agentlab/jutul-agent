"""Tests for `makker doctor`'s plotting-display check.

It must only ever WARN (never FAIL): plotting is optional, so a headless box
without xvfb is usable for simulation. The remediation differs between "xvfb not
installed" and "xvfb opted out", so both paths are covered.
"""

from __future__ import annotations

import pytest

from makker.interfaces.cli.doctor import PASS, WARN, _check_plotting_display, _Report

_MOD = "makker.interfaces.cli.doctor"


def test_plotting_check_passes_with_display(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(f"{_MOD}.plotting_display_available", lambda: True)
    monkeypatch.setattr(f"{_MOD}.has_display", lambda: True)
    report = _Report()
    _check_plotting_display(report)
    assert report.worst == PASS
    assert "Plotting display" in capsys.readouterr().out


def test_plotting_check_warns_when_xvfb_missing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(f"{_MOD}.plotting_display_available", lambda: False)
    monkeypatch.setattr(f"{_MOD}.xvfb_opted_out", lambda: False)
    report = _Report()
    _check_plotting_display(report)
    assert report.worst == WARN
    out = capsys.readouterr().out
    assert "xvfb-run not found" in out
    assert "apt-get install -y xvfb" in out


def test_plotting_check_warns_when_opted_out(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(f"{_MOD}.plotting_display_available", lambda: False)
    monkeypatch.setattr(f"{_MOD}.xvfb_opted_out", lambda: True)
    report = _Report()
    _check_plotting_display(report)
    assert report.worst == WARN
    assert "MAKKER_NO_XVFB" in capsys.readouterr().out
