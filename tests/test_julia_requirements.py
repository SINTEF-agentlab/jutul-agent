"""Tests for Julia toolchain requirement checks and startup error formatting."""

from __future__ import annotations

from pathlib import Path

import pytest

from makker.julia import requirements
from makker.juliakernel import JuliaStartupError


class _Proc:
    def __init__(self, stdout: str = "") -> None:
        self.stdout = stdout
        self.returncode = 0


# Synthetic versions derived from the supported floor so these tests never
# depend on whatever Julia happens to be installed on the machine running them.
_MIN_MAJOR, _MIN_MINOR = requirements.MIN_JULIA_VERSION
_SUPPORTED_VERSION = f"{_MIN_MAJOR}.{_MIN_MINOR}.0"  # exactly at the floor
_TOO_OLD_VERSION = f"{_MIN_MAJOR}.{_MIN_MINOR - 1}.0"  # one minor below the floor


def _fake_julia(monkeypatch: pytest.MonkeyPatch, version: str | None) -> None:
    """Pretend `julia` is (not) on PATH and reports ``version``."""

    monkeypatch.setattr(
        requirements.shutil, "which", lambda _: None if version is None else "julia"
    )
    if version is not None:
        monkeypatch.setattr(
            requirements.subprocess, "run", lambda *a, **kw: _Proc(f"julia version {version}\n")
        )


def test_check_julia_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_julia(monkeypatch, None)
    check = requirements.check_julia()
    assert not check.found
    assert not check.ok
    assert "PATH" in (check.error or "")


def test_check_julia_parses_arbitrary_version(monkeypatch: pytest.MonkeyPatch) -> None:
    # A made-up version string, just to confirm the regex parses major/minor/patch.
    _fake_julia(monkeypatch, "9.8.7")
    check = requirements.check_julia()
    assert check.found
    assert check.version == (9, 8, 7)
    assert check.version_str == "9.8.7"


def test_check_julia_accepts_supported_version(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_julia(monkeypatch, _SUPPORTED_VERSION)
    check = requirements.check_julia()
    assert check.version_ok
    assert check.ok


def test_check_julia_too_old(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_julia(monkeypatch, _TOO_OLD_VERSION)
    check = requirements.check_julia()
    assert check.found
    assert not check.version_ok
    assert not check.ok


def test_require_julia_raises_when_too_old(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_julia(monkeypatch, _TOO_OLD_VERSION)
    with pytest.raises(requirements.JuliaRequirementError, match="required"):
        requirements.require_julia()


def test_require_julia_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_julia(monkeypatch, None)
    with pytest.raises(requirements.JuliaRequirementError, match="PATH"):
        requirements.require_julia()


def test_startup_error_surfaces_julia_stderr() -> None:
    err = JuliaStartupError(
        "the Julia process exited before responding",
        julia_executable="julia",
        julia_project=Path("/tmp/ws/.makker/julia-env"),
        stderr_tail="ERROR: ArgumentError: Package Jutul not found in current path.",
        log_file=Path("/tmp/ws/julia-startup.log"),
    )
    text = str(err)
    # The real cause is front and center, not buried under a transport traceback.
    assert "Package Jutul not found" in text
    assert "julia project: " in text
    assert "julia-startup.log" in text
