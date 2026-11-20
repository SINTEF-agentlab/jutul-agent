"""Tests for display detection and the window-vs-offscreen decision."""

from __future__ import annotations

import pytest

from makker.display import (
    can_open_windows,
    has_display,
    managed_display,
    plotting_display_available,
    should_wrap_xvfb,
)


def test_one_shot_never_opens_windows() -> None:
    # A headless --prompt turn renders offscreen even with a display.
    assert can_open_windows(interactive_session=False, display=True) is False


def test_interactive_with_display_opens_windows() -> None:
    assert can_open_windows(interactive_session=True, display=True) is True


def test_interactive_without_display_renders_offscreen() -> None:
    # Interactive session but no display (e.g. SSH on Linux) -> offscreen.
    assert can_open_windows(interactive_session=True, display=False) is False


@pytest.mark.parametrize("system", ["Windows", "Darwin"])
def test_has_display_true_on_desktop_os(monkeypatch: pytest.MonkeyPatch, system: str) -> None:
    monkeypatch.setattr("makker.display.platform.system", lambda: system)
    assert has_display() is True


def test_has_display_linux_depends_on_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("makker.display.platform.system", lambda: "Linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    assert has_display() is False

    monkeypatch.setenv("DISPLAY", ":0")
    assert has_display() is True


def _headless_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    """Put the display module in a headless-Linux baseline (no display, no opt-out)."""
    monkeypatch.setattr("makker.display.platform.system", lambda: "Linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.delenv("MAKKER_NO_XVFB", raising=False)


def _set_xvfb_run(monkeypatch: pytest.MonkeyPatch, present: bool) -> None:
    monkeypatch.setattr(
        "makker.display.shutil.which",
        lambda name: "/usr/bin/xvfb-run" if (present and name == "xvfb-run") else None,
    )


def test_should_wrap_xvfb_only_on_headless_linux_with_xvfb(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _headless_linux(monkeypatch)
    _set_xvfb_run(monkeypatch, present=True)
    assert should_wrap_xvfb() is True

    # Missing xvfb-run, opt-out, a real display, or a non-Linux OS all suppress it.
    _set_xvfb_run(monkeypatch, present=False)
    assert should_wrap_xvfb() is False

    _set_xvfb_run(monkeypatch, present=True)
    monkeypatch.setenv("MAKKER_NO_XVFB", "1")
    assert should_wrap_xvfb() is False
    monkeypatch.delenv("MAKKER_NO_XVFB", raising=False)

    monkeypatch.setenv("DISPLAY", ":0")
    assert should_wrap_xvfb() is False
    monkeypatch.delenv("DISPLAY", raising=False)

    monkeypatch.setattr("makker.display.platform.system", lambda: "Darwin")
    assert should_wrap_xvfb() is False


def test_managed_display_raises_cleanly_when_xvfb_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The startup path treats a missing Xvfb as "no plotting here" (a caught
    # RuntimeError), not a hard crash; so failing fast and clearly matters.
    monkeypatch.setattr("makker.display.shutil.which", lambda _: None)
    with pytest.raises(RuntimeError, match="Xvfb"), managed_display():
        pass


def test_managed_display_closes_pipe_when_xvfb_fails_to_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # If Popen fails after the pipe is opened, neither pipe fd should leak.
    import os

    if not os.path.isdir("/proc/self/fd"):
        pytest.skip("fd-count check needs /proc (Linux)")
    monkeypatch.setattr("makker.display.xvfb_available", lambda: True)

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise PermissionError("fork denied")

    monkeypatch.setattr("makker.display.subprocess.Popen", _boom)

    before = len(os.listdir("/proc/self/fd"))
    with pytest.raises(PermissionError), managed_display():
        pass
    assert len(os.listdir("/proc/self/fd")) == before  # both pipe fds were closed


def test_plotting_display_available_on_desktop_os(monkeypatch: pytest.MonkeyPatch) -> None:
    # Desktop OSes always have a display, so plotting is available regardless of xvfb.
    monkeypatch.setattr("makker.display.platform.system", lambda: "Darwin")
    _set_xvfb_run(monkeypatch, present=False)
    assert plotting_display_available() is True


def test_plotting_display_available_headless_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    _headless_linux(monkeypatch)
    # Headless with xvfb -> the wrapped worker provides a display.
    _set_xvfb_run(monkeypatch, present=True)
    assert plotting_display_available() is True
    # Headless without xvfb -> no display anywhere, plotting unavailable.
    _set_xvfb_run(monkeypatch, present=False)
    assert plotting_display_available() is False
    # A real display makes it available even without xvfb.
    monkeypatch.setenv("DISPLAY", ":0")
    assert plotting_display_available() is True
