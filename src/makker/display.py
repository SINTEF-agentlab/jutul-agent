"""Whether a session can show the user a live Makie window.

The plotting tool always renders with GLMakie. The only environment-dependent
choice is whether to open an on-screen window (a human is watching an interactive
session on a machine with a display) or render offscreen to a file (a headless or
one-shot run). Headless Linux still renders: the caller starts a private virtual
X server (Xvfb) via :func:`managed_display` and points the Julia process at it
through ``DISPLAY``, so GLMakie draws offscreen there.

We launch ``Xvfb`` directly rather than via ``xvfb-run`` for the Julia *session*:
``xvfb-run`` runs its command with ``2>&1`` (stderr folded into stdout), which
would blur the kernel's startup diagnostics (and a long-lived session shouldn't
hinge on a wrapper script anyway). One-shot ``Pkg`` precompile subprocesses
still use ``xvfb-run``.
"""

from __future__ import annotations

import contextlib
import os
import platform
import select
import shutil
import subprocess
import time
from collections.abc import Iterator


def has_display() -> bool:
    """Best-effort check for a human-visible display server.

    Windows and macOS desktop sessions always have one. Linux and BSD need an X
    or Wayland display; a headless box (SSH, CI) has neither.
    """
    if platform.system() in ("Windows", "Darwin"):
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def can_open_windows(*, interactive_session: bool, display: bool | None = None) -> bool:
    """True when plot_julia may open a live window for the user.

    Requires both an interactive (TUI) session and a display. A one-shot
    ``--prompt`` run has nobody watching, and a headless box has nowhere to draw,
    so both render offscreen to a file instead.

    Args:
        interactive_session: True for the live TUI, False for a one-shot ``--prompt``.
        display: Override for display detection (tests); defaults to ``has_display()``.
    """
    if not interactive_session:
        return False
    return has_display() if display is None else display


def xvfb_run_available() -> bool:
    """True if the ``xvfb-run`` wrapper script is on PATH."""
    return shutil.which("xvfb-run") is not None


def xvfb_available() -> bool:
    """True if the ``Xvfb`` server binary is on PATH (ships with ``xvfb-run``)."""
    return shutil.which("Xvfb") is not None


@contextlib.contextmanager
def managed_display(*, screen: str = "1280x1024x24", timeout: float = 30.0) -> Iterator[str]:
    """Start a private ``Xvfb`` server and yield its ``DISPLAY`` (e.g. ``":7"``).

    Gives headless GLMakie an OpenGL context without ``xvfb-run``; we launch
    ``Xvfb`` directly so the Julia process keeps its stdout and stderr on the
    separate pipes the kernel protocol needs (``xvfb-run`` runs its command with
    ``2>&1``, merging them, which deadlocks the kernel's per-eval stderr sentinel).

    Uses ``Xvfb -displayfd`` so the server picks a free display number itself and
    reports it back; no ``:N`` lock-file guessing or races. The server is torn
    down on exit. Raises ``RuntimeError`` if ``Xvfb`` is missing or never reports a
    display; callers treat that as "no plotting here" rather than a hard failure.
    """

    if not xvfb_available():
        raise RuntimeError("`Xvfb` is not on PATH")
    read_fd, write_fd = os.pipe()
    try:
        proc = subprocess.Popen(
            [
                "Xvfb",
                "-displayfd",
                str(write_fd),
                "-screen",
                "0",
                screen,
                "-nolisten",
                "tcp",
                "-ac",
            ],
            pass_fds=(write_fd,),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except BaseException:
        os.close(read_fd)  # Popen failed, so the read side never reaches its finally below
        raise
    finally:
        os.close(write_fd)

    try:
        number = _read_display_number(read_fd, timeout)
    except BaseException:
        _terminate(proc)
        raise
    finally:
        os.close(read_fd)

    display = f":{number}"
    try:
        yield display
    finally:
        _terminate(proc)


def _read_display_number(read_fd: int, timeout: float) -> str:
    """Read the display number Xvfb writes to ``-displayfd`` once it is ready."""

    deadline = time.monotonic() + timeout
    buf = b""
    while not buf.endswith(b"\n"):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError("Xvfb did not report a display number in time")
        if not select.select([read_fd], [], [], remaining)[0]:
            continue
        chunk = os.read(read_fd, 64)
        if not chunk:
            break
        buf += chunk
    number = buf.decode("ascii", "replace").strip()
    if not number:
        raise RuntimeError("Xvfb exited before reporting a display number")
    return number


def _terminate(proc: subprocess.Popen) -> None:
    proc.terminate()
    with contextlib.suppress(Exception):
        proc.wait(timeout=5)


def xvfb_opted_out() -> bool:
    """True if the user disabled the headless xvfb wrap (``MAKKER_NO_XVFB``)."""
    return bool(os.environ.get("MAKKER_NO_XVFB"))


def should_wrap_xvfb() -> bool:
    """True on headless Linux where a virtual display (Xvfb) should back GLMakie.

    GLMakie needs an X/Wayland display for its OpenGL context; it has no built-in
    OSMesa/EGL switch. So on a Linux box with no ``DISPLAY`` we give the Julia
    process a private Xvfb display (the session via :func:`managed_display`, which
    the process and any ``addprocs`` workers inherit through ``DISPLAY``; one-shot
    ``Pkg`` precompiles via ``xvfb-run``). The same approach JutulDarcy uses in its
    own CI. Set ``MAKKER_NO_XVFB=1`` to opt out; plotting then has no display
    and plot_julia reports a clear error.

    The single source of truth for the headless-display decision: startup wiring
    uses it to launch the Xvfb display, and ``doctor`` / startup use the related
    ``plotting_display_available`` to warn when plotting won't work here.
    ``xvfb-run`` ships the ``Xvfb`` binary, so its presence implies both.
    """

    if platform.system() != "Linux":
        return False
    if has_display():
        return False
    if xvfb_opted_out():
        return False
    return xvfb_run_available()


def plotting_display_available() -> bool:
    """Whether GLMakie will have a display to render against.

    A real X/Wayland display (or any desktop macOS/Windows session) works
    directly; headless Linux relies on the xvfb-wrapped worker. When this is
    False, ``plot_julia`` cannot render and reports a clear error, but the rest
    of the agent (simulate, eval, file tools) still works.
    """
    return has_display() or should_wrap_xvfb()
