"""Cross-platform best-effort file opener.

Opens a file or directory in the OS default application.  Failures are
swallowed silently — the caller should not depend on the file actually
opening (e.g. headless CI environments have no viewer).
"""

from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path


def open_path(path: Path) -> None:
    """Open *path* in the OS default application (non-blocking, best-effort).

    Set ``JUTUL_AGENT_NO_OPEN=1`` to suppress opening (e.g. in CI or tests).
    """
    if os.environ.get("JUTUL_AGENT_NO_OPEN"):
        return
    system = platform.system()
    try:
        if system == "Windows":
            os.startfile(path)  # type: ignore[attr-defined]
        elif system == "Darwin":
            _spawn(["open", str(path)])
        else:
            # On a headless Linux box there's no display server; xdg-open often
            # falls through to a viewer that prints "unable to open X server"
            # to our stderr. Skip rather than spew.
            from jutul_agent.display import has_display

            if not has_display():
                return
            _spawn(["xdg-open", str(path)])
    except Exception:
        pass


def _spawn(argv: list[str]) -> None:
    """Launch a detached viewer, discarding its stdout/stderr.

    The child's own diagnostics (e.g. a viewer failing to reach the display)
    must not leak into jutul-agent's output.
    """

    subprocess.Popen(
        argv,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
    )
