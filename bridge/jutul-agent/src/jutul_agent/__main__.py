"""The last jutul-agent: install Makker as its own uv tool, and start it.

Earlier versions update themselves with ``uv tool upgrade jutul-agent`` and then
call ``jutul-agent`` or ``python -m jutul_agent`` again, with the arguments they
were started with. Both land here. So does anyone still typing ``jutul-agent``.

Whatever happens, Makker runs: if it cannot be given its own install (no uv on
the PATH, no network), the copy this package depends on is started in place, and
the step is tried again next time.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import MutableMapping
from pathlib import Path

NOTICE = "jutul-agent is now Makker. Start it with `makker` from now on."
OLD_SETTING_PREFIX = "JUTUL_AGENT_"
# Tells Makker it was started through this package, which it would otherwise
# tidy away as an install nobody uses any more.
VIA_OLD_COMMAND = "MAKKER_VIA_JUTUL_AGENT"
WINDOWS = os.name == "nt"


def translate_settings(environ: MutableMapping[str, str]) -> None:
    """Give every ``JUTUL_AGENT_*`` setting its ``MAKKER_*`` name, unless that is set."""
    for name, value in list(environ.items()):
        if name.startswith(OLD_SETTING_PREFIX):
            environ.setdefault("MAKKER_" + name[len(OLD_SETTING_PREFIX) :], value)


def _uv(*args: str, show: bool = False) -> subprocess.CompletedProcess[str] | None:
    uv = shutil.which("uv")
    if uv is None:
        return None
    try:
        if show:  # progress belongs on stderr; stdout may be what a caller reads
            return subprocess.run([uv, *args], stdout=sys.stderr, text=True, timeout=1800)
        return subprocess.run([uv, *args], capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None


def installed_makker() -> str | None:
    """The ``makker`` command of Makker's own uv tool, if that is installed."""
    name = "makker.exe" if WINDOWS else "makker"
    folders = []
    bin_dir = _uv("tool", "dir", "--bin")
    if bin_dir is not None and bin_dir.returncode == 0 and bin_dir.stdout.strip():
        folders.append(Path(bin_dir.stdout.strip()))
    folders.append(Path(sys.argv[0]).absolute().parent)  # beside the launcher we came through
    for folder in folders:
        # This package's own environment holds a `makker` too, the dependency's.
        # It is not an install of its own, so it does not count.
        if folder != Path(sys.prefix) / ("Scripts" if WINDOWS else "bin"):
            candidate = folder / name
            if candidate.is_file():
                return str(candidate)
    return None


def own_install() -> str | None:
    """Makker as its own uv tool, installed now if it is not there yet."""
    if not (Path(sys.prefix) / "uv-receipt.toml").is_file():
        return None  # not a uv tool, so there is no tool to step out of
    command = installed_makker()
    if command is not None:
        return command
    python = f"{sys.version_info.major}.{sys.version_info.minor}"
    done = _uv("tool", "install", "makker", "--python", python, show=True)
    if done is None or done.returncode != 0:
        return None
    return installed_makker()


def run(argv: list[str]) -> int:
    if not WINDOWS:
        os.execv(argv[0], argv)
    # Windows cannot replace a process, so Makker runs as a child and its exit
    # code is passed on. Ctrl+C reaches the child directly; wait for it to finish.
    child = subprocess.Popen(argv)
    while True:
        try:
            return child.wait()
        except KeyboardInterrupt:
            continue


def main() -> int:
    translate_settings(os.environ)
    os.environ[VIA_OLD_COMMAND] = "1"
    command = own_install()
    if command is not None:
        print(NOTICE, file=sys.stderr, flush=True)
        return run([command, *sys.argv[1:]])
    return run([sys.executable, "-m", "makker", *sys.argv[1:]])


if __name__ == "__main__":
    sys.exit(main())
