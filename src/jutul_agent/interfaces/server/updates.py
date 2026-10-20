"""Updating jutul-agent and this folder's Julia environment from the web UI.

The browser can do what ``jutul-agent upgrade`` does in a terminal:

1. Upgrade the install with ``uv tool upgrade jutul-agent``.
2. Bring this folder's Julia environment up to what the installed version supports,
   by running ``jutul-agent upgrade --env-only`` on the interpreter this server runs
   on. After step 1 that interpreter imports the new version, so its simulator
   templates decide, as they do in the terminal.
3. Restart the server into the new version when step 1 changed the install.

Step 1 needs a uv tool install, and is not offered on Windows, where a running
program cannot replace its own files. Updates are only accepted from a page served
by this computer, never from another site in the same browser. The commands are
fixed: a request only says whether to do step 1 and what to do with a system image.
One update runs at a time, the live sessions are closed first (their chats stay on
disk), and no session starts until it is done.
"""

from __future__ import annotations

import ipaddress
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn
from urllib.parse import urlsplit

from jutul_agent.interfaces.cli.upgrade import workspace_env_status
from jutul_agent.update_check import install_info, is_enabled, pending_update
from jutul_agent.workspace import load_workspace_config

if TYPE_CHECKING:
    from jutul_agent.interfaces.server.manager import SessionManager

# Set for the restarted server, so it starts even if the update left something to
# fix: the page that asked for the update is waiting for it, and says what is wrong.
RESTARTED_ENV_VAR = "JUTUL_AGENT_RESTARTED_AFTER_UPDATE"

# Identifies this server process, so a page can tell when a restart has happened.
BOOT_ID = uuid.uuid4().hex

_MAX_LINES = 5000
# The page polls about once a second; this lets it show the last lines before the
# server goes away.
_RESTART_DELAY_SECONDS = 2.0

# Why this install can't upgrade itself from the browser, as shown in the dialog.
_INSTALL_BLOCKERS = {
    "not-local": (
        "Updating from the browser only works when jutul-agent runs on this computer. "
        "Run `jutul-agent upgrade` where it runs, then start it again."
    ),
    "windows": (
        "On Windows, jutul-agent can't replace its own files while it runs. Stop it "
        "(Ctrl+C in its terminal), run `jutul-agent upgrade`, then start it again."
    ),
    "dev-checkout": (
        "This is a development checkout. Update it with `git pull && uv sync`, then "
        "start jutul-agent again."
    ),
    "not-uv-tool": (
        "jutul-agent wasn't installed with `uv tool install`, so it can't update "
        "itself. Upgrade it the way you installed it, then start it again."
    ),
    "no-uv": (
        "uv isn't on the PATH jutul-agent was started with. Run `uv tool upgrade "
        "jutul-agent` in a terminal, then start jutul-agent again."
    ),
    "no-restart": (
        "This server can't restart itself. Run `jutul-agent upgrade` in a terminal, "
        "then start it again."
    ),
}


class UpdateRefused(Exception):
    """The update can't start now; the message says why."""


def is_loopback_host(host: str) -> bool:
    """Whether a bind address or URL host name only reaches this computer."""
    host = host.strip("[]")
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def request_is_local(headers: Mapping[str, str]) -> bool:
    """Whether a request comes from this server's own page on this computer.

    A browser sends ``Origin`` with every POST. Requiring it to name the very host
    the request was sent to shuts out other sites open in the same browser, and
    requiring that host to be a loopback name shuts out a site that points its own
    domain at this computer.
    """
    host = headers.get("host", "")
    origin = headers.get("origin")
    if not host or not origin:
        return False
    hostname = urlsplit(f"//{host}").hostname or ""
    return is_loopback_host(hostname) and urlsplit(origin).netloc == host


def env_status(workspace: Path) -> dict[str, Any] | None:
    """This folder's simulator packages that the installed version needs updated.

    ``None`` when there is no managed environment, or it can't be read.
    """
    try:
        status = workspace_env_status(workspace)
    except (OSError, ValueError):  # an unreadable or malformed manifest
        return None
    if status is None:
        return None
    _, _, _, outdated = status
    return {
        "outdated": [
            {"package": package, "version": found, "required": spec}
            for package, found, spec in outdated
        ],
        "sysimage": bool(load_workspace_config(workspace).sysimage),
    }


def _on_windows() -> bool:
    return os.name == "nt"


def _installed_version() -> str | None:
    try:
        return version("jutul-agent")
    except PackageNotFoundError:
        return None


def _short(text: str | None) -> str | None:
    return text.split("+")[0] if text else text


@dataclass
class UpdateJob:
    install: bool
    state: str = "running"  # running, restarting, done, failed
    error: str | None = None
    lines: list[str] = field(default_factory=list)
    # Absolute index of ``lines[0]``: the oldest lines are dropped past _MAX_LINES.
    first: int = 0

    def add(self, line: str) -> None:
        self.lines.append(line)
        if len(self.lines) > _MAX_LINES:
            drop = len(self.lines) - _MAX_LINES
            del self.lines[:drop]
            self.first += drop

    def snapshot(self, since: int) -> dict[str, Any]:
        start = max(since, self.first) - self.first
        return {
            "state": self.state,
            "install": self.install,
            "error": self.error,
            "lines": self.lines[start:],
            "next": self.first + len(self.lines),
        }


class Updater:
    """Runs updates for one server; see the module docstring."""

    def __init__(self, workspace: Path, *, local: bool) -> None:
        self._workspace = workspace
        # Served on this computer only; anything else never runs an update.
        self.local = local
        self._stop_server: Callable[[], None] | None = None
        self._job: UpdateJob | None = None
        self._lock = threading.Lock()
        self._version_at_start = _installed_version()
        self.restart_requested = False

    def bind_server(self, stop: Callable[[], None]) -> None:
        """How to stop the server so it can restart into a new version."""
        self._stop_server = stop

    @property
    def running(self) -> bool:
        return self._job is not None and self._job.state in ("running", "restarting")

    def install_blocker(self) -> str | None:
        """Why the browser can't upgrade this install, or ``None`` if it can."""
        if not self.local:
            return "not-local"
        if _on_windows():
            return "windows"
        if install_info().method == "editable":
            return "dev-checkout"
        # uv keeps a receipt in the root of every environment it installs a tool into.
        if not (Path(sys.prefix) / "uv-receipt.toml").is_file():
            return "not-uv-tool"
        if shutil.which("uv") is None:
            return "no-uv"
        if self._stop_server is None:
            return "no-restart"
        return None

    def status(self, since: int = 0) -> dict[str, Any]:
        blocker = self.install_blocker()
        job = self._job
        return {
            "current": _short(self._version_at_start),
            "latest": pending_update() if is_enabled() else None,
            "local": self.local,
            "can_install": blocker is None,
            "install_note": _INSTALL_BLOCKERS.get(blocker) if blocker else None,
            # Not read during an update: after `uv tool upgrade` this process runs
            # old code over new files, so it stays away from anything it hasn't loaded.
            "env": None if self.running else env_status(self._workspace),
            "job": job.snapshot(since) if job is not None else None,
        }

    def env_update_needed(self) -> bool:
        if self.running:
            return False
        status = env_status(self._workspace)
        return bool(status and status["outdated"])

    async def start(self, manager: SessionManager, *, install: bool, sysimage: str | None) -> None:
        if not self.local:
            raise UpdateRefused(_INSTALL_BLOCKERS["not-local"])
        blocker = self.install_blocker() if install else None
        if blocker is not None:
            raise UpdateRefused(_INSTALL_BLOCKERS[blocker])
        with self._lock:
            if self.running:
                raise UpdateRefused("An update is already running.")
            if manager.any_busy():
                raise UpdateRefused(
                    "A session is starting or working. Stop it or let it finish, then update."
                )
            job = self._job = UpdateJob(install=install)
        await manager.aclose()
        threading.Thread(
            target=self._run, args=(job, sysimage), name="jutul-agent-update", daemon=True
        ).start()

    def _run(self, job: UpdateJob, sysimage: str | None) -> None:
        try:
            if job.install and self._step(job, ["uv", "tool", "upgrade", "jutul-agent"]):
                job.error = "jutul-agent could not be upgraded. See the output above."
                job.state = "failed"
                return
            env_step = [
                sys.executable,
                "-m",
                "jutul_agent",
                "upgrade",
                "--env-only",
                "--workspace",
                str(self._workspace),
            ]
            if sysimage:
                env_step += ["--sysimage", sysimage]
            if self._step(job, env_step, shown="jutul-agent upgrade --env-only"):
                job.error = "The Julia environment could not be updated. See the output above."
        except Exception as exc:
            # Never leave the job looking like it still runs: that blocks sessions.
            job.error = f"The update could not run: {exc}"
        if job.install and _installed_version() != self._version_at_start:
            job.state = "restarting"
            self.restart_requested = True
            time.sleep(_RESTART_DELAY_SECONDS)
            assert self._stop_server is not None  # checked by install_blocker
            self._stop_server()
        else:
            job.state = "failed" if job.error else "done"

    def _step(self, job: UpdateJob, argv: list[str], *, shown: str | None = None) -> int:
        """Run one command, adding its output to the job. Returns its exit code."""
        job.add(f"$ {shown or ' '.join(argv)}")
        proc = subprocess.Popen(
            argv,
            cwd=self._workspace,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            job.add(line.rstrip("\r\n"))
        return proc.wait()


def restart_server() -> NoReturn:
    """Replace this process with a fresh server, on the updated install.

    Called once the server has shut down and closed its sessions. The new process
    keeps this one's terminal and arguments, so it serves the same folder on the
    same port.
    """
    print("Restarting with the updated jutul-agent...", file=sys.stderr, flush=True)
    sys.stdout.flush()
    argv = [sys.executable, "-m", "jutul_agent", *sys.argv[1:]]
    os.execve(sys.executable, argv, {**os.environ, RESTARTED_ENV_VAR: "1"})
