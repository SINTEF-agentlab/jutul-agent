"""Updating jutul-agent and this folder's Julia environment from the web UI.

The browser can do what ``jutul-agent upgrade`` does in a terminal:

1. Upgrade the install with ``uv tool upgrade jutul-agent``.
2. Bring this folder's Julia environment up to what the installed version supports,
   by running ``jutul-agent upgrade --env-only`` on the interpreter this server runs
   on. After step 1 that interpreter imports the new version, so its simulator
   templates decide, as they do in the terminal.
3. Restart the server into the new version when step 1 changed the install.

Step 1 needs a uv tool install. On Windows a running program can't replace its own
files, so there the server hands all three steps to ``update_helper.py`` in a new
console window and exits; the helper starts jutul-agent again in that window. Either
way the restarted server reports how the update went (``report_last_update``), so the
reloaded page can show a failure.

Updates are only accepted from a page served by this computer, never from another
site in the same browser. The commands are fixed: a request only says whether to do
step 1 and what to do with a system image. One update runs at a time, the live
sessions are closed first (their chats stay on disk), and no session starts until it
is done.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

from jutul_agent.interfaces.cli.upgrade import workspace_env_status
from jutul_agent.paths import state_home
from jutul_agent.update_check import install_info, is_enabled, pending_update
from jutul_agent.workspace import load_workspace_config

if TYPE_CHECKING:
    from jutul_agent.interfaces.server.manager import SessionManager

# Set for the restarted server, so it starts even if the update left something to
# fix: the page that asked for the update is waiting for it, and says what is wrong.
RESTARTED_ENV_VAR = "JUTUL_AGENT_RESTARTED_AFTER_UPDATE"

# Hands the update to the helper in a separate window on any system, for testing.
SEPARATE_WINDOW_ENV_VAR = "JUTUL_AGENT_TEST_SEPARATE_WINDOW_UPDATE"

# Identifies this server process, so a page can tell when a restart has happened.
BOOT_ID = uuid.uuid4().hex

_MAX_LINES = 5000
# The page polls about once a second; this lets it show the last lines before the
# server goes away.
_RESTART_DELAY_SECONDS = 2.0
# How long the helper waits for this server to exit before trying anyway.
_HELPER_WAIT_SECONDS = 120

_UPGRADE_FAILED = "jutul-agent could not be upgraded. See the output above."
_ENV_FAILED = "The Julia environment could not be updated. See the output above."

# Why this install can't upgrade itself from the browser, as shown in the dialog.
_INSTALL_BLOCKERS = {
    "not-local": (
        "Updating from the browser only works when jutul-agent runs on this computer. "
        "Run `jutul-agent upgrade` where it runs, then start it again."
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


def _separate_window() -> bool:
    """Whether upgrading jutul-agent is left to the helper in a separate window."""
    return _on_windows() or os.environ.get(SEPARATE_WINDOW_ENV_VAR) == "1"


def _update_dir() -> Path:
    """Where the last update's log and outcome are kept, and the helper runs from."""
    return state_home() / "update"


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
        # The update went to the helper in a separate window, which starts the new
        # server: this one only has to exit.
        self.handed_off = False

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
            # Upgrading jutul-agent continues in a separate window, which it then runs in.
            "separate_window": _separate_window(),
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
        if job.install and _separate_window():
            self._hand_off(job, sysimage)
            return
        try:
            if job.install and self._step(job, ["uv", "tool", "upgrade", "jutul-agent"]):
                job.error = _UPGRADE_FAILED
                job.state = "failed"
                return
            env_step = [sys.executable, "-m", "jutul_agent", *self._env_step_args(sysimage)]
            if self._step(job, env_step, shown="jutul-agent upgrade --env-only"):
                job.error = _ENV_FAILED
        except Exception as exc:
            # Never leave the job looking like it still runs: that blocks sessions.
            job.error = f"The update could not run: {exc}"
        if job.install and _installed_version() != self._version_at_start:
            _record_outcome(job.lines, job.error)
            job.state = "restarting"
            self.restart_requested = True
            self._stop_soon()
        else:
            job.state = "failed" if job.error else "done"

    def _env_step_args(self, sysimage: str | None) -> list[str]:
        args = ["upgrade", "--env-only", "--workspace", str(self._workspace)]
        return [*args, "--sysimage", sysimage] if sysimage else args

    def _stop_soon(self) -> None:
        time.sleep(_RESTART_DELAY_SECONDS)
        assert self._stop_server is not None  # checked by install_blocker
        self._stop_server()

    def _hand_off(self, job: UpdateJob, sysimage: str | None) -> None:
        """Leave the update to the helper in a new window, and stop so it can run."""
        plan = self._helper_plan(sysimage)
        try:
            start_helper(plan)
        except Exception as exc:
            job.error = f"The update could not start: {exc}"
            job.state = "failed"
            return
        job.add("jutul-agent continues updating in a separate window, and runs there after.")
        job.add(f"Its output is also saved in {plan['log']}.")
        job.state = "restarting"
        self.handed_off = True
        self._stop_soon()

    def _helper_plan(self, sysimage: str | None) -> dict[str, Any]:
        """What the helper does once this server has exited; see ``update_helper.py``."""
        launcher = _launcher()
        agent = [str(launcher)] if launcher else [sys.executable, "-m", "jutul_agent"]
        wait_for = [os.getpid()]
        if launcher and _on_windows():
            # The launcher waits on this process, maybe through the environment's
            # python.exe; the file check below covers it either way.
            wait_for.append(os.getppid())
        folder = _update_dir()
        return {
            "wait_for": wait_for,
            # Windows refuses to replace a running program, so wait until it can.
            "free": [str(launcher)] if launcher and _on_windows() else [],
            "timeout": _HELPER_WAIT_SECONDS,
            "cwd": str(self._workspace),
            "steps": [
                {
                    "argv": [shutil.which("uv") or "uv", "tool", "upgrade", "jutul-agent"],
                    "shown": "uv tool upgrade jutul-agent",
                    "error": _UPGRADE_FAILED,
                },
                {
                    "argv": [*agent, *self._env_step_args(sysimage)],
                    "shown": "jutul-agent upgrade --env-only",
                    "error": _ENV_FAILED,
                },
            ],
            "relaunch": {
                "argv": [*agent, *sys.argv[1:]],
                "cwd": os.getcwd(),
                "env": {RESTARTED_ENV_VAR: "1"},
            },
            "log": str(folder / "update.log"),
            "outcome": str(folder / "outcome.json"),
        }

    def report_last_update(self) -> None:
        """Show how the update this server was restarted after went.

        Its output went to the log, and the page that asked for it reloads into this
        server, so a failure has to be reported from here.
        """
        outcome_file = _update_dir() / "outcome.json"
        try:
            outcome = json.loads(outcome_file.read_text(encoding="utf-8"))
            outcome_file.unlink()
        except (OSError, ValueError):
            return
        state = "done" if outcome.get("state") == "done" else "failed"
        job = UpdateJob(install=True, state=state, error=outcome.get("error"))
        try:
            log = (_update_dir() / "update.log").read_text(encoding="utf-8", errors="replace")
        except OSError:
            log = ""
        for line in log.splitlines():
            job.add(line)
        self._job = job

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


def _launcher() -> Path | None:
    """The ``jutul-agent`` program this process was started from, if it was."""
    path = Path(sys.argv[0]).absolute()
    return path if path.name.startswith("jutul-agent") and path.is_file() else None


def _record_outcome(lines: list[str], error: str | None) -> None:
    """Keep an update's output and outcome for the restarted server to report."""
    folder = _update_dir()
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "update.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    outcome = {"state": "failed" if error else "done", "error": error}
    (folder / "outcome.json").write_text(json.dumps(outcome), encoding="utf-8")


def start_helper(plan: dict[str, Any]) -> None:
    """Start ``update_helper.py`` on ``plan`` in a new console window.

    It runs from a copy in the state folder, on the base Python rather than the tool
    environment's, so nothing it holds open is among the files the upgrade replaces.
    """
    folder = Path(plan["log"]).parent
    folder.mkdir(parents=True, exist_ok=True)
    script = folder / "update_helper.py"
    shutil.copyfile(Path(__file__).with_name("update_helper.py"), script)
    plan_file = folder / "plan.json"
    plan_file.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    python = getattr(sys, "_base_executable", None) or sys.executable
    if not Path(python).is_file():
        python = sys.executable
    # Set for this interpreter by a Windows venv's python.exe; it would put the
    # helper back in the environment.
    env = {k: v for k, v in os.environ.items() if k != "__PYVENV_LAUNCHER__"}
    if _on_windows():
        window: dict[str, Any] = {"creationflags": getattr(subprocess, "CREATE_NEW_CONSOLE", 0)}
    else:
        # No console to open: outlive the server in a session of its own.
        window = {"start_new_session": True, "stdin": subprocess.DEVNULL}
    subprocess.Popen(
        [python, "-I", "-S", str(script), str(plan_file)], cwd=folder, env=env, **window
    )


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
