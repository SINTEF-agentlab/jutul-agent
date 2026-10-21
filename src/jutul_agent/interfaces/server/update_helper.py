"""Finish updating jutul-agent after the server that started the update has exited.

On Windows a running program's files are locked, so the server can't upgrade itself
the way it does elsewhere. It writes a plan, starts this script in a new console
window with the base Python (outside the tool's environment, so it locks nothing the
upgrade replaces), and exits. This script then:

1. waits for the server to exit, and for its launcher to be free to replace;
2. runs the plan's steps in order (``uv tool upgrade jutul-agent``, then the update
   of the folder's Julia environment), stopping at the first that fails;
3. records the outcome, which the restarted server reports to the page;
4. starts jutul-agent again with its original arguments, in this window, and stays
   to say so if it stops with an error.

Everything it shows also goes to the plan's log file. It uses only the standard
library, since it runs outside jutul-agent's environment, and nothing in it is
specific to Windows except how it waits for a process.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from collections.abc import Callable
from typing import Any

_POLL_SECONDS = 0.2
# How a program ends when Ctrl+C stops it: Windows' STATUS_CONTROL_C_EXIT, or the
# signal elsewhere. That is the user stopping jutul-agent, not an error to report.
_CTRL_C_EXITS = (0xC000013A, -signal.SIGINT)


def main(argv: list[str]) -> int:
    with open(argv[1], encoding="utf-8") as f:
        plan = json.load(f)
    server = finish_update(plan)
    if server is None:
        _pause()
        return 1
    # This window is the server's now: Ctrl+C is for it. Stay to report how it ends.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    code = server.wait()
    if code not in (0, *_CTRL_C_EXITS):
        print(f"\njutul-agent stopped with exit code {code}.", flush=True)
        _pause()
    return code


def finish_update(plan: dict[str, Any]) -> subprocess.Popen | None:
    """Do the plan, then start jutul-agent again. ``None`` if it could not start."""
    with open(plan["log"], "w", encoding="utf-8") as log:

        def say(line: str) -> None:
            log.write(line + "\n")
            log.flush()
            with contextlib.suppress(OSError, ValueError):  # no console to show it in
                print(line, flush=True)

        say("Updating jutul-agent. Keep this window open: jutul-agent starts here again after.")
        try:
            deadline = time.monotonic() + plan["timeout"]
            if not wait_for_exit(plan["wait_for"], deadline):
                say("jutul-agent hasn't exited; updating anyway.")
            for path in plan["free"]:
                if not wait_until_free(path, deadline):
                    say(f"{path} is in use, perhaps by another jutul-agent; updating anyway.")
            error = run_steps(plan["steps"], plan["cwd"], say)
        except Exception as exc:
            say(traceback.format_exc().rstrip())
            error = f"The update could not run: {exc}"
        # Starting jutul-agent again matters more than reporting how this went.
        with contextlib.suppress(OSError):
            write_outcome(plan["outcome"], error)
        if error:
            say(error)

        relaunch = plan["relaunch"]
        say("Starting jutul-agent again...")
        try:
            return subprocess.Popen(
                relaunch["argv"], cwd=relaunch["cwd"], env={**os.environ, **relaunch["env"]}
            )
        except OSError as exc:
            say(f"Could not start jutul-agent again ({exc}). Start it yourself.")
            return None


def wait_for_exit(pids: list[int], deadline: float) -> bool:
    """Wait until every process in ``pids`` has exited. False if time ran out."""
    if os.name == "nt":
        return _wait_for_exit_windows(pids, deadline)
    while any(_alive(pid) for pid in pids):
        if time.monotonic() > deadline:
            return False
        time.sleep(_POLL_SECONDS)
    return True


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)  # on Windows this would end the process instead
    except ProcessLookupError:
        return False
    except PermissionError:  # exists, but belongs to someone else
        return True
    return True


def _wait_for_exit_windows(pids: list[int], deadline: float) -> bool:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    synchronize, wait_object_0 = 0x00100000, 0

    # A process that can't be opened has exited already.
    handles = [h for h in (kernel32.OpenProcess(synchronize, False, pid) for pid in pids) if h]
    try:
        for handle in handles:
            remaining = max(0, int((deadline - time.monotonic()) * 1000))
            if kernel32.WaitForSingleObject(handle, remaining) != wait_object_0:
                return False
        return True
    finally:
        for handle in handles:
            kernel32.CloseHandle(handle)


def wait_until_free(path: str, deadline: float) -> bool:
    """Wait until ``path`` can be opened for writing, as replacing it needs.

    Windows refuses that while the file is a running program. Nothing is written.
    """
    while True:
        try:
            with open(path, "r+b"):
                return True
        except FileNotFoundError:
            return True
        except OSError:
            if time.monotonic() > deadline:
                return False
        time.sleep(_POLL_SECONDS)


def run_steps(steps: list[dict[str, Any]], cwd: str, say: Callable[[str], None]) -> str | None:
    """Run the steps in order. The error of the first that fails, else ``None``."""
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    for step in steps:
        say(f"$ {step['shown']}")
        try:
            proc = subprocess.Popen(
                step["argv"],
                cwd=cwd,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        except OSError as exc:
            say(str(exc))
            return step["error"]
        assert proc.stdout is not None
        for raw in proc.stdout:
            say(raw.decode("utf-8", "replace").rstrip("\r\n"))
        if proc.wait() != 0:
            return step["error"]
    return None


def write_outcome(path: str, error: str | None) -> None:
    partial = path + ".partial"
    with open(partial, "w", encoding="utf-8") as f:
        json.dump({"state": "failed" if error else "done", "error": error}, f)
    os.replace(partial, path)


def _pause() -> None:
    """Keep a console window open until its message has been read."""
    try:
        if sys.stdin is not None and sys.stdin.isatty():
            input("Press Enter to close this window.")
    except (EOFError, OSError):
        pass


if __name__ == "__main__":
    sys.exit(main(sys.argv))
