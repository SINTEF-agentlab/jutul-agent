"""``python -m jutul_agent.interfaces.desktop.backend``: the desktop app's backend.

Started by the Electron shell, never by hand. It takes the options of
``jutul-agent web`` (minus the ones about binding a port), builds the same
application, and serves it over its own stdin/stdout instead of over a socket.

stdout is the frame channel, so nothing else may write to it: the real stdout
is moved to a private descriptor first and file descriptor 1 is pointed at
stderr, which the shell shows as the backend's log. A stray ``print`` (ours, a
library's, a child process's) then lands in the log rather than corrupting a
frame.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import signal
import sys
from typing import BinaryIO

# Set before any session exists: Julia kernels are then reached over anonymous
# pipes they inherit at spawn, and serve live plots over a second such pipe, so
# no process of the desktop app binds or listens on anything (see
# juliakernel/config.py). This is a hard constraint of the desktop app, not a
# preference: nothing here falls back to a socket.
IPC_ENV_VAR = "JUTUL_AGENT_IPC"


def _claim_stdio() -> tuple[BinaryIO, BinaryIO]:
    """The frame streams, with fd 0 and fd 1 taken away from everything else."""
    frames_in = os.fdopen(os.dup(0), "rb", buffering=0)
    frames_out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)  # whatever writes to "stdout" from here on reaches the log
    with open(os.devnull, "rb") as devnull:
        os.dup2(devnull.fileno(), 0)  # nothing may consume frames by reading stdin
    sys.stdout = sys.stderr
    if os.name == "nt":
        import msvcrt

        for stream in (frames_in, frames_out):
            msvcrt.setmode(stream.fileno(), os.O_BINARY)
    return frames_in, frames_out


def build_parser() -> argparse.ArgumentParser:
    from jutul_agent.interfaces.cli import web as web_cmd

    parser = web_cmd.build_parser(prog="jutul-agent desktop")
    parser.description = "The backend process of the jutul-agent desktop app."
    return parser


def main(argv: list[str] | None = None) -> int:
    frames_in, frames_out = _claim_stdio()
    os.environ[IPC_ENV_VAR] = "pipe"

    from dotenv import load_dotenv

    load_dotenv()
    args = build_parser().parse_args(argv)

    from jutul_agent import __version__
    from jutul_agent.interfaces.cli import web as web_cmd
    from jutul_agent.interfaces.desktop.asgi_bridge import serve

    # ``updatable=False``: the browser's "update now" replaces and restarts a
    # server process, which has no meaning for a backend the shell owns.
    built = web_cmd.build_app(args, updatable=False, served_as="the desktop app (IPC, no port)")
    if isinstance(built, int):
        return built
    app, _updater = built

    async def run() -> None:
        def on_ready(bridge) -> None:
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGTERM, signal.SIGINT):
                # Not available on Windows' event loop; there the shell's
                # ``shutdown`` frame and the closing pipe do the same job.
                with contextlib.suppress(NotImplementedError, RuntimeError):
                    loop.add_signal_handler(sig, bridge.stopping.set)

        await serve(app, frames_in, frames_out, version=__version__, on_ready=on_ready)

    asyncio.run(run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
