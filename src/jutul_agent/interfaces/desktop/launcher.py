"""``jutul-agent desktop``: start the desktop app, or install its Electron runtime.

The app is the Electron shell in ``electron/`` plus this Python package as its
backend. Electron is the one dependency Python packaging cannot carry, so it is
installed on request (``jutul-agent desktop --install``) into the Python
environment jutul-agent itself runs from: ``<env>/share/jutul-agent-desktop``.
Removing the environment removes the app with it, and nothing is installed
globally.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ELECTRON_APP_DIR = Path(__file__).resolve().parent / "electron"
ELECTRON_ENV_VAR = "JUTUL_AGENT_ELECTRON"


def runtime_dir() -> Path:
    """Where the Electron runtime lives inside the current Python environment."""
    return Path(sys.prefix) / "share" / "jutul-agent-desktop"


def electron_binary() -> Path | None:
    """The Electron executable to run, or ``None`` if it is not installed."""
    override = os.environ.get(ELECTRON_ENV_VAR)
    if override:
        return Path(override) if Path(override).is_file() else None
    package = runtime_dir() / "node_modules" / "electron"
    marker = package / "path.txt"  # written by Electron's own installer
    if not marker.is_file():
        return None
    binary = package / "dist" / marker.read_text(encoding="utf-8").strip()
    return binary if binary.is_file() else None


def install() -> int:
    """Install the pinned Electron into this environment with npm."""
    npm = shutil.which("npm")
    node = shutil.which("node")
    if npm is None or node is None:
        print(
            "error: installing the desktop runtime needs Node.js (node and npm on PATH). "
            "Install Node 22.12 or newer, then run `jutul-agent desktop --install` again.",
            file=sys.stderr,
        )
        return 1
    target = runtime_dir()
    target.mkdir(parents=True, exist_ok=True)
    # Only the manifest and its lock: the app's own code runs from the package.
    for name in ("package.json", "package-lock.json"):
        source = ELECTRON_APP_DIR / name
        if source.is_file():
            shutil.copyfile(source, target / name)
    locked = (target / "package-lock.json").is_file()
    print(f"Installing the desktop runtime into {target} ...", file=sys.stderr)
    done = subprocess.run(
        [npm, "ci" if locked else "install", "--no-audit", "--no-fund"], cwd=target, check=False
    )
    if done.returncode != 0:
        return done.returncode
    # npm may be configured not to run packages' install scripts, and Electron's
    # is what downloads the runtime itself. Running it again is a no-op.
    script = target / "node_modules" / "electron" / "install.js"
    if electron_binary() is None and script.is_file():
        done = subprocess.run([node, str(script)], cwd=script.parent, check=False)
        if done.returncode != 0:
            return done.returncode
    if electron_binary() is None:
        print("error: Electron did not install; see npm's output above.", file=sys.stderr)
        return 1
    print("Desktop runtime installed. Start the app with `jutul-agent desktop`.", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jutul-agent desktop",
        description=(
            "Run jutul-agent as a desktop app: the web UI in its own window, with no "
            "server and no open port. Options other than those below are the ones "
            "`jutul-agent web` takes for its sessions (--sim, --model, --approval-mode, "
            "--threads, ...) and are passed on to the app's backend."
        ),
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Install the Electron runtime into this Python environment, then exit.",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="Workspace folder to open (default: the current directory).",
    )
    parser.add_argument(
        "--electron-arg",
        action="append",
        default=[],
        metavar="ARG",
        help="Extra argument for Electron itself (for development). Repeatable.",
    )
    return parser


def run(argv: list[str]) -> int:
    args, backend_args = build_parser().parse_known_args(argv)
    if args.install:
        return install()
    binary = electron_binary()
    if binary is None:
        print(
            "error: the desktop runtime (Electron) is not installed in this environment.\n"
            "Run `jutul-agent desktop --install` once (needs Node.js), or point "
            f"${ELECTRON_ENV_VAR} at an Electron executable.",
            file=sys.stderr,
        )
        return 1
    # Fail on a mistyped option here, in the terminal, rather than in a window.
    from jutul_agent.interfaces.cli import web as web_cmd

    web_cmd.build_parser(prog="jutul-agent desktop").parse_args(backend_args)

    workspace = (args.workspace or Path.cwd()).resolve()
    env = dict(os.environ)
    env["JUTUL_AGENT_PYTHON"] = sys.executable
    env["JUTUL_AGENT_BACKEND_ARGS"] = json.dumps(backend_args)
    # Set when Electron is launched from an editor's terminal; it would make the
    # binary behave as plain Node and ignore the app.
    env.pop("ELECTRON_RUN_AS_NODE", None)
    command = [str(binary), str(ELECTRON_APP_DIR), f"--workspace={workspace}", *args.electron_arg]
    try:
        return subprocess.run(command, env=env, check=False).returncode
    except KeyboardInterrupt:
        return 0
