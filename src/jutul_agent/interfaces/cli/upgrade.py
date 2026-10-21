"""``jutul-agent upgrade``: bring the install and this folder's Julia env up to date.

Two things can fall behind. The install: a newer jutul-agent may be released. The
workspace: its Julia environment keeps the simulator versions it resolved, so after
an upgrade it can be older than the bundled skills expect. One command handles both,
in order: upgrade jutul-agent the right way for how it was installed, then update
the current folder's Julia environment to what that version supports.

``--check`` reports both without changing anything; ``--env-only`` skips the first
step (also how a freshly upgraded install is asked to do the second).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from jutul_agent import __version__
from jutul_agent.interfaces.cli._helpers import add_workspace_flags
from jutul_agent.paths import workspace_root
from jutul_agent.update_check import (
    install_info,
    is_newer,
    refresh_cache,
    upgrade_command,
)

# The Windows upgrade finishes after this process exits, so the new version updates
# the workspace when it next launches there instead.
_NEXT_LAUNCH_HINT = (
    "A workspace whose Julia environment the new version needs updated offers to do "
    "it the next time you launch there (or run `jutul-agent upgrade --env-only`)."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jutul-agent upgrade",
        description=(
            "Upgrade jutul-agent to the latest release, then bring this folder's "
            "Julia environment up to the simulator versions it supports."
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Report the installed and latest versions and this folder's environment; "
            "change nothing."
        ),
    )
    parser.add_argument(
        "--env-only",
        action="store_true",
        help="Only update this folder's Julia environment; don't upgrade jutul-agent itself.",
    )
    parser.add_argument(
        "--sysimage",
        choices=("rebuild", "off"),
        help=(
            "When this folder starts from a system image, rebuild it after the update "
            "or stop using one. Asked when run from a terminal; otherwise it is rebuilt."
        ),
    )
    add_workspace_flags(parser)
    return parser


def run(args: argparse.Namespace) -> int:
    ws = workspace_root()
    if args.env_only:
        return update_workspace_env(ws, sysimage=args.sysimage)

    info = install_info()
    latest = refresh_cache(force=True)  # synchronous, bypasses the cache TTL
    print(f"Installed: {__version__}")
    print(f"Latest:    {latest or 'unknown (offline, or no release published yet)'}")
    newer = latest is not None and is_newer(latest)

    if args.check:
        if newer:
            print(f"\nA newer version is available. Upgrade with `{upgrade_command(info)}`.")
        elif latest is not None:
            print("\nYou're on the latest version.")
        _report_workspace_env(ws)
        return 0

    if newer and info.method not in {"editable", "unknown"}:
        # Upgrade the install, then let the *new* version update the environment:
        # this process still runs the old code and the old simulator templates.
        code = _run_uv_tool_upgrade()
        if code != 0 or os.name == "nt":
            return code
        return _env_step_with_new_install(ws)

    if newer and info.method == "editable":
        where = f" (in {info.location})" if info.location else ""
        print(
            f"\nThis is a development checkout{where}. Update it with "
            "`git pull && uv sync`, then run `jutul-agent upgrade` again to update "
            "this folder's Julia environment."
        )
        return 0
    if newer:
        print(
            "\njutul-agent isn't installed as a managed package, so it can't upgrade "
            "itself. Install it with `uv tool install jutul-agent`.",
            file=sys.stderr,
        )
    elif latest is not None:
        print("\nYou're on the latest version.")
    return update_workspace_env(ws, sysimage=args.sysimage)


# ---------------------------------------------------------------------------
# The workspace's Julia environment.


def workspace_env_status(ws: Path):
    """``(adapter, project, sim, outdated)`` for a managed workspace env, else ``None``.

    ``outdated`` lists ``(package, resolved, required)`` for each package the env
    resolved outside what the installed jutul-agent supports. A folder that is not a
    workspace, or whose Julia project the user owns, has nothing to update here.
    """

    from jutul_agent.interfaces.cli._helpers import known_packages_map
    from jutul_agent.simulators import registry
    from jutul_agent.workspace import (
        auto_detect_simulator,
        load_workspace_config,
        outdated_env_packages,
        resolve_julia_project,
        user_owns_root_project,
    )

    project = resolve_julia_project(ws)
    if user_owns_root_project(ws) or not (project / "Project.toml").exists():
        return None
    sim = load_workspace_config(ws).simulator or auto_detect_simulator(known_packages_map(), ws)
    if sim is None:
        return None
    try:
        adapter = registry.get(sim)
    except KeyError:
        return None
    return adapter, project, sim, outdated_env_packages(project, adapter.julia_env_template_path)


def _describe(outdated) -> tuple[str, str]:
    found = ", ".join(f"{package} {version}" for package, version, _ in outdated)
    needed = ", ".join(f"{package} {spec}" for package, _, spec in outdated)
    return found, needed


def _current_or_newer(adapter, ws: Path, project: Path) -> str:
    """What to say about an env with nothing to update: current, or newer than this."""
    from jutul_agent.simulators.env_setup import newer_simulator_note

    note = newer_simulator_note(adapter, ws, project)
    if note is None:
        return "This folder's Julia environment is up to date."
    return f"{note[0].upper()}{note[1:]} Nothing to update."


def _report_workspace_env(ws: Path) -> None:
    status = workspace_env_status(ws)
    if status is None:
        return
    adapter, project, _, outdated = status
    if outdated:
        found, needed = _describe(outdated)
        print(
            f"This folder's Julia environment has {found}; this version needs {needed}. "
            "Update it with `jutul-agent upgrade`."
        )
    else:
        print(_current_or_newer(adapter, ws, project))


def update_workspace_env(ws: Path, *, sysimage: str | None = None) -> int:
    """Update this folder's Julia environment to what the installed version supports.

    ``sysimage`` (``"rebuild"`` or ``"off"``) answers the system image question in
    advance; see ``decide_sysimage``.
    """

    status = workspace_env_status(ws)
    if status is None:
        print("No jutul-agent managed Julia environment in this folder, so nothing else to update.")
        return 0
    adapter, project, sim, outdated = status
    if not outdated:
        print(_current_or_newer(adapter, ws, project))
        return 0
    found, needed = _describe(outdated)
    # Flush before Julia writes to the same output, or a log shows this line last.
    print(f"This folder's Julia environment has {found}; updating it to {needed}.", flush=True)
    rebuild_image = decide_sysimage(ws, interactive=sys.stdin.isatty(), choice=sysimage)
    return 0 if upgrade_workspace(adapter, ws, project, sim, sysimage=rebuild_image) else 1


def decide_sysimage(ws: Path, *, interactive: bool, choice: str | None = None) -> bool:
    """Whether to rebuild this folder's system image after updating its environment.

    The image is built from the old packages, so every launch would refuse it until
    it is rebuilt. Rebuilding takes a while, so the user chooses between that and
    starting without an image: up front with ``choice`` (``"rebuild"`` or ``"off"``,
    as the web UI does), or when asked in a terminal. Otherwise it is rebuilt, as
    ``init`` would.
    """

    from jutul_agent import sysimage as sysimage_mod
    from jutul_agent.sysimage_build import TYPICAL_BUILD_MINUTES
    from jutul_agent.workspace import load_workspace_config, write_workspace_config

    config = load_workspace_config(ws)
    if not config.sysimage:
        return False
    asked = choice is None and interactive
    if asked:
        rebuild = _ask(
            f"This folder starts Julia from a system image, which must be rebuilt for the "
            f"new packages (about {TYPICAL_BUILD_MINUTES} minutes). Rebuild it? If not, "
            "the folder stops using one; build it later with `jutul-agent sysimage build`. "
            "[Y/n] "
        )
        choice = "rebuild" if rebuild else "off"
    if choice != "off":
        if not asked:
            print(
                f"This folder starts from a system image, so it will be rebuilt for the new "
                f"packages (about {TYPICAL_BUILD_MINUTES} minutes).",
                flush=True,
            )
        return True
    from dataclasses import replace

    sysimage_mod.clear(ws)
    write_workspace_config(replace(config, sysimage=False), workspace=ws)
    print("This folder no longer starts from a system image.", flush=True)
    return False


def _ask(prompt: str) -> bool:
    try:
        return input(prompt).strip().lower() in ("", "y", "yes")
    except EOFError:
        return False


def upgrade_workspace(adapter, ws: Path, project: Path, sim: str, *, sysimage: bool) -> bool:
    """Update the workspace env in place, then rebuild its system image if asked to."""

    from jutul_agent.interfaces.cli.init import _build_sysimage
    from jutul_agent.simulators.env_setup import EnvSetupError, upgrade_env
    from jutul_agent.workspace import load_workspace_config

    try:
        upgrade_env(adapter, ws, project, sim)
    except EnvSetupError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return False
    print(f"Updated the {adapter.display_name} environment at {project}.")
    return not sysimage or _build_sysimage(adapter, ws, project, load_workspace_config(ws))


def _env_step_with_new_install(ws: Path) -> int:
    """Run the environment update with the newly installed jutul-agent."""

    # The launcher this process started from: uv has just upgraded it in place.
    # Looking it up on PATH instead could find another install entirely.
    launcher = Path(sys.argv[0])
    exe = str(launcher) if launcher.name.startswith("jutul-agent") and launcher.exists() else None
    exe = exe or shutil.which("jutul-agent")
    if exe is None:
        print(f"\n{_NEXT_LAUNCH_HINT}")
        return 0
    print("\n$ jutul-agent upgrade --env-only", flush=True)
    return subprocess.run(
        [exe, "upgrade", "--env-only", "--workspace", str(ws)], check=False
    ).returncode


def _run_uv_tool_upgrade() -> int:
    if shutil.which("uv") is None:
        print(
            "\nerror: `uv` is not on PATH, so the install can't be upgraded "
            "automatically. Install uv (https://docs.astral.sh/uv/), then run "
            "`uv tool upgrade jutul-agent`.",
            file=sys.stderr,
        )
        return 1

    argv = ["uv", "tool", "upgrade", "jutul-agent"]
    if os.name == "nt":
        return _upgrade_detached_windows(argv)

    print(f"\n$ {' '.join(argv)}", flush=True)
    result = subprocess.run(argv, check=False)
    if result.returncode != 0:
        print(
            f"\nerror: upgrade failed (uv exited {result.returncode}).",
            file=sys.stderr,
        )
    return result.returncode


def _upgrade_detached_windows(argv: list[str]) -> int:
    """Run the upgrade in a separate console, then let this process exit.

    On Windows the running ``jutul-agent.exe`` launcher is locked, so ``uv tool
    upgrade`` can't overwrite it from within this process (it fails copying the
    entrypoint). Launch uv in its own console and return so jutul-agent exits and
    frees the executable; uv builds and updates the venv first, reaching the
    launcher copy only after we're gone. Falls back to manual instructions if the
    detached launch can't start.
    """

    new_console = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
    try:
        subprocess.Popen(argv, creationflags=new_console, close_fds=True)
    except OSError as exc:
        print(
            f"\nerror: couldn't start the upgrade ({exc}). Run it yourself from a "
            "new terminal (not from inside jutul-agent):\n    uv tool upgrade jutul-agent",
            file=sys.stderr,
        )
        return 1
    print(
        "\nUpgrading in a new window. Windows can't replace jutul-agent's own "
        "running executable, so this process will exit to release it — reopen "
        f"jutul-agent once the other window finishes.\n{_NEXT_LAUNCH_HINT}"
    )
    return 0
