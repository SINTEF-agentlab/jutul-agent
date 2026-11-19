"""Carrying an install over from Makker's old name, jutul-agent.

Temporary. Everything that knows the old name lives in this module, so removing
the transition later means deleting it and the handful of calls into it.

What it does for someone who used jutul-agent before:

- their data folder (API keys, chats, memory) is moved to the new name, and a
  link is left at the old place because saved chats hold paths into it;
- a workspace's ``.jutul-agent/`` folder is renamed to ``.makker/``, and its Julia
  environment loses the packages that carried the old name, so the usual
  environment update can add their successors;
- output a session wrote under ``jutul-agent-output/`` is still found, in place;
- settings still spelled ``JUTUL_AGENT_*`` are pointed out;
- the last jutul-agent release, which only exists to install Makker, is removed
  once Makker is started under its own name.

Each step leaves things as they were if it cannot finish, and is tried again the
next time Makker starts.
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import tomli_w

OLD_NAME = "jutul-agent"
OLD_WORKSPACE_DIRNAME = ".jutul-agent"
OLD_OUTPUT_DIRNAME = "jutul-agent-output"
OLD_SETTING_PREFIX = "JUTUL_AGENT_"
OLD_JULIA_PREFIX = "JutulAgent"
OLD_ENV_MARKERS = (
    ".jutul-agent-template",
    ".jutul-agent-precompiled",
    ".jutul-agent-dep-source",
    ".jutul-agent-warm-source",
)

# Set by the last jutul-agent release when it starts Makker on the user's behalf.
VIA_OLD_COMMAND = "MAKKER_VIA_JUTUL_AGENT"

_NOTED_BOTH_FOLDERS = ".noted-old-data-folder"


def _say(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# The data folder.


def _is_link(path: Path) -> bool:
    return path.is_symlink() or os.path.isjunction(path)


def resolve_state_home(new: Path) -> Path:
    """Where the data is: ``new``, or the old folder while it has not been moved."""

    old = new.with_name(OLD_NAME)
    if new.exists() or not old.is_dir():
        return new
    return old


def migrate_state_home(new: Path) -> None:
    """Move the old data folder to ``new`` and leave a link where it was.

    The link matters: saved chats hold absolute paths to their memory notes, and
    an older version still installed elsewhere on the machine reads its keys from
    the old place. If the folder cannot be moved (on Windows, while a program
    has a file in it open) it is used where it is and moved another time.
    """

    old = new.with_name(OLD_NAME)
    if _is_link(old) or not old.is_dir():
        return
    if new.exists():
        marker = new / _NOTED_BOTH_FOLDERS
        if not marker.exists():
            _say(f"Makker keeps its data in {new}. The older folder {old} is not used.")
            with contextlib.suppress(OSError):
                marker.touch()
        return
    try:
        old.rename(new)
    except OSError:
        return
    _leave_link(old, new)
    # What the old name's update check last saw says nothing about Makker's releases.
    with contextlib.suppress(OSError):
        (new / "update-check.json").unlink()
    _say(f"Moved your data from {old} to {new}.")


def _leave_link(old: Path, new: Path) -> None:
    try:
        os.symlink(new, old, target_is_directory=True)
        return
    except (OSError, NotImplementedError):
        pass
    # Windows without the privilege to make symlinks: a junction needs none.
    with contextlib.suppress(Exception):
        import _winapi

        _winapi.CreateJunction(str(new), str(old))


# ---------------------------------------------------------------------------
# A workspace.


def adopt_workspace(workspace: Path) -> None:
    """Rename a workspace's ``.jutul-agent/`` to ``.makker/``.

    One atomic rename, so two programs starting at once cannot both do it. What
    is inside is converted separately, under the environment lock
    (:func:`convert_env`), by whichever of them prepares the environment first.
    """

    from makker.workspace import WORKSPACE_DIRNAME, ensure_workspace_dir

    old = workspace / OLD_WORKSPACE_DIRNAME
    new = workspace / WORKSPACE_DIRNAME
    if new.exists() or not old.is_dir() or old.is_symlink():
        return
    try:
        old.rename(new)
    except FileNotFoundError:
        return  # another program got there first
    except OSError as exc:
        raise SystemExit(
            f"error: could not rename {old} to {new.name} ({exc}).\n"
            "Close any other Makker or jutul-agent program using this folder, then "
            "start Makker again."
        ) from exc
    ensure_workspace_dir(workspace)
    _say(f"Renamed {old.name}/ to {new.name}/ in {workspace}.")


def convert_env(julia_project: Path) -> bool:
    """Drop the old-named Julia packages from a managed environment.

    Removes them from ``Project.toml`` and deletes their copies beside it. The
    manifest is left alone, so the versions the user had resolved stay put:
    resolving prunes what the project no longer names. The packages' successors
    come in afterwards the way any package a newer template adds does. Returns
    whether anything changed. Call it with the environment lock held.
    """

    project = julia_project / "Project.toml"
    try:
        data = tomllib.loads(project.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return False

    changed = False
    for table in ("deps", "sources", "compat", "weakdeps", "extras"):
        entries = data.get(table)
        if not isinstance(entries, dict):
            continue
        for name in [name for name in entries if name.startswith(OLD_JULIA_PREFIX)]:
            del entries[name]
            changed = True
    if changed:
        project.write_text(tomli_w.dumps(data), encoding="utf-8")

    for child in julia_project.iterdir():
        is_old_package = child.name.startswith(OLD_JULIA_PREFIX) and child.is_dir()
        if is_old_package and (child / "Project.toml").is_file():
            shutil.rmtree(child, ignore_errors=True)
            changed = True
    for marker in OLD_ENV_MARKERS:
        with contextlib.suppress(OSError):
            (julia_project / marker).unlink()
    return changed


def previous_output_dir(output_dir: Path) -> Path:
    """Where a session's output folder would be under the old output folder name.

    ``output_dir`` is ``<workspace>/makker-output/sessions/<name>``. Old output is
    never moved, so that links to earlier plots and reports keep working.
    """

    return output_dir.parents[2] / OLD_OUTPUT_DIRNAME / output_dir.parent.name / output_dir.name


# ---------------------------------------------------------------------------
# Settings and the old install.


def old_settings(environ: dict[str, str] | None = None) -> list[str]:
    """Settings still under the old prefix that have no counterpart under the new one."""

    environ = os.environ if environ is None else environ
    return sorted(
        name
        for name in environ
        if name.startswith(OLD_SETTING_PREFIX)
        and "MAKKER_" + name[len(OLD_SETTING_PREFIX) :] not in environ
    )


_warned_about_settings = False


def warn_about_old_settings() -> None:
    global _warned_about_settings
    names = old_settings()
    if not names or _warned_about_settings:
        return
    _warned_about_settings = True
    renamed = ", ".join("MAKKER_" + name[len(OLD_SETTING_PREFIX) :] for name in names)
    _say(
        f"These settings use the old name and are ignored: {', '.join(names)}.\n"
        f"Rename them to: {renamed}."
    )


def _old_install_version(tool_dir: Path) -> tuple[int, int] | None:
    for pattern in ("lib/python*/site-packages", "Lib/site-packages"):
        for info in tool_dir.glob(f"{pattern}/jutul_agent-*.dist-info"):
            match = re.match(r"jutul_agent-(\d+)\.(\d+)", info.name)
            if match:
                return int(match.group(1)), int(match.group(2))
    return None


def remove_old_install() -> None:
    """Uninstall the last jutul-agent release once Makker runs under its own name.

    That release contains nothing but the step onto Makker, so it is of no use
    after it. An earlier jutul-agent, which is a working program someone may
    have kept on purpose, is never touched. Nor is anything while Makker is being
    started through the old command, which is that install running.
    """

    if os.environ.get(VIA_OLD_COMMAND):
        return
    from makker.update_check import PACKAGE, uv_tool_name

    here = Path(sys.prefix)
    if not (here / "uv-receipt.toml").is_file() or uv_tool_name() != PACKAGE:
        return
    old = here.parent / OLD_NAME
    if not (old / "uv-receipt.toml").is_file():
        return
    version = _old_install_version(old)
    if version is None or version < (0, 3):
        return
    uv = shutil.which("uv")
    if uv is None:
        return
    try:
        done = subprocess.run(
            [uv, "tool", "uninstall", OLD_NAME], capture_output=True, text=True, timeout=120
        )
    except (OSError, subprocess.SubprocessError):
        return
    if done.returncode == 0:
        _say("Removed the old jutul-agent install. The command is `makker` from now on.")


# ---------------------------------------------------------------------------
# All of it, at start.


def adopt_previous_install(workspace: Path, *, default_state_home: Path | None) -> None:
    """Bring a jutul-agent install's data and workspace across, where there is one.

    ``default_state_home`` is the data folder's standard location, or ``None``
    when the user named another one, which is then theirs and left alone.
    """

    if default_state_home is not None:
        migrate_state_home(default_state_home)
    adopt_workspace(workspace)
    remove_old_install()
