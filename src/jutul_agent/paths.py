"""Canonical filesystem locations used across the package.

Three roots, cleanly separated:

- ``PACKAGE_ROOT``: computed from this file. Read-only at runtime; per-
  simulator assets (julia envs, skills, adapter modules) live under
  ``PACKAGE_ROOT / "simulators" / <name>``.
- ``workspace_root()``: the user's working directory at invocation time
  (or an explicit override). Read/write; default for shell and file tools.
- ``state_home()``: sessions, traces, per-workspace state. Defaults to
  ``$XDG_DATA_HOME/jutul-agent`` or ``~/.local/share/jutul-agent``.

The workspace and state-home anchors are runtime-mutable so the CLI can set
them once on startup and library code can read them via the helpers.
"""

from __future__ import annotations

import hashlib
import os
import re
from datetime import date
from pathlib import Path

PACKAGE_ROOT: Path = Path(__file__).resolve().parent

SHARED_SKILLS_DIR: Path = PACKAGE_ROOT / "simulators" / "shared_skills"

_workspace_root_override: Path | None = None
_state_home_override: Path | None = None


def set_workspace_root(path: Path | None) -> None:
    global _workspace_root_override
    _workspace_root_override = path.resolve() if path is not None else None


def workspace_root() -> Path:
    if _workspace_root_override is not None:
        return _workspace_root_override
    return Path.cwd().resolve()


def set_state_home(path: Path | None) -> None:
    global _state_home_override
    _state_home_override = path.resolve() if path is not None else None


def state_home() -> Path:
    if _state_home_override is not None:
        return _state_home_override
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return Path(xdg).resolve() / "jutul-agent"
    return Path.home() / ".local" / "share" / "jutul-agent"


def user_config_path() -> Path:
    """User-global config file, at the root of the state home.

    Settings here apply across every workspace (e.g. the default model); the
    per-workspace ``.jutul-agent/config.toml`` overrides them.
    """
    return state_home() / "config.toml"


def user_skills_dir() -> Path:
    """User-global skills directory: skills applied to every simulator.

    Each subdirectory is a skill folder containing a ``SKILL.md``.
    """
    return state_home() / "skills"


def user_simulators_dir() -> Path:
    """Root for user-defined simulators and per-simulator user skills.

    Each subdirectory either holds an ``adapter.py`` (a user simulator) or
    a ``skills/`` subdirectory (per-simulator user skills for a built-in).
    """
    return state_home() / "simulators"


def is_host_path(text: str) -> bool:
    """Whether ``text`` names a real host location rather than a virtual workspace path.

    A Windows drive (``C:\\...``) is unambiguous. On POSIX a leading-slash path
    can still be virtual (``/model.jl`` means a workspace file), so it only
    counts as a host path when its first segment is a real top-level directory
    (``/etc``, ``/home``, ``/tmp``, …).
    """

    if Path(text).drive:
        return True
    first = text.lstrip("/").split("/", 1)[0] if text.startswith("/") else ""
    return bool(first) and Path("/" + first).is_dir()


def resolve_in_workspace(raw: str | Path, *, workspace: Path | None = None) -> Path | None:
    """Map an agent-visible path to the real path inside the workspace, or ``None``.

    The one place that knows the agent's path model. Accepts every form the
    agent uses (workspace-relative ``model.jl``, virtual absolute ``/model.jl``
    or ``/workspace/model.jl``, and the file's real absolute path) and returns
    the corresponding real path. Returns ``None`` when the path points outside
    the workspace (a real host path like ``/tmp/x``, or a ``..`` escape): the
    same rule the workspace file tools enforce, so previews and tools agree
    about which file a path means.

    Cross-platform note: on POSIX a leading-slash string is host-absolute and
    ``Path.is_absolute()`` catches it; on Windows the same string is *not*
    absolute (no drive letter), so leading-slash strings are routed through the
    virtual-path branch explicitly.
    """

    ws = (workspace or workspace_root()).resolve()
    text = str(raw)
    if not text:
        return None
    p = Path(text)

    if p.is_absolute():
        try:
            p.resolve().relative_to(ws)
            return p  # the file's real path, already inside the workspace
        except (ValueError, OSError):
            pass
        if is_host_path(text):
            return None  # a real machine path outside the workspace

    rel = text.lstrip("/")
    if rel.startswith("workspace/"):
        rel = rel[len("workspace/") :]
    candidate = ws / rel
    try:
        candidate.resolve().relative_to(ws)
    except (ValueError, OSError):
        return None  # `..` escape
    return candidate


def workspace_hash(workspace: Path | None = None) -> str:
    """Stable 12-char hash of the workspace's resolved path."""
    ws = (workspace or workspace_root()).resolve()
    return hashlib.sha256(str(ws).encode("utf-8")).hexdigest()[:12]


def workspace_state_dir(workspace: Path | None = None) -> Path:
    """Per-workspace state directory under ``state_home()``."""
    return state_home() / "workspaces" / workspace_hash(workspace)


WORKSPACE_OUTPUT_DIRNAME = "jutul-agent-output"


def workspace_output_dir(workspace: Path | None = None) -> Path:
    """Root of the visible per-workspace output tree (``jutul-agent-output/``).

    Named distinctly from the hidden ``.jutul-agent/`` config/env dir so the two
    don't read as the same folder.
    """
    return (workspace or workspace_root()) / WORKSPACE_OUTPUT_DIRNAME


# Session ids minted by ``session.default_session_id``: a sortable timestamp
# plus a short random suffix (e.g. ``2026-06-12-2315-3f2a``).
_DATED_SESSION_ID_RE = re.compile(r"^\d{4}-\d{2}-\d{2}-\d{4}-[0-9a-f]{4}$")


def is_dated_session_id(session_id: str) -> bool:
    """Whether ``session_id`` is a sortable timestamped id (vs a legacy UUID)."""
    return bool(_DATED_SESSION_ID_RE.match(session_id))


def session_output_dir(session_id: str, workspace: Path | None = None) -> Path:
    """Visible output directory for one session under the workspace.

    Layout: ``<workspace>/jutul-agent-output/sessions/<session_id>/`` for
    timestamped ids (the id itself sorts chronologically), or the legacy
    ``<YYYY-MM-DD>-<short_id>/`` for UUID sessions.

    This directory holds user-facing outputs (plots, transcripts, reports).
    Internal state (SQLite trace, REPL log) stays in ``workspace_state_dir``.
    """
    if is_dated_session_id(session_id):
        name = session_id
    else:
        name = f"{date.today().isoformat()}-{session_id[:8]}"
    return workspace_output_dir(workspace) / "sessions" / name


def workspace_memory_dir(workspace: Path | None = None) -> Path:
    """Per-workspace memory directory (index + per-fact note files).

    Lives alongside ``sessions/`` under the workspace's state dir so memory
    is automatically scoped to the workspace path the user invokes
    jutul-agent from. The agent maintains the contents via ``edit_file`` /
    ``write_file`` tools mounted at ``/memory/`` in the agent backend.
    """
    return workspace_state_dir(workspace) / "memory"
