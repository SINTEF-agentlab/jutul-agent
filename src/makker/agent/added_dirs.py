"""Track additional working directories the user adds to the session.

The agent's filesystem uses real paths (see ``agent.builder.build_backend``), so
an added folder is already reachable at its absolute path by every tool:
``read_file``, ``grep``, ``write_file``, ``execute``, and ``run_julia`` alike.
Adding a folder just validates the path and records it on the live backend so
the session can list what was added (via the ``--add-dir`` flag at launch or the
TUI ``/add-dir`` command). Records are session-scoped: they live until the
process exits and are not persisted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from makker.paths import workspace_root

# Collapse anything outside this set to a single dash for a readable short name.
_UNSAFE_NAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


class AddDirError(ValueError):
    """A folder could not be added (missing path, not a directory, ...)."""


@dataclass(frozen=True)
class AddedDir:
    """A folder the user added to the session, used at its real ``path``."""

    name: str
    path: Path


def _slugify(path: Path) -> str:
    """A safe, human-readable short name derived from the folder's basename."""

    base = _UNSAFE_NAME_CHARS.sub("-", path.name).strip("-._")
    return base or "dir"


def _unique_name(taken: set[str], desired: str) -> str:
    """``desired`` if free, else ``desired-2``, ``desired-3``, ... so two
    folders that share a basename get distinct names."""

    if desired not in taken:
        return desired
    index = 2
    while f"{desired}-{index}" in taken:
        index += 1
    return f"{desired}-{index}"


def resolve_dir(raw: str | Path, *, workspace: Path | None = None) -> Path:
    """Resolve a user-supplied folder to an absolute, existing directory.

    ``~`` and relative paths are resolved against the workspace root (the
    user's launch directory), matching how they'd read the path at the prompt.
    Raises ``AddDirError`` if the path is missing or not a directory.
    """

    base = (workspace or workspace_root()).resolve()
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    candidate = candidate.resolve()
    if not candidate.exists():
        raise AddDirError(f"no such directory: {candidate}")
    if not candidate.is_dir():
        raise AddDirError(f"not a directory: {candidate}")
    return candidate


def _registry(backend) -> list[AddedDir]:
    """The mutable list of added folders, stored on the live backend."""
    dirs = getattr(backend, "_added_dirs", None)
    if dirs is None:
        dirs = []
        backend._added_dirs = dirs
    return dirs


def added_dirs(backend) -> list[AddedDir]:
    """The folders added to this session, sorted by name."""
    return sorted(_registry(backend), key=lambda entry: entry.name)


def add_dir(
    backend,
    raw: str | Path,
    *,
    workspace: Path | None = None,
) -> AddedDir:
    """Validate ``raw`` and record it as an added folder; return its ``AddedDir``.

    The real-path backend already reads and writes any real path, so the agent
    uses the folder's absolute path directly; recording it just lets the session
    list and track what was added. Idempotent (an already-added folder returns
    its existing record). The workspace itself is rejected because it is already
    the working directory. Raises ``AddDirError`` for a missing or non-directory path.
    """

    ws = (workspace or workspace_root()).resolve()
    path = resolve_dir(raw, workspace=ws)
    if path == ws:
        raise AddDirError("the workspace is already the working directory, no need to add it")

    registry = _registry(backend)
    for entry in registry:
        if entry.path == path:
            return entry

    name = _unique_name({entry.name for entry in registry}, _slugify(path))
    entry = AddedDir(name=name, path=path)
    registry.append(entry)
    return entry
