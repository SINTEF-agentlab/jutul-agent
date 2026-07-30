"""Filesystem backend for the agent's workspace.

``WorkspaceShellBackend`` is the real-path default: it reads and writes the real
filesystem from the workspace, refuses writes into read-only roots (installed
package source in the shared Julia depot), and blocks shell ``julia``.
``RecursiveGrepBackend`` is the top-level composite, fixing a grep glob that
otherwise silently skips subdirectories.
"""

from __future__ import annotations

from pathlib import Path

from deepagents.backends import CompositeBackend, LocalShellBackend
from deepagents.backends.protocol import (
    DeleteResult,
    EditResult,
    ExecuteResponse,
    GlobResult,
    GrepResult,
    WriteResult,
)

from jutul_agent.agent.windows_paths import split_windows_glob

_BLOCKED_INTERPRETERS = frozenset({"julia"})


def interpreter_invocation(command: str, names: frozenset[str] | tuple[str, ...]) -> str | None:
    """Name of the interpreter a shell command launches, or ``None``.

    Looks at the executable position of each pipeline/sequence segment, so
    ``julia -e ...`` and ``echo x | julia`` match while an interpreter name
    appearing as data (``ls ~/.julia/...``, ``grep julia src/``) does not.
    Shared by the workspace backend's execute guard (which blocks only
    ``julia``, the session kernel exists precisely for it) and the bench
    scorers that audit recorded execute calls.
    """
    import re

    for segment in re.split(r"[;|&]+|\$\(", command):
        head = segment.strip().split()
        if head and Path(head[0]).name in names:
            return Path(head[0]).name
    return None


_NO_JULIA_SHELL_MSG = (
    "Error: spawning `julia` through the shell is disabled. Julia code runs in "
    "the persistent session via `run_julia`: shared state, streamed output, no "
    'cold start. Use `run_julia` for code and `include("file.jl")` for scripts.'
)

_READ_ONLY_MSG = (
    "Error: '{path}' is read-only installed package source in the shared Julia "
    "depot. Read and grep it freely (it's reference material), but don't edit "
    "it: the depot is shared across projects. Write your own code in the "
    "workspace, or to change the package itself, `Pkg.develop` it "
    "(jutul-agent init --source-path ...) and edit that checkout."
)


def _recursive_glob(glob: str | None) -> str | None:
    """Make a slash-free grep filter recursive.

    deepagents matches a bare ``*.jl`` glob only against files directly in the
    searched directory, so a type-filtered grep silently skips subdirectories
    (e.g. a package's ``src/ext/``, where Julia keeps extension code). ripgrep and
    the agent's expectation treat it as recursive, so rewrite ``*.jl`` to
    ``**/*.jl``. Patterns that already carry a path (``src/*.jl``, ``**/*.jl``)
    are left alone.
    """

    if glob and "/" not in glob:
        return f"**/{glob}"
    return glob


class RecursiveGrepBackend(CompositeBackend):
    """Top-level composite whose grep recurses on a bare ``*.ext`` filter.

    See :func:`_recursive_glob`. Normalizing here, in the one backend the grep
    tool calls, fixes every route the composite forwards to, since it passes the
    same glob down to whichever sub-backend resolves the path.

    It also rewrites a Windows drive-absolute ``glob`` pattern into a relative
    pattern plus a base path (see :func:`split_windows_glob`): ``pathlib``'s
    ``rglob`` rejects absolute patterns, so the absolute ``glob`` form the skills
    use silently matches nothing on Windows otherwise. A no-op off Windows.

    The grep overrides forward ``**kwargs`` because only ``glob`` needs
    normalizing: the middleware probes the signature for the search options it
    supports (a match cap, say) and silently falls back to trimming the full
    result set when an override doesn't accept them, so a fixed signature would
    quietly disable bounded search on every new option the framework adds.
    """

    def grep(
        self, pattern: str, path: str | None = None, glob: str | None = None, **kwargs
    ) -> GrepResult:
        return super().grep(pattern, path=path, glob=_recursive_glob(glob), **kwargs)

    async def agrep(
        self, pattern: str, path: str | None = None, glob: str | None = None, **kwargs
    ) -> GrepResult:
        return await super().agrep(pattern, path=path, glob=_recursive_glob(glob), **kwargs)

    def glob(self, pattern: str, path: str | None = None) -> GlobResult:
        pattern, path = split_windows_glob(pattern, path)
        return super().glob(pattern, path=path)

    async def aglob(self, pattern: str, path: str | None = None) -> GlobResult:
        pattern, path = split_windows_glob(pattern, path)
        return await super().aglob(pattern, path=path)


class WorkspaceShellBackend(LocalShellBackend):
    """Workspace default backend: real paths, with a guard against shell ``julia``.

    Constructed with ``virtual_mode=False`` so the file tools speak the real
    filesystem: a relative path resolves against the workspace (the backend's
    ``cwd``) and an absolute path as itself, the same way ``run_julia`` and
    ``execute`` resolve paths from that working directory. One path string
    therefore names one file in the file tools, the shell, and the REPL. Package
    source, skills, memory, and added folders are all read and written at their
    real paths through this one backend.

    ``readonly_roots`` are real directories writes are refused under: the
    shared Julia depot's installed package source. Reads and greps there are
    fine (that's how the agent studies a package); ``write``, ``edit`` and
    ``delete`` are blocked, so the agent can't corrupt the depot for other
    projects. A ``Pkg.develop`` checkout lives outside the depot and stays
    writable, so the registry-vs-dev distinction falls out of the path, with no
    special-casing.

    ``execute`` refuses to launch ``julia``: Julia belongs in the session
    kernel, where state persists and output streams; a shell julia is a cold
    process that shares nothing with the session. A prompt rule alone does not
    reliably stop models from shelling out; the tool guaranteeing it does.
    Other interpreters are deliberately not blocked: shell python and friends
    are part of general competence, and scientific results are kept honest by
    the bench's trace checks, not by a blanket ban.
    """

    def __init__(self, *args, readonly_roots: tuple[Path, ...] = (), **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._readonly_roots = tuple(Path(root).resolve() for root in readonly_roots)

    def _readonly_reason(self, file_path: str) -> str | None:
        """Corrective message if ``file_path`` lands inside a read-only root, else ``None``."""
        if not self._readonly_roots:
            return None
        path = Path(file_path)
        target = path if path.is_absolute() else self.cwd / path
        try:
            target = target.resolve()
        except OSError:
            return None
        for root in self._readonly_roots:
            if target == root or root in target.parents:
                return _READ_ONLY_MSG.format(path=file_path)
        return None

    def write(self, file_path: str, content: str) -> WriteResult:
        reason = self._readonly_reason(file_path)
        if reason is not None:
            return WriteResult(error=reason)
        return super().write(file_path, content)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        reason = self._readonly_reason(file_path)
        if reason is not None:
            return EditResult(error=reason)
        return super().edit(file_path, old_string, new_string, replace_all=replace_all)

    def delete(self, file_path: str) -> DeleteResult:
        # Deleting a directory removes it recursively, so the depot guard matters
        # here more than for write/edit: one call could take out a package's
        # whole source tree.
        reason = self._readonly_reason(file_path)
        if reason is not None:
            return DeleteResult(error=reason)
        return super().delete(file_path)

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        if interpreter_invocation(command, _BLOCKED_INTERPRETERS) is not None:
            return ExecuteResponse(output=_NO_JULIA_SHELL_MSG, exit_code=2)
        return super().execute(command, timeout=timeout)

    async def aexecute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        if interpreter_invocation(command, _BLOCKED_INTERPRETERS) is not None:
            return ExecuteResponse(output=_NO_JULIA_SHELL_MSG, exit_code=2)
        return await super().aexecute(command, timeout=timeout)
