"""The Session object: unit of work for one jutul-agent invocation."""

from __future__ import annotations

import contextlib
import re
import shutil
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from jutul_agent.julia.session import JuliaSession
from jutul_agent.paths import session_output_dir, workspace_state_dir
from jutul_agent.simulators.base import SimulatorAdapter
from jutul_agent.trace import TraceLog

TITLE_FILENAME = "title"
_SLUG_MAX_CHARS = 32
_TITLE_MAX_CHARS = 80


def default_session_id(now: datetime | None = None) -> str:
    """A sortable session id: minute-resolution timestamp + short random suffix.

    ``2026-06-12-2315-3f2a`` sorts chronologically in every directory listing;
    the suffix keeps two sessions started the same minute distinct.
    """
    stamp = (now or datetime.now()).strftime("%Y-%m-%d-%H%M")
    return f"{stamp}-{uuid.uuid4().hex[:4]}"


def _slugify_title(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    if len(slug) > _SLUG_MAX_CHARS:
        slug = slug[:_SLUG_MAX_CHARS].rsplit("-", 1)[0] or slug[:_SLUG_MAX_CHARS]
    return slug


def truncate_title(text: str, max_chars: int = _TITLE_MAX_CHARS) -> str:
    """Trim ``text`` to at most ``max_chars`` on a word boundary, with an ellipsis
    (counted within the limit) when it had to be cut."""
    if len(text) <= max_chars:
        return text
    # Leave room for the ellipsis so the result never exceeds max_chars, even when
    # the cut falls mid-word (no space to break on).
    head = text[: max_chars - 1]
    return (head.rsplit(" ", 1)[0] or head) + "…"


def derive_session_title(prompt: str) -> str:
    """A short human-readable title from the session's first prompt."""
    first_line = next((line.strip() for line in prompt.splitlines() if line.strip()), "")
    return truncate_title(re.sub(r"\s+", " ", first_line))


def read_session_title(state_dir: Path) -> str | None:
    """The stored title for a session state dir, if one was adopted."""
    path = state_dir / TITLE_FILENAME
    try:
        title = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return title or None


@dataclass(frozen=True)
class SessionInfo:
    """One resumable session on disk, as shown by listings and pickers."""

    session_id: str
    state_dir: Path
    title: str | None
    started: datetime


def _started_from_id(session_id: str) -> datetime | None:
    try:
        return datetime.strptime(session_id[:15], "%Y-%m-%d-%H%M")
    except ValueError:
        return None


def list_sessions(state_root: Path | None = None) -> list[SessionInfo]:
    """Every session under this workspace's state dir, newest first.

    The start time comes from the timestamped id when present (legacy UUID
    sessions fall back to the directory's mtime), so mixed listings still
    sort sensibly.
    """
    root = sessions_root(state_root)
    if not root.is_dir():
        return []
    infos: list[SessionInfo] = []
    for entry in root.iterdir():
        if not entry.is_dir() or not (entry / "trace.sqlite").exists():
            continue
        started = _started_from_id(entry.name)
        if started is None:
            started = datetime.fromtimestamp(entry.stat().st_mtime)
        infos.append(
            SessionInfo(
                session_id=entry.name,
                state_dir=entry,
                title=read_session_title(entry),
                started=started,
            )
        )
    infos.sort(key=lambda info: info.started, reverse=True)
    return infos


def resolve_session_id(text: str, *, state_root: Path | None = None) -> str | None:
    """Resolve an exact session id or a unique prefix to a stored session."""
    text = text.strip()
    if not text:
        return None
    ids = [info.session_id for info in list_sessions(state_root)]
    if text in ids:
        return text
    matches = [sid for sid in ids if sid.startswith(text)]
    return matches[0] if len(matches) == 1 else None


def sessions_root(state_root: Path | None = None) -> Path:
    """Where session subdirectories live.

    Defaults to ``$STATE_HOME/workspaces/<hash>/sessions/`` via
    ``workspace_state_dir()``. Tests can pass an explicit ``state_root``
    that holds ``sessions/`` directly.
    """
    base = state_root if state_root is not None else workspace_state_dir()
    return base / "sessions"


def session_dir(session_id: str, *, state_root: Path | None = None) -> Path:
    return sessions_root(state_root) / session_id


def last_session_path(state_root: Path | None = None) -> Path:
    base = state_root if state_root is not None else workspace_state_dir()
    return base / "last-session"


def read_last_session(state_root: Path | None = None) -> str | None:
    p = last_session_path(state_root)
    if not p.exists():
        return None
    sid = p.read_text(encoding="utf-8").strip()
    return sid or None


def write_last_session(session_id: str, *, state_root: Path | None = None) -> None:
    p = last_session_path(state_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(session_id, encoding="utf-8")


def _existing_output_dir(session_id: str) -> Path | None:
    """The session's existing output dir, accounting for an adopted title slug.

    ``adopt_title`` renames ``sessions/<sid>/`` to ``sessions/<sid>-<slug>/``, so a
    resumed session finds its folder by prefix. Both names can end up on disk (a
    stray empty ``<sid>/`` next to the real ``<sid>-<slug>/``); prefer the one that
    actually holds the session's output — its ``artifacts/`` for a plotting session,
    else any non-empty dir (a chat-only session keeps ``report.html`` at the root,
    not under ``artifacts/``) — so a resumed plot or report still resolves rather
    than the empty stray dir winning just because it sorts first.
    """
    base = session_output_dir(session_id)
    candidates: list[Path] = []
    if base.is_dir():
        candidates.append(base)
    if base.parent.is_dir():
        candidates += [m for m in sorted(base.parent.glob(base.name + "-*")) if m.is_dir()]
    if not candidates:
        return None

    def _has_artifacts(d: Path) -> bool:
        art = d / "artifacts"
        return art.is_dir() and any(art.iterdir())

    def _has_output(d: Path) -> bool:
        # Real output: a file at the root (e.g. report.html) or a non-empty subdir.
        # A stray dir whose only entry is an empty ``artifacts/`` must not count.
        return any(e.is_file() or (e.is_dir() and any(e.iterdir())) for e in d.iterdir())

    return (
        next((c for c in candidates if _has_artifacts(c)), None)
        or next((c for c in candidates if _has_output(c)), None)
        or candidates[0]
    )


def _ensure_jutul_agent_gitignore(output_dir: Path) -> None:
    """Drop a ``.gitignore`` at the root of ``<workspace>/jutul-agent-output/``
    so generated sessions, transcripts, and reports stay out of the user's repo.

    ``output_dir`` is ``<workspace>/jutul-agent-output/sessions/<date>-<sid>/``;
    the gitignore goes two levels up at
    ``<workspace>/jutul-agent-output/.gitignore``.
    """
    root = output_dir.parent.parent
    gitignore = root / ".gitignore"
    if gitignore.exists():
        return
    gitignore.write_text("*\n", encoding="utf-8")


@dataclass
class Session:
    """A live jutul-agent session. Construct via ``Session.create``.

    Direct construction is supported for tests that want to wire a Session
    around a pre-built trace, but production code should always go through
    ``create`` so the on-disk layout and the ``session_start`` lifecycle
    event are guaranteed.
    """

    julia: JuliaSession
    state_dir: Path
    output_dir: Path
    trace: TraceLog
    simulator: SimulatorAdapter
    session_id: str
    ephemeral_memory: bool = False
    # Whether plot_julia may open a live Makie window for the user (interactive
    # session with a display). Headless and one-shot runs render offscreen to a file.
    open_windows: bool = False
    # Human-readable title derived from the first prompt (see ``adopt_title``).
    title: str | None = None
    # Whether this session continues an earlier conversation. The thread state
    # is restored from the checkpointer; the Julia REPL is not.
    resumed: bool = False
    # The front end driving this session ("tui" or "web"); tools that only receive
    # the session (e.g. a capability's ToolFactory) read this to match the surface
    # ``build_agent`` was composed for.
    surface: str = "tui"
    _ephemeral_memory_dir: Path | None = field(default=None, repr=False)
    # Folders holding a report written this session, whose sidecar transcript is
    # refreshed at turn end (see ``refresh_report_transcripts``).
    _report_transcript_dirs: set[Path] = field(default_factory=set, repr=False)

    @classmethod
    def create(
        cls,
        *,
        julia: JuliaSession,
        simulator: SimulatorAdapter,
        session_id: str | None = None,
        state_root: Path | None = None,
        ephemeral_memory: bool = False,
        open_windows: bool = False,
        surface: str = "tui",
    ) -> Session:
        sid = session_id or default_session_id()
        dir_ = session_dir(sid, state_root=state_root)
        dir_.mkdir(parents=True, exist_ok=True)

        out_dir = session_output_dir(sid)
        try:
            (out_dir / "artifacts").mkdir(parents=True, exist_ok=True)
            _ensure_jutul_agent_gitignore(out_dir)
        except OSError:
            out_dir = dir_  # fall back to state_dir if workspace is not writable

        trace = TraceLog(dir_ / "trace.sqlite")
        trace.append(
            "session_start",
            {"session_id": sid, "simulator": simulator.name},
        )
        # The agent builder seeds the memory index when it mounts the dir.
        ephemeral_dir = (
            Path(tempfile.mkdtemp(prefix="jutul-agent-ephemeral-")) if ephemeral_memory else None
        )
        return cls(
            julia=julia,
            state_dir=dir_,
            output_dir=out_dir,
            trace=trace,
            simulator=simulator,
            session_id=sid,
            ephemeral_memory=ephemeral_memory,
            open_windows=open_windows,
            surface=surface,
            _ephemeral_memory_dir=ephemeral_dir,
        )

    @classmethod
    def resume(
        cls,
        *,
        julia: JuliaSession,
        simulator: SimulatorAdapter,
        session_id: str,
        state_root: Path | None = None,
        ephemeral_memory: bool = False,
        open_windows: bool = False,
        surface: str = "tui",
    ) -> Session:
        """Reopen an earlier session: same id, trace, and output folder.

        The conversation itself comes back through the per-session
        checkpointer (the thread key is the session id); this restores the
        on-disk identity around it. The Julia kernel is the caller's fresh
        instance; REPL state does not survive across processes.
        """
        dir_ = session_dir(session_id, state_root=state_root)
        if not (dir_ / "trace.sqlite").exists():
            raise FileNotFoundError(f"no session trace at {dir_}")

        out_dir = _existing_output_dir(session_id) or session_output_dir(session_id)
        try:
            (out_dir / "artifacts").mkdir(parents=True, exist_ok=True)
            _ensure_jutul_agent_gitignore(out_dir)
        except OSError:
            out_dir = dir_

        trace = TraceLog(dir_ / "trace.sqlite")
        trace.append(
            "session_resume",
            {"session_id": session_id, "simulator": simulator.name},
        )
        ephemeral_dir = (
            Path(tempfile.mkdtemp(prefix="jutul-agent-ephemeral-")) if ephemeral_memory else None
        )
        return cls(
            julia=julia,
            state_dir=dir_,
            output_dir=out_dir,
            trace=trace,
            simulator=simulator,
            session_id=session_id,
            ephemeral_memory=ephemeral_memory,
            open_windows=open_windows,
            surface=surface,
            title=read_session_title(dir_),
            resumed=True,
            _ephemeral_memory_dir=ephemeral_dir,
        )

    def memory_dir(self, *, workspace_memory: Path) -> Path:
        """Resolved memory directory for this session."""
        if self.ephemeral_memory and self._ephemeral_memory_dir is not None:
            return self._ephemeral_memory_dir
        return workspace_memory

    def adopt_title(self, prompt: str) -> None:
        """Derive the session title from its first prompt and adopt it.

        Stores the title beside the trace (for session listings), records it
        as a trace event, and renames the *output* directory to carry a slug
        so result folders read like the work they hold. The state directory
        and ``session_id`` never change: open SQLite handles, the ``/session/``
        mount, and the checkpointer thread key all point there. Best-effort
        and idempotent; a failed rename just keeps the plain name.
        """
        if self.title is not None:
            return
        title = derive_session_title(prompt)
        if not title:
            return
        self.title = title
        with contextlib.suppress(OSError):
            (self.state_dir / TITLE_FILENAME).write_text(title + "\n", encoding="utf-8")
        self.trace.append("session_title", {"session_id": self.session_id, "title": title})

        slug = _slugify_title(title)
        if not slug or self.output_dir == self.state_dir:
            return
        target = self.output_dir.with_name(f"{self.output_dir.name}-{slug}")
        try:
            self.output_dir.rename(target)
        except OSError:
            return
        self.output_dir = target

    def retitle(self, title: str) -> None:
        """Replace the session title after it was first adopted (e.g. an LLM one).

        Unlike ``adopt_title`` this overwrites an existing title: it updates the
        title file that session listings read and records a trace event. The
        output-directory slug is deliberately left as the first-prompt one, so
        nothing on disk has to be renamed (and no open handles are disturbed);
        only the displayed name changes. Best-effort and a no-op for empty input.
        """
        title = re.sub(r"\s+", " ", title).strip()
        if not title or title == self.title:
            return
        self.title = title
        with contextlib.suppress(OSError):
            (self.state_dir / TITLE_FILENAME).write_text(title + "\n", encoding="utf-8")
        self.trace.append("session_title", {"session_id": self.session_id, "title": title})

    def note_report(self, report_path: Path) -> None:
        """Remember a report so its sidecar transcript is refreshed at turn end.

        ``write_report`` runs mid-turn, before the model's closing message, so
        the transcript it writes beside the report is a snapshot. Recording the
        folder here lets the turn loop rewrite it from the complete trace once
        the turn settles.
        """
        self._report_transcript_dirs.add(Path(report_path).parent)

    def refresh_report_transcripts(self) -> None:
        """Rewrite the transcript beside each report written this session.

        Called when a turn settles (no pending interrupts), so the linked
        transcript includes the model's closing message. Best-effort: a write
        failure must never disturb the turn loop.
        """
        if not self._report_transcript_dirs:
            return
        from jutul_agent.transcript.report import write_sidecar_transcript

        events = list(self.trace.iter_events())
        for directory in self._report_transcript_dirs:
            with contextlib.suppress(OSError):
                write_sidecar_transcript(directory, events)

    def finalize(self) -> None:
        self.trace.append("session_end", {"session_id": self.session_id})
        self.trace.close()
        if self.ephemeral_memory and self._ephemeral_memory_dir is not None:
            shutil.rmtree(self._ephemeral_memory_dir, ignore_errors=True)
            self._ephemeral_memory_dir = None
