"""Holds the running sessions and stands new ones up on request.

The manager maps a session id to its ``SessionHost``. New sessions are built by
a *host factory*, which defaults to standing up a real session from the
simulator registry. Tests inject a factory that returns a host wrapping fakes,
so the server can be exercised without a Julia kernel or a model.

Each live host holds a Julia kernel (a real OS process, hundreds of MB) and a
SQLite checkpointer open, so the registry is bounded: it keeps the most recently
created/resumed sessions and closes the rest (LRU). Without this, browsing back
through history — each resume stands up a fresh host — would accumulate kernels
until the server stops. A closed session stays fully resumable; only its live
kernel is reclaimed.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jutul_agent.session_host import SessionHost

if TYPE_CHECKING:
    from jutul_agent.agent.capabilities import Capability

HostFactory = Callable[..., Awaitable[SessionHost]]

# How many live sessions (and so Julia kernels) to keep at once. A single user
# works in one session at a time; a few extra cover quickly flipping between
# recent chats. The oldest beyond this are closed (and stay resumable on disk).
#
# The cap exists because each live session holds a Julia process, which for a
# reservoir workload is one to two gigabytes resident before it simulates
# anything. Raise it with JUTUL_AGENT_MAX_LIVE_SESSIONS when there is memory to
# spare and you flip between more chats than this; the cost of being wrong is
# only that a resume restarts Julia.
DEFAULT_MAX_LIVE = 4


def _configured_max_live() -> int:
    """``DEFAULT_MAX_LIVE``, or the env override when it is a positive integer.

    A bad value is ignored rather than fatal: the server starting with the default
    beats it refusing to start over a typo in an optional tuning knob.
    """
    raw = os.environ.get("JUTUL_AGENT_MAX_LIVE_SESSIONS", "").strip()
    if not raw:
        return DEFAULT_MAX_LIVE
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MAX_LIVE
    return value if value > 0 else DEFAULT_MAX_LIVE


class SessionBusyError(RuntimeError):
    """An operation needed an idle session, but a connection is attached to it."""


@dataclass(frozen=True)
class SessionLaunchDefaults:
    """Server-wide knobs applied to every session a host factory stands up.

    These are fixed when ``jutul-agent web`` launches (one folder, one Julia
    environment), unlike the model and approval policy which a request can
    override per session. Empty/None means "use the session's own default".
    """

    julia_project: Path | None = None
    threads: str | None = None
    add_dirs: tuple[Path, ...] = ()
    ephemeral_memory: bool = False
    sysimage: bool | None = None


def make_host_factory(defaults: SessionLaunchDefaults | None = None) -> HostFactory:
    """A real host factory that bakes in the server's launch defaults.

    The manager stays generic (it forwards only the per-request fields); the
    launch-wide knobs ride along in this closure, so neither the manager nor a
    test factory needs to know about them.
    """
    launch = defaults or SessionLaunchDefaults()

    async def factory(
        *,
        sim: str,
        model: str | None,
        approval_mode: str | None,
        workspace: Any | None,
        resume: bool,
        session_id: str | None,
        extensions: Sequence[Capability],
    ) -> SessionHost:
        from jutul_agent.simulators import registry

        adapter = registry.get(sim)
        return await SessionHost.start(
            simulator=adapter,
            surface="web",
            model=model,
            approval_mode=approval_mode,
            workspace=workspace,
            resume=resume,
            session_id=session_id,
            extensions=extensions,
            julia_project=launch.julia_project,
            threads=launch.threads,
            add_dirs=launch.add_dirs,
            ephemeral_memory=launch.ephemeral_memory,
            sysimage=launch.sysimage,
        )

    return factory


_default_host_factory = make_host_factory()


class SessionManager:
    """Registry of running ``SessionHost``s, keyed by session id."""

    def __init__(
        self, *, host_factory: HostFactory | None = None, max_live: int | None = None
    ) -> None:
        self._host_factory = host_factory or _default_host_factory
        # Insertion order is recency: a (re)registered host moves to the end, so
        # the front is the least recently created/resumed and is evicted first.
        self._hosts: OrderedDict[str, SessionHost] = OrderedDict()
        self._lock = asyncio.Lock()
        self._max_live = max(1, _configured_max_live() if max_live is None else max_live)

    async def create(
        self,
        *,
        sim: str,
        model: str | None = None,
        approval_mode: str | None = None,
        workspace: Any | None = None,
        extensions: Sequence[Capability] = (),
    ) -> SessionHost:
        host = await self._host_factory(
            sim=sim,
            model=model,
            approval_mode=approval_mode,
            workspace=workspace,
            resume=False,
            session_id=None,
            extensions=extensions,
        )
        await self._register(host)
        return host

    async def resume(
        self,
        session_id: str,
        *,
        sim: str,
        model: str | None = None,
        approval_mode: str | None = None,
        workspace: Any | None = None,
        extensions: Sequence[Capability] = (),
    ) -> SessionHost:
        host = await self._host_factory(
            sim=sim,
            model=model,
            approval_mode=approval_mode,
            workspace=workspace,
            resume=True,
            session_id=session_id,
            extensions=extensions,
        )
        await self._register(host)
        return host

    async def _register(self, host: SessionHost) -> None:
        """Add ``host`` as the most-recent session, then close any it displaces.

        A host already registered under this id (a re-resume) is replaced and the
        stale one closed, and anything beyond ``max_live`` is evicted oldest-first
        — skipping any host a client is still connected to, so an in-use kernel is
        never torn down mid-turn (if every live host is attached, the cap is exceeded
        rather than killing an active one). Kernels are torn down outside the lock so
        a slow shutdown can't block other sessions; eviction never raises (a closed
        kernel stays resumable on disk).
        """
        to_close: list[SessionHost] = []
        async with self._lock:
            stale = self._hosts.pop(host.session_id, None)
            if stale is not None and stale is not host:
                to_close.append(stale)
            self._hosts[host.session_id] = host  # newest → most-recently-used end
            while len(self._hosts) > self._max_live:
                victim = next(
                    (sid for sid, h in self._hosts.items() if not h.attached and h is not host),
                    None,
                )
                if victim is None:  # everything else is in use; keep them all
                    break
                to_close.append(self._hosts.pop(victim))
        for old in to_close:
            with contextlib.suppress(Exception):
                await old.aclose()

    def get(self, session_id: str) -> SessionHost | None:
        return self._hosts.get(session_id)

    async def acquire(self, session_id: str) -> SessionHost | None:
        """Claim a live host for a new connection, atomically.

        Promotes it (so a concurrent create/resume can't evict it out from under the
        connection) and attaches it, both under the lock — eviction skips attached
        hosts, so doing this in one locked step closes the window where a host fetched
        for a socket gets torn down before the socket attaches. Returns ``None`` if the
        session is not live (evicted or never created) or is already attached by
        another connection.
        """
        async with self._lock:
            host = self._hosts.get(session_id)
            if host is None or host.attached:
                return None
            host.attach()
            self._hosts.move_to_end(session_id)
            return host

    def promote(self, session_id: str) -> None:
        """Mark a live host most-recently-used, so reattaching to it protects it
        from eviction the same way a fresh create/resume would."""
        if session_id in self._hosts:
            self._hosts.move_to_end(session_id)

    def list_ids(self) -> list[str]:
        return list(self._hosts)

    async def close(self, session_id: str, *, require_idle: bool = False) -> bool:
        """Close and unregister a session; ``False`` if it was not registered.

        With ``require_idle`` set, refuse (raise ``SessionBusyError``) when a connection
        is attached, so a stray delete can't tear a kernel down under a live turn. The
        attached check and the pop are one atomic step under the lock, with no
        TOCTOU gap for a connection to attach in between.
        """
        async with self._lock:
            host = self._hosts.get(session_id)
            if host is None:
                return False
            if require_idle and host.attached:
                raise SessionBusyError(session_id)
            del self._hosts[session_id]
        await host.aclose()
        return True

    async def aclose(self) -> None:
        """Close every running session (used on server shutdown).

        One session's teardown raising (e.g. a kernel already gone) must not abort
        the loop and leave the other kernels orphaned, so each close is isolated.
        """
        for session_id in self.list_ids():
            with contextlib.suppress(Exception):
                await self.close(session_id)
