"""Backend-agnostic abstraction over a persistent Julia runtime.

The Protocol stays minimal: ``eval``, ``reset``, and ``restart`` plus the async
context-manager lifecycle. Add methods only when a concrete caller needs them.
"""

from __future__ import annotations

from typing import Protocol

# The canonical result type lives with the kernel that produces it; re-exported
# here so the protocol and every consumer share one ``EvalResult`` identity.
from makker.juliakernel.result import EvalResult, OnChunk

__all__ = ["EvalResult", "JuliaSession", "OnChunk"]


class JuliaSession(Protocol):
    """A persistent Julia runtime accessible via async tool calls."""

    async def __aenter__(self) -> JuliaSession: ...

    async def __aexit__(self, *exc_info: object) -> None: ...

    async def eval(self, code: str, on_chunk: OnChunk | None = None) -> EvalResult:
        """Evaluate ``code`` and return its result.

        ``on_chunk`` (optional) receives output fragments live as the eval
        produces them, for callers that stream the output to a UI. The returned
        ``EvalResult`` still carries the full cleaned output regardless.
        """
        ...

    async def reset(self) -> EvalResult:
        """Respawn a fresh Julia worker, clearing all state.

        A cooperative reset that talks to the running runtime; backends that
        can't reset should still return an ``EvalResult`` rather than raising.
        """
        ...

    async def restart(self) -> None:
        """Force the runtime down and start fresh, without relying on it responding.

        The hard recovery for when an evaluation can't be interrupted and the
        runtime is wedged (the TUI's cancel path). Unlike ``reset``, it must not
        depend on the current session being responsive.
        """
        ...
