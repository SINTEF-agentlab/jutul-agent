"""Local Ollama integration: reachability, installed models, and streaming pull.

Thin async wrapper over the ``ollama`` package. The default host is
``http://localhost:11434`` (override with ``$OLLAMA_HOST``). If the server is
down, ``is_reachable`` is ``False`` and the checks return empty rather than
raising.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass

DEFAULT_HOST = "http://localhost:11434"

# Most context (KV cache) to allocate for a local model unless overridden; a
# memory cap, not a capability claim. The daemon's reported maximum can be far
# larger than fits in memory.
DEFAULT_NUM_CTX = 65536


def host() -> str:
    """Ollama server URL the client will talk to."""
    return os.environ.get("OLLAMA_HOST") or DEFAULT_HOST


def model_name(model_id: str) -> str:
    """The Ollama model tag from an ``ollama:<name>`` id (``<name>``)."""
    _, sep, rest = model_id.partition(":")
    return rest if sep else model_id


@dataclass(frozen=True)
class PullProgress:
    """One progress update from a streaming pull."""

    status: str
    fraction: float | None  # 0..1 when the layer reports total + completed


def _client():
    from ollama import AsyncClient

    return AsyncClient()


async def is_reachable() -> bool:
    """True if the Ollama server answers (and the package is importable)."""
    try:
        await _client().list()
        return True
    except Exception:
        return False


async def installed_models() -> list[str]:
    """Names (tags) of the models pulled locally; empty if the server is down."""
    names: list[str] = []
    with contextlib.suppress(Exception):
        resp = await _client().list()
        for model in getattr(resp, "models", None) or []:
            name = getattr(model, "model", None)
            if name is None and isinstance(model, dict):
                name = model.get("model")
            if name:
                names.append(name)
    return names


async def capabilities(name: str) -> list[str]:
    """Capability tags the daemon reports for a pulled model (e.g. ``tools``,
    ``vision``); empty on any error."""
    with contextlib.suppress(Exception):
        info = await _client().show(name)
        caps = getattr(info, "capabilities", None)
        if caps is None and isinstance(info, dict):
            caps = info.get("capabilities")
        if caps:
            return list(caps)
    return []


async def supports_tools(name: str) -> bool:
    """Whether the daemon exposes tool calling for ``name``."""
    return "tools" in await capabilities(name)


def thinks(name: str) -> bool:
    """Whether the daemon reports ``name`` as a thinking-capable model.

    Sync because it runs at model-build time; best-effort (False when the
    daemon can't answer).
    """
    with contextlib.suppress(Exception):
        from ollama import Client

        info = Client(timeout=2.0).show(name)
        caps = getattr(info, "capabilities", None)
        if caps is None and isinstance(info, dict):
            caps = info.get("capabilities")
        return "thinking" in (caps or [])
    return False


def context_window(name: str) -> int | None:
    """The model's max context length (tokens) the daemon reports, or None.

    Lets callers size ``num_ctx`` to the model rather than guess. Sync because it
    runs at model-build time; best-effort (None when the daemon can't answer).
    """
    with contextlib.suppress(Exception):
        from ollama import Client

        info = Client(timeout=2.0).show(name)
        modelinfo = getattr(info, "modelinfo", None)
        if modelinfo is None and isinstance(info, dict):
            modelinfo = info.get("modelinfo") or info.get("model_info")
        lengths = [
            value
            for key, value in (modelinfo or {}).items()
            if isinstance(value, int) and key.endswith("context_length")
        ]
        if lengths:
            return max(lengths)
    return None


def ctx_budget() -> int:
    """The context budget cap, overridable with ``$MAKKER_OLLAMA_NUM_CTX``."""
    try:
        return int(os.environ["MAKKER_OLLAMA_NUM_CTX"])
    except (KeyError, ValueError):
        return DEFAULT_NUM_CTX


def num_ctx(name: str) -> int:
    """Context window (tokens) to load ``name`` with: its reported max, capped
    at the budget; the budget itself when the daemon can't report a maximum.

    This is the *loaded* window — the single figure the rest of the app sizes
    against. The model is built with it, the auto-compaction trigger is a
    fraction of it, and ``/context`` measures against it. Keeping all three on
    this one value is what stops compaction from being scheduled past a window
    the model was never loaded with.
    """
    reported = context_window(name)
    budget = ctx_budget()
    return min(reported, budget) if reported else budget


def _matches(installed: str, name: str) -> bool:
    # Installed tags carry a version (e.g. "llama3.1:latest"); match a bare name
    # against the tag's base so "llama3.1" finds "llama3.1:latest".
    if installed == name:
        return True
    return ":" not in name and installed.split(":", 1)[0] == name


async def is_installed(name: str) -> bool:
    """True if ``name`` (a bare model name or full tag) is already pulled."""
    return any(_matches(installed, name) for installed in await installed_models())


async def pull(name: str) -> AsyncIterator[PullProgress]:
    """Stream a model pull, yielding ``PullProgress`` until it completes.

    Raises whatever the ``ollama`` client raises (e.g. a connection error).
    """
    stream = await _client().pull(name, stream=True)
    async for chunk in stream:
        status = getattr(chunk, "status", None) or ""
        total = getattr(chunk, "total", None)
        completed = getattr(chunk, "completed", None)
        fraction = completed / total if total and completed is not None else None
        yield PullProgress(status=status, fraction=fraction)
