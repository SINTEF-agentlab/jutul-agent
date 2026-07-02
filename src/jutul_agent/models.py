"""Model catalog and provider metadata.

The in-app selector is built by *discovery*: ``discover_models()`` reads the
tool-calling models each installed provider package ships in its bundled
``data/_profiles.py`` (the same metadata LangChain uses), so new models appear
when the provider package updates; no hardcoded list to maintain. Ollama has
no static profiles and is discovered from the daemon by the selector. Any
``provider:model`` string ``init_chat_model`` accepts also works via free-text.

``PROVIDERS`` holds the per-provider API-key variable and the pip package that
supplies the LangChain integration.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProviderInfo:
    name: str
    label: str
    package: str
    key_env_var: str | None = None
    local: bool = False


@dataclass(frozen=True)
class ModelInfo:
    id: str
    label: str
    note: str = ""

    @property
    def provider(self) -> str:
        return provider_of(self.id)


# The agent's default model and the env var that overrides it. They live here, in a
# light module, so the CLI (help text, version, doctor) can read them without
# importing the agent builder, which pulls the whole LangChain/deepagents stack.
DEFAULT_MODEL = "openai:gpt-5.4-mini"
MODEL_ENV_VAR = "JUTUL_AGENT_MODEL"


PROVIDERS: dict[str, ProviderInfo] = {
    "openai": ProviderInfo("openai", "OpenAI", "langchain-openai", "OPENAI_API_KEY"),
    "anthropic": ProviderInfo("anthropic", "Anthropic", "langchain-anthropic", "ANTHROPIC_API_KEY"),
    "google_genai": ProviderInfo(
        "google_genai", "Google", "langchain-google-genai", "GOOGLE_API_KEY"
    ),
    "ollama": ProviderInfo("ollama", "Ollama (local)", "langchain-ollama", local=True),
}


def provider_of(model_id: str) -> str:
    prefix, sep, _ = model_id.partition(":")
    return prefix if sep else ""


def provider_info(model_id: str) -> ProviderInfo | None:
    return PROVIDERS.get(provider_of(model_id))


def key_env_var(model_id: str) -> str | None:
    info = provider_info(model_id)
    return info.key_env_var if info is not None else None


def is_local(model_id: str) -> bool:
    info = provider_info(model_id)
    return bool(info and info.local)


def is_ollama_cloud(model_id: str) -> bool:
    """Ollama-hosted (``:cloud``) models run remotely; no local pull needed."""
    return provider_of(model_id) == "ollama" and model_id.endswith(":cloud")


def resolve_model(
    explicit: str | None = None,
    *,
    workspace_model: str | None = None,
    user_model: str | None = None,
) -> str:
    """Resolve the model id by precedence, highest first.

    ``--model`` (``explicit``) > workspace config > user-global config >
    ``$JUTUL_AGENT_MODEL`` (a dev/CI override) > ``DEFAULT_MODEL``. Lives here
    (not in the agent builder) so lightweight commands like ``doctor`` and
    ``init`` can resolve a model without importing the agent stack.
    """
    import os

    return (
        explicit or workspace_model or user_model or os.environ.get(MODEL_ENV_VAR) or DEFAULT_MODEL
    )


def model_profile(model_id: str) -> dict:
    """The provider package's bundled profile for the model (``{}`` when unknown).

    Profiles are the maintained capability source: keying decisions on them
    means new models are covered by upgrading the provider package, never by
    editing a list here. Reading one builds the model, which needs the provider
    key; any failure (no key, offline, unknown model) reads as an empty profile.
    """
    try:
        from langchain.chat_models import init_chat_model

        return init_chat_model(model_id).profile or {}
    except Exception:
        return {}


def context_window(model_id: str) -> int | None:
    """Input-context size in tokens for ``model_id``, best effort.

    The provider package's bundled profile data answers first (reached by
    building the model, which needs the provider key the session already
    has). Providers whose models the data cannot cover have a live fallback
    in ``_WINDOW_FALLBACKS``: the Ollama daemon reports the loaded model (capped
    at the memory budget), and the Gemini API covers models newer than the
    bundled data. ``None`` when no source can answer — callers should degrade to
    absolute counts.
    """
    value = model_profile(model_id).get("max_input_tokens")
    if isinstance(value, int) and value > 0:
        return value
    fallback = _WINDOW_FALLBACKS.get(provider_of(model_id))
    return fallback(model_id) if fallback else None


def _ollama_window(model_id: str) -> int | None:
    from jutul_agent import ollama_client

    # The *loaded* window (capped at the memory budget), not the daemon's
    # theoretical maximum: the model is built with this, so the context figure
    # and the auto-compaction trigger must agree with it.
    return ollama_client.num_ctx(ollama_client.model_name(model_id))


def _google_context_window(name: str) -> int | None:
    """The model's input limit from the Gemini API, for models the installed
    provider package has no profile entry for yet.

    Two attempts: the SDK's transport occasionally fails transiently on a
    fresh client, and this lookup runs once per session.
    """
    import contextlib
    import os

    key = os.environ.get("GOOGLE_API_KEY")
    if not key or not name:
        return None
    for _ in range(2):
        with contextlib.suppress(Exception):
            from google.genai import Client

            with Client(api_key=key) as client:
                limit = client.models.get(model=name).input_token_limit
            return int(limit) if limit else None
    return None


# Live context-window lookups per provider, for models the bundled profile
# data cannot answer. Late-bound lambdas so the underlying helpers stay
# patchable in tests; providers without an entry simply report no window.
_WINDOW_FALLBACKS: dict[str, Callable[[str], int | None]] = {
    "ollama": lambda model_id: _ollama_window(model_id),
    "google_genai": lambda model_id: _google_context_window(model_id.partition(":")[2]),
}


# Ollama ships no profile data, so unlike the cloud providers it has no
# discoverable catalog; a short hand-maintained list is the only way to offer
# browse-and-pull. This is the one curated list we keep; edit it as the local
# landscape moves. Local tags are pulled on first use; cloud models are hosted
# by Ollama (`ollama signin`) and switch directly.
#
# Qwen3.6 is the current sweet spot for local agentic coding (Apache 2.0): the
# 27B dense leads its size class on agentic-coding benchmarks (~17 GB at Q4),
# and the 35B-A3B MoE (~3B active) decodes far faster with long context, good
# for batched eval rollouts. NB: all open-weight models are Python-centric, so
# Julia/JutulDarcy tool-use is their weak axis; validate on real tasks.
RECOMMENDED_OLLAMA_LOCAL: tuple[str, ...] = (
    "qwen3.6:27b",
    "qwen3.6:35b-a3b",
)
# Frontier-tier models hosted by Ollama; far better at tool use than small
# local models.
OLLAMA_CLOUD: tuple[str, ...] = (
    "deepseek-v4-flash:cloud",
    "glm-5.1:cloud",
    "kimi-k2.6:cloud",
    "minimax-m2.7:cloud",
)


@lru_cache(maxsize=1)
def discover_models() -> dict[str, list[ModelInfo]]:
    """Tool-calling models from installed provider packages, grouped by provider.

    Each non-local provider's ``data/_profiles.py`` is read and filtered to
    models that support tool calling and text I/O. Providers are ordered as in
    ``PROVIDERS``; models within a provider are sorted newest-ish first (reverse
    name order, which tracks the version-in-name scheme of the major providers).
    Cached for the process; call ``discover_models.cache_clear()`` to refresh.
    """
    grouped: dict[str, list[ModelInfo]] = {}
    for name, info in PROVIDERS.items():
        if info.local:
            continue  # Ollama ships no profiles; the selector probes the daemon.
        names = sorted(
            (m for m, profile in _load_profiles(info.package).items() if _is_chat_capable(profile)),
            reverse=True,
        )
        if names:
            grouped[name] = [ModelInfo(f"{name}:{m}", m) for m in names]
    return grouped


def is_known_model(model_id: str) -> bool:
    """Whether ``model_id`` is in the discovered catalog (vs free-text)."""
    return any(m.id == model_id for m in discover_models().get(provider_of(model_id), ()))


def _is_chat_capable(profile: dict[str, Any]) -> bool:
    return (
        bool(profile.get("tool_calling"))
        and profile.get("text_inputs", True) is not False
        and profile.get("text_outputs", True) is not False
    )


def _load_profiles(dist_name: str) -> dict[str, Any]:
    """Load ``_PROFILES`` from a provider package's ``data/_profiles.py``.

    Best-effort: a missing package or unreadable module yields an empty dict so
    discovery degrades to "that provider contributes nothing" rather than failing.
    """
    module_root = dist_name.replace("-", "_")
    spec = importlib.util.find_spec(module_root)
    if spec is None:
        return {}
    if spec.origin:
        base = Path(spec.origin).parent
    elif spec.submodule_search_locations:
        base = Path(next(iter(spec.submodule_search_locations)))
    else:
        return {}
    path = base / "data" / "_profiles.py"
    if not path.exists():
        return {}
    file_spec = importlib.util.spec_from_file_location(f"{module_root}.data._profiles", path)
    if file_spec is None or file_spec.loader is None:
        return {}
    module = importlib.util.module_from_spec(file_spec)
    try:
        file_spec.loader.exec_module(module)
    except Exception:
        return {}
    return getattr(module, "_PROFILES", {}) or {}


async def local_model_error(model_id: str) -> str | None:
    """Why a local (Ollama) model cannot serve this agent right now, or ``None``.

    Checks reachability, installed-ness, and tool support, in that order; a
    non-local model always passes. A front end with a pull UI (the TUI) handles
    the not-installed case interactively; this is the shared preflight for
    surfaces that can only report the problem.
    """
    from jutul_agent import ollama_client

    if not is_local(model_id):
        return None
    if not await ollama_client.is_reachable():
        return (
            f"Ollama isn't reachable at {ollama_client.host()}. Start it with "
            "`ollama serve` (or install it from https://ollama.com), then try again."
        )
    if is_ollama_cloud(model_id):
        return None
    name = ollama_client.model_name(model_id)
    if not await ollama_client.is_installed(name):
        return f"`{name}` isn't pulled on this machine. Run `ollama pull {name}`, then retry."
    if not await ollama_client.supports_tools(name):
        return (
            f"`{model_id}` doesn't support tool calling, which this agent requires. "
            "Pick a tool-capable model."
        )
    return None
