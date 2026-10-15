"""User-global API-key storage and provider-credential checks.

Keys live in a global ``.env`` file at ``state_home()/.env``, loaded on startup
alongside any repo or workspace ``.env``. Shell environment and a project
``.env`` take precedence; this file is the fallback filled in when the user
enters a key interactively (at ``init``, in the model selector, via the
``jutul-agent key`` command, or in the web UI). Keys are kept out of
``config.toml`` so that file stays safe to share.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values, load_dotenv, set_key

from jutul_agent.models import PROVIDERS, ProviderInfo, key_env_var, provider_info
from jutul_agent.paths import state_home


def user_env_path() -> Path:
    """Global secrets file at the root of the jutul-agent home."""
    return state_home() / ".env"


def load_user_credentials() -> None:
    """Load the global ``.env`` without overriding existing variables."""
    path = user_env_path()
    if path.exists():
        load_dotenv(path, override=False)


def missing_credential(model_id: str) -> str | None:
    """Env var the model's provider needs but that isn't set, else ``None``.

    ``None`` when the provider needs no key (local models like Ollama) or the
    key is already present in the environment.
    """
    env_var = key_env_var(model_id)
    if env_var is None or os.environ.get(env_var):
        return None
    return env_var


def store_credential(env_var: str, value: str) -> Path:
    """Persist ``env_var`` to the global ``.env`` and set it in the environment.

    Returns the file path. ``os.environ`` is updated so the key works without a
    restart.
    """
    value = value.strip()
    path = user_env_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)
    # Restrict to the owner on POSIX; a no-op on Windows.
    with contextlib.suppress(OSError):
        os.chmod(path, 0o600)
    set_key(str(path), env_var, value)
    os.environ[env_var] = value
    return path


# --- Provider-key management (the `key` command, the web UI, doctor) ---------


def key_providers() -> list[ProviderInfo]:
    """Providers that authenticate with an API key (local ones are excluded)."""
    return [info for info in PROVIDERS.values() if info.key_env_var is not None]


def provider_by_name(name: str) -> ProviderInfo | None:
    """The provider whose ``name`` or ``label`` matches ``name`` (case-insensitive).

    Accepts the catalog name (``openai``, ``google_genai``), the human label
    (``OpenAI``, ``Google``), or the bare env-var prefix (``google``), so a user
    can name a provider the obvious way.
    """
    needle = name.strip().lower()
    for info in PROVIDERS.values():
        if needle in {info.name.lower(), info.label.lower()}:
            return info
    # Fall back to a prefix match on the name, so "google" finds "google_genai".
    for info in PROVIDERS.values():
        if info.key_env_var is not None and info.name.lower().startswith(needle):
            return info
    return None


def mask_secret(value: str) -> str:
    """A safe-to-display form of a secret: first/last few chars, middle hidden."""
    value = value.strip()
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:3]}{'*' * 6}{value[-4:]}"


# Name endings that mark a credential (``*_API_KEY``, ``AZURE_OPENAI_AD_TOKEN``,
# ``OPENAI_ADMIN_KEY``, ``CLIENT_SECRET``, ...).
_SECRET_ENV_SUFFIXES = ("_API_KEY", "_ADMIN_KEY", "_TOKEN", "_SECRET")


def is_secret_env_var(name: str) -> bool:
    """Whether ``name`` holds a credential that processes running agent-written
    code (the Julia kernel, shell commands) must not inherit."""
    return name.upper().endswith(_SECRET_ENV_SUFFIXES) or any(
        info.key_env_var == name for info in PROVIDERS.values()
    )


def scrubbed_environ() -> dict[str, str]:
    """``os.environ`` without credentials, for processes that run agent-written code.

    Neither Julia nor the shell needs a provider key, and a model that prints its
    environment while debugging would otherwise copy one into the trace and into
    every transcript.
    """
    return {name: value for name, value in os.environ.items() if not is_secret_env_var(name)}


@dataclass(frozen=True)
class KeyStatus:
    """How one provider's API key is configured, for display and the web UI."""

    provider: str  # catalog name, e.g. "openai"
    label: str  # human label, e.g. "OpenAI"
    env_var: str  # the variable the provider reads, e.g. "OPENAI_API_KEY"
    is_set: bool  # whether the key is present in the environment
    masked: str | None  # a masked preview of the active value, when set
    source: str  # "file" (the saved global .env), "environment", or "none"
    shadowed: bool  # a higher-precedence source overrides the saved file value


def key_status() -> list[KeyStatus]:
    """The configuration of every key-based provider's credential.

    ``source`` distinguishes a key saved in the global ``.env`` (which the app
    manages) from one set in the shell or a project ``.env`` (which the app
    cannot change and which takes precedence). ``shadowed`` flags the footgun
    where a saved key is present but overridden by the environment, so editing
    the saved key would have no effect until the shell value is cleared.
    """
    file_values = dotenv_values(user_env_path())
    statuses: list[KeyStatus] = []
    for info in key_providers():
        env_var = info.key_env_var
        assert env_var is not None  # key_providers() filters these in
        active = os.environ.get(env_var) or ""
        in_file = bool((file_values.get(env_var) or "").strip())
        file_value = (file_values.get(env_var) or "").strip()
        if active:
            source = "file" if in_file else "environment"
            shadowed = in_file and active != file_value
        else:
            source = "none"
            shadowed = False
        statuses.append(
            KeyStatus(
                provider=info.name,
                label=info.label,
                env_var=env_var,
                is_set=bool(active),
                masked=mask_secret(active) if active else None,
                source=source,
                shadowed=shadowed,
            )
        )
    return statuses


def store_credential_for_provider(provider: str, value: str) -> tuple[ProviderInfo, Path]:
    """Save ``value`` as the key for ``provider``. Raises ``KeyError`` if unknown.

    ``provider`` may be a model id (``openai:gpt-...``), a provider name, a
    label, or the bare provider prefix; the env var is resolved from it.
    """
    info = provider_info(provider) or provider_by_name(provider)
    if info is None or info.key_env_var is None:
        raise KeyError(provider)
    value = value.strip()
    if not value:
        raise ValueError("empty API key")
    path = store_credential(info.key_env_var, value)
    return info, path
