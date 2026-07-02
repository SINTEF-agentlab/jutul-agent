"""User-global configuration shared across every workspace.

Stored at ``state_home()/config.toml`` (see ``paths.user_config_path``). It
shares the top-level ``model`` key with the per-workspace config, and the
workspace value wins (see ``resolve_model`` in ``agent.builder``). Secrets are
not stored here — API keys go in the global ``.env`` instead.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from jutul_agent.paths import user_config_path


@dataclass(frozen=True)
class UserConfig:
    """Contents of the user-global ``config.toml``. Empty is fine."""

    model: str | None = None


def load_user_config() -> UserConfig:
    """Read the user-global ``config.toml`` if present; else an empty config."""

    path = user_config_path()
    if not path.exists():
        return UserConfig()

    data = tomllib.loads(path.read_text(encoding="utf-8"))
    model = data.get("model")
    return UserConfig(model=model if isinstance(model, str) else None)


def write_user_config(config: UserConfig) -> Path:
    """Persist ``config`` to the user-global config file. Returns the path."""

    path = user_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    if config.model:
        from jutul_agent.workspace import toml_basic_string

        lines.append(f"model = {toml_basic_string(config.model)}")

    body = "\n".join(lines)
    path.write_text(body + "\n" if body else "", encoding="utf-8")
    return path
