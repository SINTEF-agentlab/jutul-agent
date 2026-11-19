"""Tests for the user-global config (state-home root config.toml)."""

from __future__ import annotations

from pathlib import Path

from makker.paths import user_config_path
from makker.user_config import UserConfig, load_user_config, write_user_config

# The autouse ``_reset_workspace_overrides`` fixture points state_home at a
# fresh tmp dir per test, so ``user_config_path()`` is isolated here.


def test_load_user_config_missing_returns_empty() -> None:
    assert load_user_config() == UserConfig()


def test_user_config_round_trip() -> None:
    path = write_user_config(UserConfig(model="anthropic:claude-sonnet-4-6"))
    assert path == user_config_path()
    assert path.read_text(encoding="utf-8") == 'model = "anthropic:claude-sonnet-4-6"\n'
    assert load_user_config().model == "anthropic:claude-sonnet-4-6"


def test_write_empty_user_config_writes_empty_file() -> None:
    path = write_user_config(UserConfig())
    assert path.read_text(encoding="utf-8") == ""
    assert load_user_config() == UserConfig()


def test_user_config_path_under_state_home() -> None:
    # config.toml sits at the root of the makker home, beside workspaces/.
    from makker.paths import state_home

    assert user_config_path() == state_home() / "config.toml"
    assert isinstance(user_config_path(), Path)
