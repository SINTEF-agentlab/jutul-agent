"""Tests for user-global credential storage and provider checks."""

from __future__ import annotations

import os
import sys

import pytest

from jutul_agent.credentials import (
    key_providers,
    key_status,
    load_user_credentials,
    mask_secret,
    missing_credential,
    provider_by_name,
    store_credential,
    store_credential_for_provider,
    user_env_path,
)
from jutul_agent.paths import state_home

# The autouse ``_reset_workspace_overrides`` fixture isolates state_home per
# test, so ``user_env_path()`` points at a fresh tmp dir.

_VAR = "JUTUL_AGENT_TEST_KEY"


@pytest.fixture(autouse=True)
def _clean_test_var() -> None:
    # store_credential mutates os.environ directly, so clean up around each test.
    os.environ.pop(_VAR, None)
    yield
    os.environ.pop(_VAR, None)


def test_user_env_path_under_state_home() -> None:
    assert user_env_path() == state_home() / ".env"


def test_store_credential_writes_file_and_sets_env() -> None:
    path = store_credential(_VAR, "secret-value")
    assert path == user_env_path()
    assert os.environ[_VAR] == "secret-value"
    assert _VAR in path.read_text(encoding="utf-8")
    assert "secret-value" in path.read_text(encoding="utf-8")


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file mode")
def test_store_credential_locks_file_mode() -> None:
    path = store_credential(_VAR, "secret-value")
    assert (path.stat().st_mode & 0o777) == 0o600


@pytest.mark.parametrize(
    ("spec", "env_var"),
    [
        ("openai:gpt-5.4", "OPENAI_API_KEY"),
        ("openrouter:vendor/test-model", "OPENROUTER_API_KEY"),
    ],
)
def test_missing_credential_present_absent_and_local(
    monkeypatch: pytest.MonkeyPatch, spec: str, env_var: str
) -> None:
    monkeypatch.delenv(env_var, raising=False)
    assert missing_credential(spec) == env_var
    monkeypatch.setenv(env_var, "x")
    assert missing_credential(spec) is None
    # Local providers need no key; unknown providers have none either.
    assert missing_credential("ollama:llama3.1") is None
    assert missing_credential("madeup:model") is None


def test_load_user_credentials_does_not_override_existing(monkeypatch: pytest.MonkeyPatch) -> None:
    path = user_env_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'{_VAR}="from-file"\nJUTUL_AGENT_TEST_OTHER="loaded"\n', encoding="utf-8")

    monkeypatch.setenv(_VAR, "from-shell")
    load_user_credentials()
    # Shell value wins; a var only in the file is loaded.
    assert os.environ[_VAR] == "from-shell"
    assert os.environ.get("JUTUL_AGENT_TEST_OTHER") == "loaded"
    os.environ.pop("JUTUL_AGENT_TEST_OTHER", None)


@pytest.fixture
def _clear_provider_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(var, raising=False)


def test_mask_secret_hides_the_middle() -> None:
    assert mask_secret("sk-abcdefghijklmnop") == "sk-******mnop"
    # Short secrets reveal nothing.
    assert mask_secret("short") == "*****"


def test_key_providers_excludes_local() -> None:
    names = {p.name for p in key_providers()}
    assert {"openai", "anthropic", "google_genai", "openrouter"} <= names
    assert "ollama" not in names  # local, no key


def test_provider_by_name_accepts_name_label_and_prefix() -> None:
    assert provider_by_name("openai").name == "openai"
    assert provider_by_name("OpenAI").name == "openai"
    assert provider_by_name("google").name == "google_genai"  # prefix match
    assert provider_by_name("Google").name == "google_genai"  # label
    assert provider_by_name("OpenRouter").name == "openrouter"
    assert provider_by_name("nope") is None


def test_key_status_reports_none_file_and_environment(
    _clear_provider_keys: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    by_provider = {s.provider: s for s in key_status()}
    assert by_provider["openai"].source == "none" and not by_provider["openai"].is_set

    # A key saved to the global .env reads as "file".
    store_credential_for_provider("openai", "sk-savedsavedsaved")
    saved = {s.provider: s for s in key_status()}["openai"]
    assert saved.is_set and saved.source == "file" and not saved.shadowed
    assert saved.masked and "saved" not in (saved.masked or "")  # masked, not the raw value

    # A shell value that differs from the saved file value is flagged as shadowing.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-shell-only")
    anth = {s.provider: s for s in key_status()}["anthropic"]
    assert anth.is_set and anth.source == "environment"


def test_key_status_flags_shadowed_when_env_overrides_file(
    _clear_provider_keys: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # File holds one value; the environment holds another that wins on load.
    path = user_env_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('GOOGLE_API_KEY="file-value-123456"\n', encoding="utf-8")
    monkeypatch.setenv("GOOGLE_API_KEY", "env-value-987654")
    google = {s.provider: s for s in key_status()}["google_genai"]
    assert google.is_set and google.shadowed


def test_store_credential_for_provider_rejects_unknown_and_empty() -> None:
    with pytest.raises(KeyError):
        store_credential_for_provider("nope", "x")
    with pytest.raises(ValueError):
        store_credential_for_provider("openai", "   ")
