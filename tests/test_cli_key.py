"""Tests for the ``jutul-agent key`` subcommand."""

from __future__ import annotations

import sys

import pytest

import jutul_agent.interfaces.cli.main  # noqa: F401  (ensure the submodule is imported)
from jutul_agent.interfaces.cli import main

# The package re-exports the ``main`` function, shadowing the submodule attribute,
# so reach the module object through sys.modules to patch its ``load_dotenv``.
cli_main = sys.modules["jutul_agent.interfaces.cli.main"]

# The autouse ``_reset_workspace_overrides`` fixture points state_home at a fresh
# tmp dir, so the global .env these tests read is empty unless a test writes it.


@pytest.fixture(autouse=True)
def _no_provider_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    # main() calls load_dotenv(), which finds a .env relative to the caller's file
    # (so the repo's own .env would seed real keys here). Stub it so the status the
    # test sees comes only from the isolated tmp state-home .env.
    monkeypatch.setattr(cli_main, "load_dotenv", lambda *a, **k: None)


def test_key_show_lists_providers(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["key", "--show"]) == 0
    out = capsys.readouterr().out
    assert "OPENAI_API_KEY" in out
    assert "ANTHROPIC_API_KEY" in out
    assert "OPENROUTER_API_KEY" in out
    assert "not set" in out
    assert ".env" in out  # tells the user where keys live


def test_key_show_reports_a_saved_key(capsys: pytest.CaptureFixture[str]) -> None:
    from jutul_agent.credentials import store_credential_for_provider

    store_credential_for_provider("openai", "sk-saved-key-abcdef123456")
    assert main(["key", "--show"]) == 0
    out = capsys.readouterr().out
    assert "saved" in out
    assert "sk-saved-key-abcdef123456" not in out  # masked, never the raw value


def test_key_set_unknown_provider_errors(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["key", "bogus"]) == 2
    assert "unknown provider" in capsys.readouterr().err


def test_key_set_requires_a_tty(capsys: pytest.CaptureFixture[str]) -> None:
    # Non-interactive (pytest has no TTY): setting a key is refused with guidance.
    assert main(["key", "openai"]) == 2
    assert "interactive terminal" in capsys.readouterr().err
