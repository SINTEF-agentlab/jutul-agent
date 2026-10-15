"""Agent-run code (the Julia kernel, shell commands) does not inherit credentials."""

from __future__ import annotations

import shlex
import sys
from pathlib import Path

import pytest

from jutul_agent.credentials import is_secret_env_var, scrubbed_environ

_SECRETS = {
    "AZURE_OPENAI_API_KEY": "azure-secret",
    "OPENAI_API_KEY": "openai-secret",
    "AZURE_OPENAI_AD_TOKEN": "ad-secret",
    "SOME_TOOL_API_KEY": "tool-secret",
    "GITHUB_TOKEN": "gh-secret",
    "OPENAI_ADMIN_KEY": "admin-secret",
    "AZURE_CLIENT_SECRET": "client-secret",
}


@pytest.fixture
def secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in _SECRETS.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test-resource.openai.azure.com/")


def test_secret_names() -> None:
    for name in (*_SECRETS, "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "OPENROUTER_API_KEY"):
        assert is_secret_env_var(name), name
    for name in ("PATH", "HOME", "DISPLAY", "JULIA_DEPOT_PATH", "AZURE_OPENAI_ENDPOINT"):
        assert not is_secret_env_var(name), name


def test_scrubbed_environ_drops_only_credentials(secrets: None) -> None:
    env = scrubbed_environ()
    assert not set(_SECRETS) & set(env)
    assert env["AZURE_OPENAI_ENDPOINT"] == "https://test-resource.openai.azure.com/"
    assert "PATH" in env


def test_julia_launch_env_withholds_filtered_names(secrets: None) -> None:
    from jutul_agent.juliakernel import JuliaKernel, KernelConfig

    config = KernelConfig(env={"DISPLAY": ":99"}, env_filter=is_secret_env_var)
    _, _, env = JuliaKernel(config)._build_launch(12345)
    assert not set(_SECRETS) & set(env)
    # Explicit extras still apply, and the control token is set after filtering.
    assert env["DISPLAY"] == ":99"
    assert "JK_TOKEN" in env
    # Without a filter the kernel inherits everything, as before.
    _, _, plain = JuliaKernel(KernelConfig())._build_launch(12345)
    assert plain["AZURE_OPENAI_API_KEY"] == "azure-secret"


def test_shell_commands_run_without_credentials(secrets: None, tmp_path: Path) -> None:
    from jutul_agent.agent.builder import build_backend

    names = ", ".join(repr(name) for name in _SECRETS)
    probe = f"import os; print(sorted(set(os.environ) & {{{names}}}))"
    result = build_backend(workspace=tmp_path).execute(
        f"{shlex.quote(sys.executable)} -c {shlex.quote(probe)}"
    )
    assert result.exit_code == 0, result.output
    assert result.output.strip() == "[]"
