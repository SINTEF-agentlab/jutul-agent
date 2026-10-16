"""The Azure OpenAI provider: deployments, their profiles, and the built model."""

from __future__ import annotations

import pytest

from jutul_agent import models
from jutul_agent.credentials import missing_credential, provider_by_name
from jutul_agent.models import PROVIDERS, discover_models, key_env_var, model_profile


@pytest.fixture
def azure_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test-resource.openai.azure.com/")
    monkeypatch.setenv("OPENAI_API_VERSION", "2025-04-01-preview")


@pytest.fixture
def fresh_catalog():
    discover_models.cache_clear()
    yield
    discover_models.cache_clear()


def _reasoning_base() -> tuple[str, dict]:
    """A bundled OpenAI reasoning model with a known window, from the installed data."""
    profiles = models._load_profiles(PROVIDERS["openai"].package)
    name = next(
        name
        for name, profile in sorted(profiles.items())
        if profile.get("reasoning_output")
        and profile.get("temperature") is False
        and profile.get("tool_calling")
        and profile.get("max_input_tokens")
    )
    return name, profiles[name]


def test_provider_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    assert key_env_var("azure_openai:any-deployment") == "AZURE_OPENAI_API_KEY"
    assert provider_by_name("azure").name == "azure_openai"
    assert provider_by_name("Azure OpenAI").name == "azure_openai"
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    assert missing_credential("azure_openai:any-deployment") == "AZURE_OPENAI_API_KEY"


def test_deployment_reads_the_profile_of_the_model_behind_it() -> None:
    base, profile = _reasoning_base()
    # The exact name and the usual <model>-<tag> naming both resolve, offline.
    assert models.azure_deployment_profile(base) == profile
    assert models.azure_deployment_profile(f"{base}-prod") == profile
    assert model_profile(f"azure_openai:{base}-prod") == profile
    assert models.context_window(f"azure_openai:{base}-prod-eu") == profile["max_input_tokens"]
    # A name that hides the model reads as unprofiled, never as a wrong profile.
    assert models.azure_deployment_profile("my-deployment") == {}


def test_catalog_offers_the_configured_deployments(
    monkeypatch: pytest.MonkeyPatch, fresh_catalog: None
) -> None:
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "main-deployment")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENTS", " other ,main-deployment,,")
    ids = [model.id for model in discover_models()["azure_openai"]]
    assert ids == ["azure_openai:main-deployment", "azure_openai:other"]


def test_catalog_has_no_azure_section_without_deployments(
    monkeypatch: pytest.MonkeyPatch, fresh_catalog: None
) -> None:
    for var in models.AZURE_DEPLOYMENT_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    # Azure ids name deployments: OpenAI's catalog must not appear under them.
    assert "azure_openai" not in discover_models()


def test_built_deployment_reasons_over_the_responses_api(azure_env: None) -> None:
    from deepagents.profiles.harness.harness_profiles import _harness_profile_for_model
    from langchain_openai import AzureChatOpenAI

    from jutul_agent.agent.builder import (
        _OPENAI_TIMEOUT,
        _resolve_model_for_agent,
        register_provider_profiles,
    )

    register_provider_profiles()
    base, profile = _reasoning_base()
    model = _resolve_model_for_agent(f"azure_openai:{base}-prod")
    assert isinstance(model, AzureChatOpenAI)
    # The deployment is the request's model, which is how Azure routes Responses calls.
    assert model.model_name == f"{base}-prod"
    assert model.use_responses_api is True
    assert model.reasoning == {"effort": "medium", "summary": "auto"}
    assert model.request_timeout == _OPENAI_TIMEOUT
    assert model.profile["max_input_tokens"] == profile["max_input_tokens"]
    # The instance reports provider "azure"; our harness profile still resolves.
    assert _harness_profile_for_model(model, None).general_purpose_subagent.enabled is False


def test_unprofiled_deployment_is_built_plain_with_the_timeout(azure_env: None) -> None:
    from langchain_openai import AzureChatOpenAI

    from jutul_agent.agent.builder import _OPENAI_TIMEOUT, _resolve_model_for_agent

    model = _resolve_model_for_agent("azure_openai:my-deployment")
    assert isinstance(model, AzureChatOpenAI)
    assert model.reasoning is None
    assert not model.use_responses_api
    assert model.request_timeout == _OPENAI_TIMEOUT
