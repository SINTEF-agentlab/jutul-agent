"""Tests for the model catalog (discovery) and provider metadata."""

from __future__ import annotations

import httpx
import pytest

from jutul_agent.models import (
    DEFAULT_MODEL,
    OLLAMA_CLOUD,
    PROVIDERS,
    RECOMMENDED_OLLAMA_LOCAL,
    discover_available_models,
    discover_models,
    is_known_model,
    is_local,
    is_ollama_cloud,
    key_env_var,
    missing_provider_error,
    provider_info,
    provider_of,
)


def test_provider_of_handles_missing_prefix() -> None:
    assert provider_of("anthropic:test-model") == "anthropic"
    assert provider_of("bare-model-name") == ""


def test_key_env_var_and_local_flag() -> None:
    assert key_env_var("openai:test-model") == "OPENAI_API_KEY"
    assert key_env_var("anthropic:test-model") == "ANTHROPIC_API_KEY"
    assert key_env_var("google_genai:test-model") == "GOOGLE_API_KEY"
    assert key_env_var("openrouter:vendor/test-model") == "OPENROUTER_API_KEY"
    assert key_env_var("ollama:test-model") is None
    assert key_env_var("madeup:model") is None
    assert is_local("ollama:test-model") is True
    assert is_local("openai:test-model") is False
    assert is_local("openrouter:vendor/test-model") is False


def test_missing_provider_error_names_the_fix() -> None:
    problem = missing_provider_error("test-model")
    assert problem is not None
    # The message has to point at the spec itself: every downstream preflight is
    # keyed on the prefix, so without it the failure surfaces much later.
    assert "provider prefix" in problem
    assert "openai:test-model" in problem


def test_missing_provider_error_accepts_any_prefixed_spec() -> None:
    assert missing_provider_error("openai:test-model") is None
    assert missing_provider_error("ollama:test-model:tag") is None
    # A provider this module has no entry for still reaches init_chat_model.
    assert missing_provider_error("madeup:model") is None


def test_provider_info_unknown_returns_none() -> None:
    assert provider_info("madeup:model") is None
    assert provider_info("openai:test-model") is PROVIDERS["openai"]


def test_discovery_groups_real_models_by_provider() -> None:
    catalog = discover_models()
    # Bundled providers ship profiles, so each contributes models; Ollama has no
    # static profiles and is discovered from the daemon by the selector, not here.
    assert "openai" in catalog
    assert "anthropic" in catalog
    assert "openrouter" in catalog
    assert "ollama" not in catalog
    for provider, models in catalog.items():
        assert models, provider
        for model in models:
            assert model.provider == provider
            assert model.id.startswith(f"{provider}:")
            # The label is the bare model name (no provider prefix).
            assert model.label == model.id.partition(":")[2]


def test_discovery_includes_the_default_model() -> None:
    assert is_known_model(DEFAULT_MODEL)


def test_openrouter_profile_uses_installed_provider_data(monkeypatch) -> None:
    """Keep a real profile lookup alongside synthetic builder capability tests."""
    from jutul_agent import models

    monkeypatch.setenv("OPENROUTER_API_KEY", "router-key")
    profiles = models._load_profiles(PROVIDERS["openrouter"].package)
    info = next(
        info
        for info in discover_models()["openrouter"]
        if profiles[info.label].get("max_input_tokens")
    )
    assert models.model_profile(info.id)["tool_calling"] is True
    assert models.context_window(info.id) == profiles[info.label]["max_input_tokens"]


def test_live_discovery_adds_new_ids_without_replacing_profiles(monkeypatch) -> None:
    from jutul_agent import models

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    known = discover_models()["openai"][0].id
    monkeypatch.setattr(
        models, "_openai_available", lambda: ["new-test-model", known.partition(":")[2]]
    )
    monkeypatch.setattr(models, "_anthropic_available", lambda: ["new-test-model"])
    monkeypatch.setattr(models, "_google_available", lambda: ["new-test-model"])
    monkeypatch.setattr(models, "_openrouter_available", lambda: ["vendor/new-tool-model"])

    catalog = discover_available_models()
    assert {"openai:new-test-model", "anthropic:new-test-model", "google_genai:new-test-model"} <= {
        model.id for group in catalog.values() for model in group
    }
    assert sum(model.id == known for model in catalog["openai"]) == 1
    assert next(model for model in catalog["openai"] if model.id == "openai:new-test-model").note
    assert any(model.id == "openrouter:vendor/new-tool-model" for model in catalog["openrouter"])


def test_live_discovery_falls_back_to_profiles_when_provider_fails(monkeypatch) -> None:
    from jutul_agent import models

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    def fail() -> list[str]:
        raise RuntimeError("offline")

    monkeypatch.setattr(models, "_openai_available", fail)
    assert discover_available_models() == discover_models()


@pytest.fixture
def openrouter_http(monkeypatch):
    """Keep real HTTP handling; replace only the network transport."""
    for info in PROVIDERS.values():
        if info.key_env_var:
            monkeypatch.delenv(info.key_env_var, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-key")
    monkeypatch.delenv("OPENROUTER_API_BASE", raising=False)
    responses: list[httpx.Response | Exception] = []
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    client_class = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: client_class(transport=httpx.MockTransport(respond), **kwargs),
    )
    return responses, requests


@pytest.mark.parametrize(
    "base_url", ["https://openrouter.ai/api/v1", "https://router.example/api/v1/"]
)
def test_openrouter_live_catalog_filters_for_text_and_tools(
    openrouter_http, monkeypatch, base_url
) -> None:
    from jutul_agent import models

    if base_url.endswith("/"):
        monkeypatch.setenv("OPENROUTER_API_BASE", base_url)
    responses, requests = openrouter_http
    text = {"input_modalities": ["text", "image"], "output_modalities": ["text"]}
    valid = {"id": "vendor/chat:free", "supported_parameters": ["tools"], "architecture": text}
    responses.append(
        httpx.Response(
            200,
            json={
                "data": [
                    valid,
                    valid,
                    {**valid, "supported_parameters": []},
                    {"id": "vendor/no-metadata"},
                    {**valid, "architecture": {**text, "output_modalities": ["image"]}},
                    {**valid, "architecture": {**text, "input_modalities": ["audio"]}},
                    {**valid, "architecture": None},
                    {**valid, "supported_parameters": "tools"},
                    {**valid, "id": None},
                    None,
                    {**valid, "id": "vendor/another-chat"},
                ]
            },
        )
    )
    assert models._openrouter_available() == ["vendor/another-chat", "vendor/chat:free"]
    assert len(requests) == 1
    assert str(requests[0].url) == f"{base_url.rstrip('/')}/models"
    assert requests[0].headers["Authorization"] == "Bearer router-key"
    assert 0 < requests[0].extensions["timeout"]["read"] <= 10


@pytest.mark.parametrize(
    "failure",
    [
        # A valid body ensures this case depends on HTTP status checking.
        httpx.Response(
            401,
            json={
                "data": [
                    {
                        "id": "vendor/test-model",
                        "supported_parameters": ["tools"],
                        "architecture": {
                            "input_modalities": ["text"],
                            "output_modalities": ["text"],
                        },
                    }
                ]
            },
        ),
        httpx.Response(200, content="invalid JSON"),
        httpx.Response(200, json={"data": None}),
        httpx.Response(200, json=[]),
        httpx.ReadTimeout("test timeout"),
    ],
    ids=["http", "json", "data-schema", "root-schema", "timeout"],
)
def test_openrouter_catalog_failures_preserve_bundled_models(openrouter_http, failure) -> None:
    responses, requests = openrouter_http
    responses.append(failure)
    assert discover_available_models() == discover_models()
    assert len(requests) == 1


def test_openrouter_discovery_without_key_makes_no_request(openrouter_http, monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY")
    _, requests = openrouter_http
    assert discover_available_models() == discover_models()
    assert not requests


def test_is_known_model_rejects_free_text() -> None:
    assert is_known_model(discover_models()["openai"][0].id)
    assert not is_known_model("openrouter:some/model")
    assert not is_known_model("anthropic:haiku")  # not a real id


def test_ollama_cloud_detection() -> None:
    assert is_ollama_cloud("ollama:test-model:cloud")
    assert not is_ollama_cloud("ollama:test-model")
    assert not is_ollama_cloud("openai:test-model")


def test_curated_ollama_lists_are_well_formed() -> None:
    # Local tags are pulled (no :cloud); cloud tags are hosted (always :cloud).
    assert RECOMMENDED_OLLAMA_LOCAL
    assert all(not t.endswith(":cloud") for t in RECOMMENDED_OLLAMA_LOCAL)
    assert OLLAMA_CLOUD
    assert all(t.endswith(":cloud") for t in OLLAMA_CLOUD)


def test_context_window_google_sdk_fallback(monkeypatch) -> None:
    """A Gemini model with no bundled profile falls back to the API lookup."""
    from jutul_agent import models

    # Without a key the lookup declines instead of constructing a client.
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    assert models._google_context_window("unknown-test-model") is None

    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setattr(models, "_google_context_window", lambda name: 1_048_576)
    assert models.context_window("google_genai:unknown-test-model") == 1_048_576
