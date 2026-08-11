"""Tests for the model catalog (discovery) and provider metadata."""

from __future__ import annotations

from jutul_agent.models import (
    DEFAULT_MODEL,
    OLLAMA_CLOUD,
    PROVIDERS,
    RECOMMENDED_OLLAMA_LOCAL,
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
    assert provider_of("anthropic:claude-sonnet-4-6") == "anthropic"
    assert provider_of("bare-model-name") == ""


def test_key_env_var_and_local_flag() -> None:
    assert key_env_var("openai:gpt-5.5") == "OPENAI_API_KEY"
    assert key_env_var("anthropic:claude-opus-4-8") == "ANTHROPIC_API_KEY"
    assert key_env_var("google_genai:gemini-3-flash-preview") == "GOOGLE_API_KEY"
    assert key_env_var("ollama:llama4") is None
    assert key_env_var("madeup:model") is None
    assert is_local("ollama:llama4") is True
    assert is_local("openai:gpt-5.5") is False


def test_missing_provider_error_names_the_fix() -> None:
    problem = missing_provider_error("gpt-5.6-terra")
    assert problem is not None
    # The message has to point at the spec itself: every downstream preflight is
    # keyed on the prefix, so without it the failure surfaces much later.
    assert "provider prefix" in problem
    assert "openai:gpt-5.6-terra" in problem


def test_missing_provider_error_accepts_any_prefixed_spec() -> None:
    assert missing_provider_error("openai:gpt-5.4-mini") is None
    assert missing_provider_error("ollama:qwen3:8b") is None
    # A provider this module has no entry for still reaches init_chat_model.
    assert missing_provider_error("madeup:model") is None


def test_provider_info_unknown_returns_none() -> None:
    assert provider_info("madeup:model") is None
    assert provider_info("openai:gpt-5.5") is PROVIDERS["openai"]


def test_discovery_groups_real_models_by_provider() -> None:
    catalog = discover_models()
    # Bundled providers ship profiles, so each contributes models; Ollama has no
    # static profiles and is discovered from the daemon by the selector, not here.
    assert "openai" in catalog
    assert "anthropic" in catalog
    assert "ollama" not in catalog
    for provider, models in catalog.items():
        assert models, provider
        for model in models:
            assert model.provider == provider
            assert model.id.startswith(f"{provider}:")
            # The label is the bare model name (no provider prefix).
            assert ":" not in model.label


def test_discovery_includes_the_default_model() -> None:
    assert is_known_model(DEFAULT_MODEL)


def test_is_known_model_rejects_free_text() -> None:
    assert is_known_model("openai:gpt-5.4-mini")
    assert not is_known_model("openrouter:some/model")
    assert not is_known_model("anthropic:haiku")  # not a real id


def test_ollama_cloud_detection() -> None:
    assert is_ollama_cloud("ollama:glm-5.1:cloud")
    assert not is_ollama_cloud("ollama:llama4")
    assert not is_ollama_cloud("openai:gpt-5.5")


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
    assert models._google_context_window("gemini-3.5-flash") is None

    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setattr(models, "_google_context_window", lambda name: 1_048_576)
    assert models.context_window("google_genai:gemini-3.5-flash") == 1_048_576
