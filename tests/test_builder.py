"""Tests for agent builder helpers."""

from __future__ import annotations

import pytest

from jutul_agent.agent.builder import _set_profile_window, register_provider_profiles
from jutul_agent.models import DEFAULT_MODEL, MODEL_ENV_VAR, resolve_model


def test_set_profile_window_feeds_loaded_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """The loaded window goes into the model profile, where the stock summarizer
    reads its trigger from."""
    from jutul_agent import models

    class _Model:
        profile = None

    monkeypatch.setattr(models, "context_window", lambda model_id: 65_536)
    model = _Model()
    _set_profile_window(model, "ollama:qwen3.6:27b")
    assert model.profile["max_input_tokens"] == 65_536

    # No discoverable window → leave the model's native profile untouched.
    monkeypatch.setattr(models, "context_window", lambda model_id: None)
    untouched = _Model()
    _set_profile_window(untouched, "ollama:mystery")
    assert untouched.profile is None


def test_resolve_model_prefers_explicit() -> None:
    assert resolve_model("anthropic:claude-test") == "anthropic:claude-test"


def test_resolve_model_uses_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(MODEL_ENV_VAR, "openai:gpt-custom")
    assert resolve_model(None) == "openai:gpt-custom"


def test_resolve_model_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(MODEL_ENV_VAR, raising=False)
    assert resolve_model(None) == DEFAULT_MODEL


def test_resolve_model_explicit_beats_config_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(MODEL_ENV_VAR, "openai:env")
    assert (
        resolve_model("anthropic:cli", workspace_model="x:ws", user_model="x:user")
        == "anthropic:cli"
    )


def test_resolve_model_workspace_beats_user_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(MODEL_ENV_VAR, "openai:env")
    resolved = resolve_model(None, workspace_model="anthropic:ws", user_model="x:user")
    assert resolved == "anthropic:ws"


def test_resolve_model_for_agent_handles_ollama_and_cloud(monkeypatch: pytest.MonkeyPatch) -> None:
    from deepagents.profiles.harness.harness_profiles import _harness_profile_for_model
    from langchain_core.language_models import BaseChatModel

    from jutul_agent import ollama_client
    from jutul_agent.agent.builder import _resolve_model_for_agent

    register_provider_profiles()
    monkeypatch.delenv("JUTUL_AGENT_OLLAMA_NUM_CTX", raising=False)
    monkeypatch.setattr(ollama_client, "context_window", lambda name: 262144)
    monkeypatch.setattr(ollama_client, "thinks", lambda name: True)
    # Every model of a provider that accepts a timeout is built here, reasoning
    # or not: an unbounded request hangs the turn just the same.
    for spec in ("google_genai:gemini-2.0-flash", "openai:gpt-4.1-mini"):
        built = _resolve_model_for_agent(spec)
        assert isinstance(built, BaseChatModel), spec
    # A version-tagged Ollama id (>1 colon) becomes a built instance whose context
    # is sized from the model and capped at the budget, and; crucially; our
    # harness profile now resolves for it (it does NOT for such a spec as a string).
    model = _resolve_model_for_agent("ollama:qwen3.6:27b")
    assert isinstance(model, BaseChatModel)
    assert getattr(model, "num_ctx", None) == 65536  # min(262144, budget)
    # Thinking-capable local models get think mode requested explicitly, so
    # the thinking is separated instead of silently swallowing the turn.
    assert getattr(model, "reasoning", None) is True
    profile = _harness_profile_for_model(model, None)
    # Parallel tool calls are not suppressed per provider; no prompt suffix.
    assert not profile.system_prompt_suffix
    assert profile.general_purpose_subagent.enabled is False

    # A local model without the thinking capability keeps think mode unset.
    monkeypatch.setattr(ollama_client, "thinks", lambda name: False)
    model = _resolve_model_for_agent("ollama:plain-model")
    assert getattr(model, "reasoning", None) is None


def test_resolve_model_for_agent_enables_openai_reasoning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from deepagents.profiles.harness.harness_profiles import _harness_profile_for_model
    from langchain_core.language_models import BaseChatModel

    from jutul_agent.agent.builder import _resolve_model_for_agent

    register_provider_profiles()
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    # The bundled model profile says gpt-5.4-mini reasons → effort + summaries.
    model = _resolve_model_for_agent("openai:gpt-5.4-mini")
    assert isinstance(model, BaseChatModel)
    assert getattr(model, "reasoning", None) == {"effort": "medium", "summary": "auto"}
    profile = _harness_profile_for_model(model, None)
    assert profile.general_purpose_subagent.enabled is False
    # Non-reasoning models, and the -chat hybrids whose API rejects
    # reasoning.effort, are still built (for the timeout) but without it.
    for spec in ("openai:gpt-4.1-mini", "openai:gpt-5-chat-latest"):
        built = _resolve_model_for_agent(spec)
        assert isinstance(built, BaseChatModel), spec
        assert getattr(built, "reasoning", None) is None, spec


def test_resolve_model_for_agent_enables_anthropic_thinking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from langchain_core.language_models import BaseChatModel

    from jutul_agent.agent.builder import _resolve_model_for_agent

    register_provider_profiles()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    model = _resolve_model_for_agent("anthropic:claude-sonnet-4-6")
    assert isinstance(model, BaseChatModel)
    assert getattr(model, "thinking", None) == {"type": "enabled", "budget_tokens": 10_000}
    assert getattr(model, "max_tokens", None) == 24_000
    # A model the installed provider package has no profile for is still built
    # (for the timeout), just without thinking (a fictional id so the assertion
    # can't be invalidated by the package later learning a real model).
    unknown = _resolve_model_for_agent("anthropic:claude-imaginary-0-0")
    assert isinstance(unknown, BaseChatModel)
    assert getattr(unknown, "thinking", None) is None


def test_resolve_model_for_agent_degrades_without_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reading the profile builds the model; with no key the spec string
    passes through unchanged instead of failing the agent build here."""
    from jutul_agent.agent.builder import _resolve_model_for_agent

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_ADMIN_KEY", raising=False)
    assert _resolve_model_for_agent("openai:gpt-5.4-mini") == "openai:gpt-5.4-mini"


def test_resolve_model_user_beats_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(MODEL_ENV_VAR, "openai:env")
    assert resolve_model(None, user_model="openai:user") == "openai:user"


def test_resolve_model_for_agent_enables_gemini_thoughts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from langchain_core.language_models import BaseChatModel

    from jutul_agent.agent.builder import _resolve_model_for_agent

    register_provider_profiles()
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    # Bundled profile marks it a thinking model → thoughts made visible.
    model = _resolve_model_for_agent("google_genai:gemini-2.5-flash")
    assert isinstance(model, BaseChatModel)
    assert getattr(model, "include_thoughts", None) is True
    # No profile entry (newer than the package data) → treated as thinking.
    model = _resolve_model_for_agent("google_genai:gemini-3.5-flash")
    assert isinstance(model, BaseChatModel)
    assert getattr(model, "include_thoughts", None) is True
    # Legacy models the data marks non-reasoning are built without thoughts.
    model = _resolve_model_for_agent("google_genai:gemini-2.0-flash")
    assert isinstance(model, BaseChatModel)
    assert getattr(model, "include_thoughts", None) is None


def test_agent_tool_surface_is_pinned(tmp_path) -> None:
    """Pin the tools the model is offered, and require a gate on each mutating one.

    The framework decides which file tools exist, so an upgrade can add one or
    stop installing one without a line of our code changing. Pinning the set
    turns either into a failure here instead of a surprise in a live session.
    """
    from jutul_agent.agent.approval import ApprovalMode, interrupt_on_for_mode
    from jutul_agent.agent.builder import build_agent
    from jutul_agent.lab.fakes import (
        FakeJulia,
        make_fake_adapter,
        make_scripted_model,
        scripted_final,
    )
    from jutul_agent.session import Session

    session = Session.create(
        julia=FakeJulia(), state_root=tmp_path, simulator=make_fake_adapter(tmp_path)
    )
    agent, _ = build_agent(session, model=make_scripted_model([scripted_final("done")]))
    names = set(agent.nodes["tools"].bound.tools_by_name)

    assert names == {
        # Ours.
        "run_julia",
        "reset_julia",
        "plot_julia",
        "recapture_plot",
        "close_plots",
        "record_attempt",
        "write_report",
        "remember",
        # The framework's.
        "ls",
        "read_file",
        "write_file",
        "edit_file",
        "delete",
        "glob",
        "grep",
        "execute",
        "write_todos",
        # Only because the scripted model matches no harness profile, so the
        # stock general-purpose subagent survives; see the test below.
        "task",
    }
    # Everything that mutates the filesystem or spawns a process asks first.
    assert {"write_file", "edit_file", "delete", "execute"} == set(
        interrupt_on_for_mode(ApprovalMode.ASK)
    )
    # No tool reaches a person as its raw agent-facing id.
    from jutul_agent.tool_labels import tool_label

    assert [name for name in sorted(names) if tool_label(name) == name] == []


def test_agent_drops_delegation_without_a_subagent(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model our profile applies to is offered no ``task`` tool.

    The delegation tool exists only while some subagent does, and our profile
    turns the stock general-purpose one off. A model with no registered profile
    keeps it, so the surface a real session sees has to be pinned against a
    profiled model rather than a scripted one.
    """
    from jutul_agent.agent.builder import _resolve_model_for_agent, build_agent
    from jutul_agent.lab.fakes import FakeJulia, make_fake_adapter
    from jutul_agent.session import Session

    register_provider_profiles()
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    session = Session.create(
        julia=FakeJulia(), state_root=tmp_path, simulator=make_fake_adapter(tmp_path)
    )
    agent, _ = build_agent(session, model=_resolve_model_for_agent("openai:gpt-5.4-mini"))
    assert "task" not in set(agent.nodes["tools"].bound.tools_by_name)


def test_every_built_model_gets_a_bounded_request_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The provider packages construct their HTTP clients with no timeout at
    all, so a stalled stream would hang the turn forever, showing nothing. Every
    model we build carries one instead."""
    import httpx

    from jutul_agent.agent.builder import _resolve_model_for_agent

    register_provider_profiles()
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    openai = _resolve_model_for_agent("openai:gpt-5.4-mini")
    effective = openai.root_client._client.timeout
    assert effective.read and effective.read > 0
    assert effective.connect and effective.connect <= 30

    anthropic = _resolve_model_for_agent("anthropic:claude-sonnet-4-6")
    assert isinstance(anthropic._client.timeout, (int, float, httpx.Timeout))
