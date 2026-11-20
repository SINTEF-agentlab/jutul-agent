"""Tests for the Ollama client wrapper (no real server required)."""

from __future__ import annotations

import pytest

from makker import ollama_client
from makker.ollama_client import _matches, host, model_name


class _Model:
    def __init__(self, name: str) -> None:
        self.model = name


class _ListResp:
    def __init__(self, names: list[str]) -> None:
        self.models = [_Model(n) for n in names]


class _ShowResp:
    def __init__(self, capabilities: list[str]) -> None:
        self.capabilities = capabilities


def _fake_client(*, names=None, list_error=None, pull_chunks=None, caps=None, show_error=None):
    class _Chunk:
        def __init__(self, status, total=None, completed=None):
            self.status, self.total, self.completed = status, total, completed

    class _Client:
        async def list(self):
            if list_error is not None:
                raise list_error
            return _ListResp(names or [])

        async def pull(self, name, stream=False):
            async def _gen():
                for status, total, completed in pull_chunks or []:
                    yield _Chunk(status, total, completed)

            return _gen()

        async def show(self, name):
            if show_error is not None:
                raise show_error
            return _ShowResp(caps or [])

    return _Client()


def test_model_name_strips_provider_prefix() -> None:
    assert model_name("ollama:llama3.1") == "llama3.1"
    assert model_name("llama3.1") == "llama3.1"


def test_matches_handles_latest_tag() -> None:
    assert _matches("llama3.1:latest", "llama3.1")
    assert _matches("llama3.1", "llama3.1")
    assert not _matches("qwen2.5:latest", "llama3.1")


def test_host_defaults_and_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    assert host() == "http://localhost:11434"
    monkeypatch.setenv("OLLAMA_HOST", "http://box:1234")
    assert host() == "http://box:1234"


async def test_is_reachable_true_and_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ollama_client, "_client", lambda: _fake_client(names=[]))
    assert await ollama_client.is_reachable() is True

    monkeypatch.setattr(
        ollama_client, "_client", lambda: _fake_client(list_error=ConnectionError("down"))
    )
    assert await ollama_client.is_reachable() is False


async def test_installed_models_and_is_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ollama_client,
        "_client",
        lambda: _fake_client(names=["llama3.1:latest", "qwen2.5:latest"]),
    )
    assert set(await ollama_client.installed_models()) == {"llama3.1:latest", "qwen2.5:latest"}
    assert await ollama_client.is_installed("llama3.1") is True
    assert await ollama_client.is_installed("mistral") is False


async def test_supports_tools_from_capabilities(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ollama_client, "_client", lambda: _fake_client(caps=["completion", "tools", "thinking"])
    )
    assert await ollama_client.capabilities("qwen3:14b") == ["completion", "tools", "thinking"]
    assert await ollama_client.supports_tools("qwen3:14b") is True


async def test_supports_tools_false_when_capability_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    # A daemon too old for the model's template reports a reduced set (no tools).
    monkeypatch.setattr(
        ollama_client, "_client", lambda: _fake_client(caps=["completion", "vision"])
    )
    assert await ollama_client.supports_tools("qwen3.6:27b") is False


async def test_capabilities_empty_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ollama_client, "_client", lambda: _fake_client(show_error=ConnectionError("down"))
    )
    assert await ollama_client.capabilities("x") == []
    assert await ollama_client.supports_tools("x") is False


def test_context_window_reads_max_context_length(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Info:
        def __init__(self) -> None:
            self.modelinfo = {"qwen35.context_length": 262144, "general.parameter_count": 27}

    class _SyncClient:
        def __init__(self, *a, **k) -> None:
            pass

        def show(self, name):
            return _Info()

    monkeypatch.setattr("ollama.Client", _SyncClient)
    assert ollama_client.context_window("qwen3.6:27b") == 262144


def test_context_window_none_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class _SyncClient:
        def __init__(self, *a, **k) -> None:
            pass

        def show(self, name):
            raise ConnectionError("down")

    monkeypatch.setattr("ollama.Client", _SyncClient)
    assert ollama_client.context_window("x") is None


def test_ctx_budget_default_and_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MAKKER_OLLAMA_NUM_CTX", raising=False)
    assert ollama_client.ctx_budget() == 65536
    monkeypatch.setenv("MAKKER_OLLAMA_NUM_CTX", "16384")
    assert ollama_client.ctx_budget() == 16384
    monkeypatch.setenv("MAKKER_OLLAMA_NUM_CTX", "not-an-int")
    assert ollama_client.ctx_budget() == 65536  # bad value falls back to the default


def test_num_ctx_clamps_model_context_to_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MAKKER_OLLAMA_NUM_CTX", raising=False)  # budget = 65536
    # A big-context model is capped at the budget...
    monkeypatch.setattr(ollama_client, "context_window", lambda name: 262144)
    assert ollama_client.num_ctx("qwen3.6:27b") == 65536
    # ...a small model keeps its own (smaller) max, not an inflated value...
    monkeypatch.setattr(ollama_client, "context_window", lambda name: 8192)
    assert ollama_client.num_ctx("tiny") == 8192
    # ...and when the daemon can't report, fall back to the budget.
    monkeypatch.setattr(ollama_client, "context_window", lambda name: None)
    assert ollama_client.num_ctx("mystery") == 65536


async def test_installed_models_empty_when_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ollama_client, "_client", lambda: _fake_client(list_error=ConnectionError("down"))
    )
    assert await ollama_client.installed_models() == []
    assert await ollama_client.is_installed("llama3.1") is False


async def test_pull_yields_progress_with_fraction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ollama_client,
        "_client",
        lambda: _fake_client(pull_chunks=[("pulling", 100, 50), ("success", None, None)]),
    )
    progress = [p async for p in ollama_client.pull("llama3.1")]
    assert progress[0].status == "pulling"
    assert progress[0].fraction == 0.5
    assert progress[1].status == "success"
    assert progress[1].fraction is None
