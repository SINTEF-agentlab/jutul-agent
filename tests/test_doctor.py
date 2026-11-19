"""Tests for the model-aware checks in `makker doctor`."""

from __future__ import annotations

import pytest

from makker import ollama_client
from makker.interfaces.cli import doctor


class _RecordingReport:
    def __init__(self) -> None:
        self.lines: list[tuple[str, str, str]] = []

    def line(self, status: str, label: str, detail: str = "", fix: str = "") -> None:
        self.lines.append((status, label, detail))

    def status_for(self, label: str) -> str | None:
        return next((s for s, lab, _ in self.lines if lab == label), None)


def _async_return(value):
    async def _fn(*args, **kwargs):
        return value

    return _fn


def test_check_reports_model_and_passes_when_key_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    report = _RecordingReport()
    doctor._check_model_and_key(report, "openai:gpt-5.4")
    assert report.status_for("Model") == doctor.PASS
    assert report.status_for("Provider API key") == doctor.PASS


def test_check_fails_when_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    report = _RecordingReport()
    doctor._check_model_and_key(report, "anthropic:claude-sonnet-4-6")
    assert report.status_for("Provider API key") == doctor.FAIL


def test_check_unknown_provider_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    report = _RecordingReport()
    doctor._check_model_and_key(report, "madeup:model")
    assert report.status_for("Provider API key") == doctor.WARN


def test_check_ollama_pulled_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ollama_client, "is_reachable", _async_return(True))
    monkeypatch.setattr(ollama_client, "is_installed", _async_return(True))
    monkeypatch.setattr(ollama_client, "supports_tools", _async_return(True))
    report = _RecordingReport()
    doctor._check_model_and_key(report, "ollama:llama3.1")
    assert report.status_for("Ollama model") == doctor.PASS


def test_check_ollama_without_tools_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ollama_client, "is_reachable", _async_return(True))
    monkeypatch.setattr(ollama_client, "is_installed", _async_return(True))
    monkeypatch.setattr(ollama_client, "supports_tools", _async_return(False))
    report = _RecordingReport()
    doctor._check_model_and_key(report, "ollama:qwen3.6:27b")
    assert report.status_for("Ollama model") == doctor.FAIL


def test_check_ollama_cloud_passes_without_pull(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ollama_client, "is_reachable", _async_return(True))

    async def _must_not_check(name):
        raise AssertionError("cloud models are not pulled/capability-checked")

    monkeypatch.setattr(ollama_client, "is_installed", _must_not_check)
    monkeypatch.setattr(ollama_client, "supports_tools", _must_not_check)
    report = _RecordingReport()
    doctor._check_model_and_key(report, "ollama:glm-5.1:cloud")
    assert report.status_for("Ollama model") == doctor.PASS


def test_check_ollama_unreachable_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ollama_client, "is_reachable", _async_return(False))
    report = _RecordingReport()
    doctor._check_model_and_key(report, "ollama:llama3.1")
    assert report.status_for("Ollama server") == doctor.WARN
