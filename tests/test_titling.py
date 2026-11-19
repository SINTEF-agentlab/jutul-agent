"""Best-effort LLM session titling: reply cleaning and the guarded happy path."""

from __future__ import annotations

import pytest

from makker.agent.titling import _clean, generate_session_title


def test_clean_trims_quotes_punctuation_and_extra_lines() -> None:
    assert _clean('"Reservoir Sweep Study."') == "Reservoir Sweep Study"
    assert _clean("Chen2020 Discharge\n\n(some chatter)") == "Chen2020 Discharge"
    assert _clean("   ") is None
    assert _clean(["not", "a", "string"]) is not None  # falls back to content_to_str


def test_clean_truncates_overlong_titles() -> None:
    out = _clean("word " * 40)
    assert out is not None and len(out) <= 61 and out.endswith("…")


@pytest.mark.asyncio
async def test_generate_title_returns_none_without_model_or_text() -> None:
    assert await generate_session_title(None, "hello") is None
    assert await generate_session_title("anthropic:x", "   ") is None


@pytest.mark.asyncio
async def test_generate_title_uses_the_model(monkeypatch) -> None:
    class _Resp:
        content = "Immiscible Injector Producer Sweep"

    class _Model:
        async def ainvoke(self, _messages):
            return _Resp()

    import langchain.chat_models as cm

    monkeypatch.setattr(cm, "init_chat_model", lambda _id: _Model())
    title = await generate_session_title("anthropic:claude", "User: sweep\nAssistant: done")
    assert title == "Immiscible Injector Producer Sweep"


@pytest.mark.asyncio
async def test_generate_title_swallows_model_errors(monkeypatch) -> None:
    def _boom(_id):
        raise RuntimeError("no api key")

    import langchain.chat_models as cm

    monkeypatch.setattr(cm, "init_chat_model", _boom)
    assert await generate_session_title("anthropic:claude", "User: hi\nAssistant: yo") is None
