"""Live end-to-end smoke: real LLM coordinating workspace files and Julia."""

from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path

import pytest

from makker.agent.builder import build_agent
from makker.agent.turns import TurnRunner
from makker.juliakernel import JuliaKernel, KernelConfig
from makker.paths import set_workspace_root
from makker.session import Session
from makker.simulators.jutuldarcy import JUTULDARCY
from makker.trace import TraceLog

_PROVIDER_KEYS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")
_LIVE_MODEL_ENV = "MAKKER_LIVE_MODEL"
_DEFAULT_LIVE_MODEL = "openai:gpt-5.4-mini"
_EXPECTED_SUM = "105"


def _has_julia() -> bool:
    return shutil.which("julia") is not None


def _has_any_provider_key() -> bool:
    return any(os.environ.get(k) for k in _PROVIDER_KEYS)


pytestmark = pytest.mark.skipif(
    not (_has_julia() and _has_any_provider_key()),
    reason=("Requires Julia + one of: " + ", ".join(_PROVIDER_KEYS)),
)


async def _attempt(tmp_path: Path) -> None:
    from langchain.chat_models import init_chat_model

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "data.jl").write_text("sum([7, 14, 21, 28, 35])\n", encoding="utf-8")
    set_workspace_root(workspace)

    model_label = os.environ.get(_LIVE_MODEL_ENV, _DEFAULT_LIVE_MODEL)
    model = init_chat_model(model_label, temperature=0)

    config = KernelConfig()
    async with JuliaKernel(config) as julia:
        session = Session.create(julia=julia, state_root=tmp_path / "state", simulator=JUTULDARCY)
        try:
            agent, _ = build_agent(session, model=model)
            runner = TurnRunner(agent, thread_id=session.session_id, trace=session.trace)
            prompt = (
                "Read the workspace file `/data.jl` with read_file, then use run_julia "
                "to evaluate its contents as Julia code. Reply with the numeric result."
            )
            result = await runner.run_prompt(prompt)
            final = "\n".join(str(getattr(m, "content", m)) for m in result.messages)
            assert _EXPECTED_SUM in final, f"expected {_EXPECTED_SUM} in final message: {final!r}"
        finally:
            session.finalize()

        log = TraceLog(session.state_dir / "trace.sqlite")
        try:
            tool_names = [
                event.payload.get("name")
                for event in log.iter_events()
                if event.kind == "tool_call"
            ]
        finally:
            log.close()

        assert "read_file" in tool_names, f"trace tool calls: {tool_names}"
        assert "run_julia" in tool_names, f"trace tool calls: {tool_names}"


async def test_live_workspace_read_and_run_julia(tmp_path: Path) -> None:
    for attempt in range(2):
        try:
            await _attempt(tmp_path)
            return
        except AssertionError:
            if attempt == 0:
                await asyncio.sleep(2)
                continue
            raise
