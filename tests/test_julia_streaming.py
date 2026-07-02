"""Live streaming of run_julia output to the UI.

Covers the *producer* half: ``run_julia`` forwards each kernel output fragment
to langgraph's active tool-output-delta writer, so the TUI's existing
``output_deltas`` -> ``TurnToolEvent(delta)`` -> ``ToolBlock.append_output`` path
renders it live. The display half (terminal-correct rendering of those deltas)
lives in ``test_widgets.py``; the turns -> delta plumbing in ``test_turns.py``.
"""

from __future__ import annotations

from pathlib import Path

from langgraph.pregel._tools import _tool_call_writer

from jutul_agent.agent.tools import make_run_julia_tool
from jutul_agent.julia.session import EvalResult
from jutul_agent.lab.fakes import FakeJulia, make_fake_adapter
from jutul_agent.session import Session


def _session(julia: FakeJulia, tmp_path: Path) -> Session:
    return Session.create(
        julia=julia,
        state_root=tmp_path,
        simulator=make_fake_adapter(tmp_path),
        session_id="stream-test",
    )


async def _call(tool, **args) -> str:
    msg = await tool.ainvoke({"type": "tool_call", "name": "run_julia", "id": "c1", "args": args})
    return str(getattr(msg, "content", msg))


async def test_run_julia_streams_chunks_as_output_deltas(tmp_path: Path) -> None:
    """With a delta writer active (as inside a streaming graph), each kernel
    output fragment is forwarded as a tool-output delta, in order."""

    captured: list[str] = []
    token = _tool_call_writer.set(captured.append)
    try:
        julia = FakeJulia(stream_chunks=["tick 1\n", "tick 2\n"], answers={"work()": "done"})
        tool = make_run_julia_tool(_session(julia, tmp_path))
        out = await _call(tool, code="work()")
    finally:
        _tool_call_writer.reset(token)

    assert captured == ["tick 1\n", "tick 2\n"]
    assert out.strip() == "done"  # the final result is unchanged by streaming


async def test_run_julia_runs_without_a_delta_writer(tmp_path: Path) -> None:
    """Standalone (no streaming graph -> no writer in context): the eval still
    runs and returns its result, just without live deltas."""

    julia = FakeJulia(stream_chunks=["noise\n"], answers={"work()": "done"})
    tool = make_run_julia_tool(_session(julia, tmp_path))
    out = await _call(tool, code="work()")
    assert out.strip() == "done"


async def test_run_julia_keeps_output_when_the_eval_errors(tmp_path: Path) -> None:
    """On an error, the tool surfaces the pre-throw output and the error, not the
    error alone."""

    julia = FakeJulia(
        eval_handler=lambda _code: EvalResult(
            output="progress: step 1", error="boom", stdout="progress: step 1"
        )
    )
    tool = make_run_julia_tool(_session(julia, tmp_path))
    out = await _call(tool, code="boom()")
    assert "progress: step 1" in out  # breadcrumb is not dropped
    assert "ERROR: boom" in out
