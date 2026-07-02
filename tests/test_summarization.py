"""Tests for the /context trigger figure and the manual /compact path.

Auto-compaction is deepagents' stock SummarizationMiddleware (covered by the
builder and recorder tests); this module covers the manual ``compact_thread``,
which drives the same engine non-mutatingly and recoverably.
"""

from __future__ import annotations

from pathlib import Path

from jutul_agent.agent.builder import build_agent
from jutul_agent.agent.summarization import auto_compact_trigger_tokens, compact_thread
from jutul_agent.agent.turns import TurnRunner
from jutul_agent.lab.fakes import FakeJulia, make_fake_adapter, make_scripted_model, scripted_final
from jutul_agent.session import Session


def test_auto_compact_trigger_tokens() -> None:
    # Mirrors deepagents' stock summarizer default (0.85 of the window).
    assert auto_compact_trigger_tokens(65_536) == int(65_536 * 0.85)
    assert auto_compact_trigger_tokens(200_000) == 170_000
    # No discoverable window → deepagents' fixed fallback.
    assert auto_compact_trigger_tokens(None) == 170_000


async def test_compact_thread_round_trip(tmp_path: Path) -> None:
    """/compact records a summarization event without rewriting the raw log."""
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    adapter = make_fake_adapter(tmp_path)
    session = Session.create(julia=FakeJulia(), state_root=tmp_path, simulator=adapter)
    # Substantial replies so the summary is meaningfully smaller than the turns
    # it replaces (a few words would summarize to something larger).
    finals = [scripted_final(f"answer {i}: " + "detail " * 80) for i in range(6)]
    ckpt = session.state_dir / "checkpoints.sqlite"
    async with AsyncSqliteSaver.from_conn_string(str(ckpt)) as saver:
        agent, backend = build_agent(session, model=make_scripted_model(finals), checkpointer=saver)
        runner = TurnRunner(agent, thread_id=session.session_id, trace=session.trace)
        for i in range(6):
            await runner.run_prompt(f"question {i}")

        result = await compact_thread(
            agent,
            thread_id=session.session_id,
            model=make_scripted_model([scripted_final("SUMMARY of earlier work")]),
            backend=backend,
            trace=session.trace,
        )
        assert result is not None
        # 12 messages (6 turns), keep 8 → summarize the oldest 4.
        assert result.messages_summarized == 4
        assert result.messages_kept == 8
        assert result.freed_tokens > 0
        assert result.offloaded is True

        state = await agent.aget_state({"configurable": {"thread_id": session.session_id}})
        # Non-mutating: the raw conversation log is preserved in full...
        assert len(state.values["messages"]) == 12
        # ...and the compaction is recorded as an event the next turn applies.
        assert state.values["_summarization_event"]["cutoff_index"] == 4

    # The offloaded turns were written somewhere recoverable under the session.
    offloaded = list((session.state_dir / "conversation_history").glob("*.md"))
    assert offloaded and any("question" in p.read_text(encoding="utf-8") for p in offloaded)

    events = [e for e in session.trace.iter_events() if e.kind == "context_compaction"]
    assert len(events) == 1 and events[0].payload.get("manual") is True
    session.finalize()


async def test_compact_thread_skips_short_threads() -> None:
    class _NoState:
        pass

    assert (
        await compact_thread(_NoState(), thread_id="t", model=None, backend=None, trace=None)
    ) is None
