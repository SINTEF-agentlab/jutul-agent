"""Integration test: real agent + Textual TUI, driven by a scripted LLM."""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.widgets import Markdown

from _tui import submit_prompt
from jutul_agent.agent.builder import build_agent
from jutul_agent.interfaces.tui import TUIApp
from jutul_agent.interfaces.tui.widgets import MessageBlock, ToolBlock
from jutul_agent.lab.fakes import (
    FakeJulia,
    make_fake_adapter,
    make_scripted_model,
    scripted_final,
    scripted_tool_call,
)
from jutul_agent.session import Session
from jutul_agent.trace import TraceLog


async def test_multi_tool_turn_renders_full_block_sequence(
    session_with_pkg: tuple[Session, object],
) -> None:
    session, julia = session_with_pkg

    model = make_scripted_model(
        [
            scripted_tool_call(
                tool_name="run_julia",
                args={"code": "2 + 2"},
                tool_call_id="call_eval_1",
                content="Let me compute that.",
            ),
            scripted_tool_call(
                tool_name="run_julia",
                args={"code": "print(pkgdir(FakePkg))"},
                tool_call_id="call_eval_2",
            ),
            scripted_final("Result is 4 and FakePkg lives at the printed path."),
        ]
    )

    agent, _ = build_agent(session, model=model)
    app = TUIApp(agent=agent, session=session, model_label="fake:script")

    async with app.run_test() as pilot:
        await submit_prompt(pilot, "compute")

        blocks = list(app.query(MessageBlock))
        titles = [b.border_title for b in blocks]
        tool_blocks = list(app.query(ToolBlock))
        markdown_count = len(list(app.query(Markdown)))
        assistant_text = "\n".join(b._content for b in blocks if b.has_class("assistant"))

    assert "Welcome" in titles
    assert "You" in titles
    assert "Assistant" in titles
    assert [block.border_title for block in tool_blocks] == [
        "Julia",
        "Julia",
    ]
    assert markdown_count >= 4
    assert "2 + 2" in julia.calls

    # The pkgdir path is a distinctive tool result the assistant never echoes;
    # it must surface only in the tool cards, never dumped into an Assistant
    # block. (The "2 + 2" -> "4" result is intentionally not checked here: the
    # assistant legitimately *says* "Result is 4" in its own prose.)
    pkg_path = str(julia._pkgdir["FakePkg"])
    assert any(pkg_path in block._output for block in tool_blocks)
    assert pkg_path not in assistant_text

    session.finalize()
    trace_text = _trace_payloads_concat(session)
    assert "2 + 2" in trace_text
    assert "pkgdir(FakePkg)" in trace_text


async def test_run_julia_output_streams_live_into_tool_card(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end live streaming: a run_julia producing carriage-return progress
    delivers each fragment to the tool card *before* the call finishes, through the
    real graph (tool -> delta writer -> output_deltas -> TurnToolEvent -> ToolBlock).

    The final ``set_result`` overwrites the streamed buffer, so we spy on
    ``append_output`` to observe that the deltas actually flowed live (and in
    order), then confirm the rendered text collapsed the bar like a terminal.
    """

    chunks = [
        "Progress   0%|        |\r",
        "Progress  50%|####    |\r",
        "Progress 100%|########|\n",
    ]
    julia = FakeJulia(stream_chunks=chunks, answers={"solve()": "Progress 100%|########|"})
    session = Session.create(
        julia=julia,
        state_root=tmp_path,
        simulator=make_fake_adapter(tmp_path),
        session_id="stream-e2e",
    )

    recorded: list[str] = []
    original = ToolBlock.append_output

    async def _spy(self: ToolBlock, delta: str) -> None:
        recorded.append(delta)
        await original(self, delta)

    monkeypatch.setattr(ToolBlock, "append_output", _spy)

    model = make_scripted_model(
        [
            scripted_tool_call(
                tool_name="run_julia",
                args={"code": "solve()"},
                tool_call_id="call_solve",
            ),
            scripted_final("Solve finished."),
        ]
    )
    agent, _ = build_agent(session, model=model)
    app = TUIApp(agent=agent, session=session, model_label="fake:script")

    async with app.run_test() as pilot:
        await submit_prompt(pilot, "solve")
        tool_blocks = list(app.query(ToolBlock))
        final_output = tool_blocks[0]._output if tool_blocks else ""

    session.finalize()

    # Each fragment reached the card live, in order, so the streaming path is wired.
    assert recorded == chunks
    # And the final card shows one collapsed bar, not three stacked lines.
    assert final_output == "Progress 100%|########|"


def _trace_payloads_concat(session: Session) -> str:
    log = TraceLog(session.state_dir / "trace.sqlite")
    try:
        return "\n".join(repr(e.payload) for e in log.iter_events())
    finally:
        log.close()


async def test_status_revert_to_ready_after_turn(
    session_with_pkg: tuple[Session, object],
) -> None:
    session, _ = session_with_pkg

    model = make_scripted_model([scripted_final("hello back")])
    agent, _ = build_agent(session, model=model)
    app = TUIApp(agent=agent, session=session)

    async with app.run_test() as pilot:
        await submit_prompt(pilot, "hi")

        prompt_input = app.query_one("#prompt")
        assert prompt_input.disabled is False  # type: ignore[attr-defined]

    session.finalize()
