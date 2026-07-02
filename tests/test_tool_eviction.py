"""Large tool results are offloaded to the backend, not dumped into context.

The framework evicts an oversized tool result to ``/large_tool_results`` under
the backend's artifacts root and replaces it inline with a recoverable pointer.
That offload needs a writable artifacts root; this also guards that wiring,
which silently failed when it defaulted to the filesystem root ("/").
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from jutul_agent.agent.builder import build_agent
from jutul_agent.julia.session import EvalResult
from jutul_agent.lab.fakes import (
    FakeJulia,
    make_fake_adapter,
    make_scripted_model,
    scripted_final,
    scripted_tool_call,
)
from jutul_agent.session import Session


async def test_large_tool_result_offloaded_to_backend(tmp_path: Path) -> None:
    # run_julia returns a result far above the ~80k-char eviction threshold.
    big = "x" * 120_000
    julia = FakeJulia(eval_handler=lambda code: EvalResult(output=big))
    adapter = make_fake_adapter(tmp_path)
    session = Session.create(julia=julia, state_root=tmp_path, simulator=adapter)
    model = make_scripted_model(
        [
            scripted_tool_call(tool_name="run_julia", args={"code": "big()"}, tool_call_id="t1"),
            scripted_final("done"),
        ]
    )
    ckpt = session.state_dir / "checkpoints.sqlite"
    async with AsyncSqliteSaver.from_conn_string(str(ckpt)) as saver:
        agent, _ = build_agent(session, model=model, checkpointer=saver, approval_mode="auto")
        cfg = {"configurable": {"thread_id": session.session_id}}
        await agent.ainvoke({"messages": [HumanMessage(content="run it")]}, cfg)
        state = await agent.aget_state(cfg)

    results = [
        m for m in state.values["messages"] if isinstance(m, ToolMessage) and m.name == "run_julia"
    ]
    assert results
    inline = str(results[-1].content)
    # The dump is not inline; a recoverable pointer replaces it.
    assert len(inline) < len(big)
    assert "too large" in inline.lower()
    # The full result was written under the session's artifacts root.
    offloaded = list((session.state_dir / "large_tool_results").glob("*"))
    assert offloaded and any(big in p.read_text(encoding="utf-8") for p in offloaded)
    session.finalize()
