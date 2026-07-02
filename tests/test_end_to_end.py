"""End-to-end agent loop test against a scripted LLM and stub Julia."""

from __future__ import annotations

from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver

from fakes import (
    FakeJulia,
    make_fake_adapter,
    make_scripted_model,
    scripted_final,
    scripted_tool_call,
)
from jutul_agent.agent.builder import build_agent
from jutul_agent.agent.turns import TurnRunner, TurnTextDelta
from jutul_agent.session import Session
from jutul_agent.trace import TraceLog


async def _run_turn(agent, session: Session, prompt: str):
    runner = TurnRunner(agent, thread_id=session.session_id, trace=session.trace)
    return await runner.run_prompt(prompt)


async def test_agent_loop_drives_run_julia(tmp_path: Path) -> None:
    julia = FakeJulia(answers={"2 + 2": "4"})
    adapter = make_fake_adapter(tmp_path)
    session = Session.create(julia=julia, state_root=tmp_path, simulator=adapter)

    model = make_scripted_model(
        [
            scripted_tool_call(
                tool_name="run_julia",
                args={"code": "2 + 2"},
                tool_call_id="call_eval_1",
            ),
            scripted_final("The answer is 4."),
        ]
    )

    agent, _ = build_agent(session, model=model)
    result = await _run_turn(agent, session, "Compute 2 + 2 in Julia.")

    final_messages = result.messages
    last = final_messages[-1]
    assert "4" in str(getattr(last, "content", last))
    assert "2 + 2" in julia.calls

    session.finalize()
    kinds = _trace_kinds(session)
    assert "session_start" in kinds
    assert "message_user" in kinds
    assert kinds.count("tool_call") == 1
    assert kinds.count("tool_result") == 1
    assert "message_assistant" in kinds
    assert kinds[-1] == "session_end"


async def test_tool_output_is_not_streamed_as_assistant_prose(tmp_path: Path) -> None:
    """Real agent: a tool result must not be streamed back as assistant text.

    ``run.messages`` projects the tool node's ToolMessage with its full content
    as ``.text``; without the node filter that content was rendered as an
    assistant message (the skill/memory/file "dump"). Drive the real agent and
    assert the sentinel tool output never appears in the streamed prose."""

    sentinel = "SENTINEL_TOOL_OUTPUT_42_do_not_echo"
    julia = FakeJulia(answers={"probe()": sentinel})
    adapter = make_fake_adapter(tmp_path)
    session = Session.create(julia=julia, state_root=tmp_path, simulator=adapter)

    model = make_scripted_model(
        [
            scripted_tool_call(
                tool_name="run_julia",
                args={"code": "probe()"},
                tool_call_id="call_probe_1",
            ),
            scripted_final("Done — the probe ran successfully."),
        ]
    )
    agent, _ = build_agent(session, model=model)
    runner = TurnRunner(agent, thread_id=session.session_id, trace=session.trace)

    streamed: list[str] = []
    try:
        result = await runner.run_prompt(
            "Run the probe.",
            on_message=lambda msg: (
                streamed.append(msg.text)
                if isinstance(msg, TurnTextDelta) and msg.text
                else None
            ),
        )
    finally:
        session.finalize()

    streamed_text = "".join(streamed)
    assert sentinel not in streamed_text
    assert streamed_text == "Done — the probe ran successfully."
    # The tool genuinely ran and its result is in the final state — it just
    # arrives as a tool result, not as assistant prose.
    assert any(sentinel in str(getattr(m, "content", "")) for m in result.messages)


async def test_read_file_output_not_streamed_as_assistant_prose(tmp_path: Path) -> None:
    """The reported symptom: `read_file` / `grep` / `glob` results (file, skill,
    and memory text) rendered as assistant messages in the TUI. Drive the real
    agent through `read_file` on a workspace file and assert its body never
    appears in the streamed prose — only as the tool result."""

    sentinel = "SENTINEL_FILE_BODY_zzz_do_not_echo"
    (tmp_path / "notes.md").write_text(
        f"# heading\nline a\n{sentinel}\n## section\nmore text\n", encoding="utf-8"
    )
    julia = FakeJulia()
    adapter = make_fake_adapter(tmp_path)
    session = Session.create(julia=julia, state_root=tmp_path, simulator=adapter)

    model = make_scripted_model(
        [
            scripted_tool_call(
                tool_name="read_file",
                args={"file_path": "/notes.md"},
                tool_call_id="call_read_1",
            ),
            scripted_final("Read notes.md; nothing to add."),
        ]
    )
    agent, _ = build_agent(session, model=model)
    runner = TurnRunner(agent, thread_id=session.session_id, trace=session.trace)

    streamed: list[str] = []
    try:
        await runner.run_prompt(
            "Read notes.md",
            on_message=lambda msg: (
                streamed.append(msg.text)
                if isinstance(msg, TurnTextDelta) and msg.text
                else None
            ),
        )
    finally:
        session.finalize()

    streamed_text = "".join(streamed)
    assert sentinel not in streamed_text
    assert streamed_text == "Read notes.md; nothing to add."


async def test_execute_tool_requires_approval(tmp_path: Path) -> None:
    julia = FakeJulia()
    adapter = make_fake_adapter(tmp_path)
    session = Session.create(julia=julia, state_root=tmp_path, simulator=adapter)

    model = make_scripted_model(
        [
            scripted_tool_call(
                tool_name="execute",
                args={"command": "pwd"},
                tool_call_id="call_exec_1",
            )
        ]
    )

    agent, _ = build_agent(
        session,
        model=model,
        checkpointer=MemorySaver(),
    )
    runner = TurnRunner(agent, thread_id=session.session_id, trace=session.trace)

    try:
        result = await runner.run_prompt("Show the workspace directory.")

        assert len(result.interrupts) == 1
        interrupt = result.interrupts[0].value
        assert interrupt["action_requests"] == [
            {
                "name": "execute",
                "args": {"command": "pwd"},
                "description": "Run a shell command in the workspace.",
            }
        ]
        assert interrupt["review_configs"] == [
            {
                "action_name": "execute",
                "allowed_decisions": ["approve", "reject"],
            }
        ]
    finally:
        session.finalize()

    kinds = _trace_kinds(session)
    assert "hitl_request" in kinds


async def test_write_file_requires_approval(tmp_path: Path) -> None:
    julia = FakeJulia()
    adapter = make_fake_adapter(tmp_path)
    session = Session.create(julia=julia, state_root=tmp_path, simulator=adapter)

    model = make_scripted_model(
        [
            scripted_tool_call(
                tool_name="write_file",
                args={"file_path": "/notes.txt", "content": "hello\n"},
                tool_call_id="call_write_1",
            )
        ]
    )

    agent, _ = build_agent(
        session,
        model=model,
        checkpointer=MemorySaver(),
    )
    runner = TurnRunner(agent, thread_id=session.session_id, trace=session.trace)

    try:
        result = await runner.run_prompt("Create a note file.")

        assert len(result.interrupts) == 1
        interrupt = result.interrupts[0].value
        assert interrupt["action_requests"] == [
            {
                "name": "write_file",
                "args": {"file_path": "/notes.txt", "content": "hello\n"},
                "description": "Write a file in the workspace.",
            }
        ]
        assert interrupt["review_configs"] == [
            {
                "action_name": "write_file",
                "allowed_decisions": ["approve", "reject"],
            }
        ]
    finally:
        session.finalize()


async def test_edit_file_requires_approval(tmp_path: Path) -> None:
    julia = FakeJulia()
    adapter = make_fake_adapter(tmp_path)
    session = Session.create(julia=julia, state_root=tmp_path, simulator=adapter)

    model = make_scripted_model(
        [
            scripted_tool_call(
                tool_name="edit_file",
                args={
                    "file_path": "/notes.txt",
                    "old_string": "hello",
                    "new_string": "goodbye",
                },
                tool_call_id="call_edit_1",
            )
        ]
    )

    agent, _ = build_agent(
        session,
        model=model,
        checkpointer=MemorySaver(),
    )
    runner = TurnRunner(agent, thread_id=session.session_id, trace=session.trace)

    try:
        result = await runner.run_prompt("Update the note file.")

        assert len(result.interrupts) == 1
        interrupt = result.interrupts[0].value
        assert interrupt["action_requests"] == [
            {
                "name": "edit_file",
                "args": {
                    "file_path": "/notes.txt",
                    "old_string": "hello",
                    "new_string": "goodbye",
                },
                "description": "Edit a file in the workspace.",
            }
        ]
        assert interrupt["review_configs"] == [
            {
                "action_name": "edit_file",
                "allowed_decisions": ["approve", "reject"],
            }
        ]
    finally:
        session.finalize()


def _trace_kinds(session: Session) -> list[str]:
    db = session.state_dir / "trace.sqlite"
    log = TraceLog(db)
    try:
        return [e.kind for e in log.iter_events()]
    finally:
        log.close()


async def test_resumed_thread_restores_conversation(tmp_path: Path) -> None:
    """Same checkpoint db + same thread id ⇒ the next process sees the turn."""
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    adapter = make_fake_adapter(tmp_path)
    session = Session.create(julia=FakeJulia(), state_root=tmp_path, simulator=adapter)
    ckpt = session.state_dir / "checkpoints.sqlite"
    async with AsyncSqliteSaver.from_conn_string(str(ckpt)) as saver:
        agent, _ = build_agent(
            session, model=make_scripted_model([scripted_final("Hello!")]), checkpointer=saver
        )
        await _run_turn(agent, session, "Hi there")
    session.finalize()

    resumed = Session.resume(
        julia=FakeJulia(),
        simulator=adapter,
        session_id=session.session_id,
        state_root=tmp_path,
    )
    async with AsyncSqliteSaver.from_conn_string(str(ckpt)) as saver:
        agent2, _ = build_agent(
            resumed, model=make_scripted_model([scripted_final("Again")]), checkpointer=saver
        )
        state = await agent2.aget_state({"configurable": {"thread_id": resumed.session_id}})
        contents = [str(getattr(m, "content", "")) for m in state.values["messages"]]
        assert any("Hi there" in c for c in contents)
        assert any("Hello!" in c for c in contents)
    resumed.finalize()


async def test_delete_is_interrupted_before_it_runs(tmp_path: Path) -> None:
    """A recursive delete must reach the human before it touches the filesystem.

    The tool is the framework's, added to the file toolset by a version bump
    rather than by us, so the gate is worth asserting end to end and not just in
    the approval config.
    """
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (scratch / "run.jl").write_text("x = 1\n", encoding="utf-8")

    session = Session.create(
        julia=FakeJulia(), state_root=tmp_path, simulator=make_fake_adapter(tmp_path)
    )
    model = make_scripted_model(
        [
            scripted_tool_call(
                tool_name="delete",
                args={"file_path": str(scratch)},
                tool_call_id="call_delete_1",
            )
        ]
    )
    agent, _ = build_agent(session, model=model, checkpointer=MemorySaver())
    runner = TurnRunner(agent, thread_id=session.session_id, trace=session.trace)

    try:
        result = await runner.run_prompt("Clean up the scratch directory.")
        assert len(result.interrupts) == 1
        assert result.interrupts[0].value["action_requests"][0]["name"] == "delete"
        assert scratch.exists(), "delete ran before the human answered"
    finally:
        session.finalize()
