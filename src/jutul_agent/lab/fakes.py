"""Shared test fakes: chat model, Julia session, simulator adapter, v3 stream.

These let the test suite drive the real agent runtime (LangGraph + deepagents
+ middleware + tools) end-to-end without external dependencies on a provider
API or a running Julia process.

The ``v3_*_event`` helpers form a small script DSL; ``ScriptedV3Agent``
parses a list of those events into the same typed-projection shape that
``langgraph.stream.run_stream.AsyncGraphRunStream`` exposes (``.messages``,
``.tool_calls``, ``.interrupts()``, ``.output()``) so the production
``TurnRunner`` can consume scripted and real agents through one interface.
"""

from __future__ import annotations

import inspect
import uuid
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langgraph.types import Command

from jutul_agent.julia.session import EvalResult
from jutul_agent.juliakernel.result import OutputChunk
from jutul_agent.simulators.base import SimulatorAdapter
from jutul_agent.trace import Event


class ScriptedChatModel(FakeMessagesListChatModel):
    """Replays a scripted sequence of ``AIMessage``s through the agent loop."""

    def bind_tools(self, tools, **_kwargs):  # type: ignore[override]
        return self


def scripted_tool_call(
    *,
    tool_name: str,
    args: dict,
    tool_call_id: str | None = None,
    content: str = "",
) -> AIMessage:
    """Build an ``AIMessage`` that requests one tool call."""

    return AIMessage(
        content=content,
        tool_calls=[
            {
                "id": tool_call_id or f"call_{uuid.uuid4().hex[:12]}",
                "name": tool_name,
                "args": args,
            }
        ],
    )


def scripted_final(content: str) -> AIMessage:
    return AIMessage(content=content)


def make_scripted_model(responses: Sequence[BaseMessage]) -> ScriptedChatModel:
    return ScriptedChatModel(responses=list(responses))


# ---------------------------------------------------------------------------
# Trace event factory (shared by transcript renderer tests).


def make_event(eid: int, kind: str, payload: dict) -> Event:
    return Event(
        id=eid,
        timestamp=f"2026-05-19T10:00:{eid:02d}+00:00",
        kind=kind,
        payload=payload,
    )


# ---------------------------------------------------------------------------
# HITL interrupt helpers.


@dataclass(frozen=True)
class Interrupt:
    """Minimal stand-in for LangGraph's interrupt object."""

    id: str
    value: dict[str, Any]


# ---------------------------------------------------------------------------
# v3 event-stream script DSL. The helpers below produce small dicts; a
# ``ScriptedV3Agent`` reads them and synthesizes typed projections.


def v3_message_event(payload: Any) -> dict[str, Any]:
    """A ``messages`` event carrying one BaseMessage or content-block dict."""

    return {"kind": "message", "payload": payload}


def v3_tool_event(payload: dict[str, Any]) -> dict[str, Any]:
    """A ``tools`` event (``tool-started`` / ``tool-finished`` / ``tool-error``)."""

    return {"kind": "tool", "payload": payload}


def v3_values_event(
    messages: list[BaseMessage],
    *,
    interrupts: Sequence[Any] = (),
) -> dict[str, Any]:
    """A ``values`` event carrying full state and any interrupts."""

    return {
        "kind": "values",
        "messages": list(messages),
        "interrupts": list(interrupts),
    }


def tool_call_events(
    *,
    tool_name: str,
    tool_call_id: str,
    args: dict[str, Any],
    output: str,
    final_text: str,
    final_message: AIMessage | None = None,
) -> list[dict[str, Any]]:
    """Build v3 events for one tool call round-trip followed by a final message."""

    human = HumanMessage(content="hello")
    ai_with_tool = AIMessage(
        content="",
        tool_calls=[{"id": tool_call_id, "name": tool_name, "args": args}],
    )
    tool_msg = ToolMessage(content=output, tool_call_id=tool_call_id, name=tool_name)
    final = final_message if final_message is not None else AIMessage(content=final_text)
    return [
        v3_message_event(human),
        v3_message_event(ai_with_tool),
        v3_tool_event(
            {
                "event": "tool-started",
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "input": args,
            }
        ),
        v3_tool_event({"event": "tool-finished", "tool_call_id": tool_call_id, "output": output}),
        v3_message_event(tool_msg),
        v3_message_event(final),
        v3_values_event([human, ai_with_tool, tool_msg, final]),
    ]


# ---------------------------------------------------------------------------
# Typed-projection mocks.


class _AsyncDeltaIter:
    """Awaitable+async-iterable view over a list of deltas.

    Iterating yields deltas in order; awaiting joins them. Mirrors the
    pattern of langchain's ``AsyncTextProjection``.
    """

    def __init__(self, items: list[Any], *, join: bool = False) -> None:
        self._items = items
        self._join = join

    def __aiter__(self):
        async def it():
            for item in self._items:
                yield item

        return it()

    def __await__(self):
        async def _result():
            if self._join:
                return "".join(self._items)
            return list(self._items)

        return _result().__await__()


class FakeMessageStream:
    """Per-message typed-projection mock (text/reasoning/tool_calls/output).

    ``node`` mirrors ``langgraph_node`` on the real ``ChatModelStream``: the
    model node (``"model"``) carries the assistant's visible turn, while tool
    results surface under ``"tools"``. The turn runner keys on it to avoid
    dumping tool output into the chat as prose.
    """

    def __init__(
        self,
        *,
        text_deltas: list[str],
        reasoning_deltas: list[str],
        tool_call_chunks: list[dict[str, Any]],
        tool_calls_final: list[dict[str, Any]],
        output: AIMessage | AIMessageChunk,
        node: str = "model",
    ) -> None:
        self.text = _AsyncDeltaIter(text_deltas, join=True)
        self.reasoning = _AsyncDeltaIter(reasoning_deltas, join=True)
        self.tool_calls = _AsyncToolCallChunks(tool_call_chunks, tool_calls_final)
        self.node = node
        self._output = output

    def __await__(self):
        async def _resolve():
            return self._output

        return _resolve().__await__()

    @property
    def output(self):
        # Mirror the live API: `await stream.output` returns the AIMessage.
        return _AwaitableValue(self._output)


class _AsyncToolCallChunks:
    """Iterable for chunk deltas + awaitable for finalized tool calls."""

    def __init__(self, chunks: list[dict[str, Any]], final: list[dict[str, Any]]) -> None:
        self._chunks = chunks
        self._final = final

    def __aiter__(self):
        async def it():
            for chunk in self._chunks:
                yield chunk

        return it()

    def __await__(self):
        async def _result():
            return list(self._final)

        return _result().__await__()


class _AwaitableValue:
    def __init__(self, value: Any) -> None:
        self._value = value

    def __await__(self):
        async def _resolve():
            return self._value

        return _resolve().__await__()


class FakeToolCallStream:
    """Per-tool typed-projection mock."""

    def __init__(
        self,
        *,
        tool_call_id: str | None,
        tool_name: str,
        input: dict[str, Any] | None,
        output: Any = None,
        error: str | None = None,
        output_deltas: list[Any] | None = None,
    ) -> None:
        self.tool_call_id = tool_call_id
        self.tool_name = tool_name
        self.input = input
        self.output_deltas = _AsyncDeltaIter(list(output_deltas or []))
        self.output = output
        self.error = error
        self.completed = True


class FakeTypedRun:
    """Mock ``AsyncGraphRunStream`` with the projections TurnRunner consumes."""

    def __init__(
        self,
        *,
        messages: list[FakeMessageStream],
        tool_calls: list[FakeToolCallStream],
        interrupts: list[Any],
        output: dict[str, Any],
    ) -> None:
        self._messages = messages
        self._tool_calls = tool_calls
        self._interrupts = interrupts
        self._output = output

    @property
    def messages(self):
        items = self._messages

        class _Iter:
            def __aiter__(self):
                async def it():
                    for m in items:
                        yield m

                return it()

        return _Iter()

    @property
    def tool_calls(self):
        items = self._tool_calls

        class _Iter:
            def __aiter__(self):
                async def it():
                    for c in items:
                        yield c

                return it()

        return _Iter()

    async def interrupts(self) -> list[Any]:
        return list(self._interrupts)

    async def output(self) -> dict[str, Any]:
        return dict(self._output)


def _build_typed_run(events: Iterable[dict[str, Any]]) -> FakeTypedRun:
    """Translate the v3 event script DSL into a typed-projection mock."""

    messages: list[FakeMessageStream] = []
    tool_calls: list[FakeToolCallStream] = []
    interrupts: list[Any] = []
    seen_interrupt_ids: set[str] = set()
    final_messages: list[BaseMessage] = []
    tool_call_buf: dict[str, dict[str, Any]] = {}

    # Active "open" message under construction (for content-block streams
    # that arrive across multiple events).
    open_text: list[str] = []
    open_reasoning: list[str] = []
    open_tool_chunks: list[dict[str, Any]] = []
    has_open = False

    def _flush_open() -> None:
        nonlocal has_open, open_text, open_reasoning, open_tool_chunks
        if not has_open:
            return
        joined = "".join(open_text)
        final = AIMessageChunk(content=joined, chunk_position="last")
        finalized_calls = [chunk for chunk in open_tool_chunks if chunk.get("name")]
        messages.append(
            FakeMessageStream(
                text_deltas=list(open_text),
                reasoning_deltas=list(open_reasoning),
                tool_call_chunks=list(open_tool_chunks),
                tool_calls_final=finalized_calls,
                output=final,
            )
        )
        open_text = []
        open_reasoning = []
        open_tool_chunks = []
        has_open = False

    for event in events:
        kind = event.get("kind")

        if kind == "message":
            payload = event.get("payload")

            if isinstance(payload, ToolMessage):
                # The real ``run.messages`` projection surfaces a ToolMessage as
                # its own stream (``node="tools"``) whose ``.text`` is the full
                # tool result. Reproduce that so the turn runner's node filter is
                # exercised. The result must render as a tool card, never prose.
                _flush_open()
                content = payload.content if isinstance(payload.content, str) else ""
                messages.append(
                    FakeMessageStream(
                        text_deltas=[content] if content else [],
                        reasoning_deltas=[],
                        tool_call_chunks=[],
                        tool_calls_final=[],
                        output=AIMessage(content=content),
                        node="tools",
                    )
                )
                continue

            if isinstance(payload, HumanMessage):
                _flush_open()
                continue

            if isinstance(payload, AIMessageChunk):
                # Coalesce consecutive chunks into one streamed message; the
                # ``chunk_position="last"`` marker (or the next non-chunk
                # event) closes the stream.
                content = payload.content if isinstance(payload.content, str) else ""
                if content:
                    open_text.append(content)
                for call in payload.tool_calls or []:
                    open_tool_chunks.append(
                        {
                            "id": call.get("id"),
                            "name": call.get("name"),
                            "args": call.get("args") or {},
                        }
                    )
                has_open = True
                if getattr(payload, "chunk_position", None) == "last":
                    _flush_open()
                continue

            if isinstance(payload, AIMessage):
                # A complete pre-built message: emit it as a single-delta stream.
                _flush_open()
                content = payload.content if isinstance(payload.content, str) else ""
                tool_call_chunks: list[dict[str, Any]] = []
                tool_calls_final: list[dict[str, Any]] = []
                for call in payload.tool_calls or []:
                    chunk = {
                        "id": call.get("id"),
                        "name": call.get("name"),
                        "args": call.get("args") or {},
                    }
                    tool_call_chunks.append(chunk)
                    tool_calls_final.append(chunk)
                messages.append(
                    FakeMessageStream(
                        text_deltas=[content] if content else [],
                        reasoning_deltas=[],
                        tool_call_chunks=tool_call_chunks,
                        tool_calls_final=tool_calls_final,
                        output=payload,
                    )
                )
                continue

            if isinstance(payload, dict):
                event_type = payload.get("event")
                if event_type == "content-block-delta":
                    delta = payload.get("delta") or {}
                    dtype = delta.get("type")
                    if dtype == "text-delta":
                        open_text.append(str(delta.get("text", "")))
                        has_open = True
                    elif dtype == "reasoning-delta":
                        open_reasoning.append(str(delta.get("reasoning", "")))
                        has_open = True
                elif event_type == "content-block-finish":
                    block = payload.get("content") or {}
                    if block.get("type") in {"tool_call", "server_tool_call"}:
                        args = block.get("args")
                        if not isinstance(args, dict):
                            args = {"value": args} if args is not None else {}
                        chunk = {
                            "id": block.get("id"),
                            "name": block.get("name"),
                            "args": args,
                        }
                        open_tool_chunks.append(chunk)
                        has_open = True
                elif event_type == "message-finish":
                    _flush_open()
                continue

        elif kind == "tool":
            payload = event.get("payload") or {}
            tool_event = payload.get("event")
            tool_id = payload.get("tool_call_id")
            if tool_event == "tool-started":
                tool_call_buf[tool_id] = {
                    "tool_call_id": tool_id,
                    "tool_name": payload.get("tool_name") or "tool",
                    "input": payload.get("input"),
                    "output": None,
                    "error": None,
                    "deltas": [],
                }
            elif tool_event == "tool-output-delta":
                buf = tool_call_buf.get(tool_id)
                if buf is not None:
                    delta = payload.get("delta")
                    if delta is not None:
                        buf.setdefault("deltas", []).append(delta)
            elif tool_event == "tool-finished":
                buf = tool_call_buf.pop(tool_id, None)
                if buf is None:
                    buf = {
                        "tool_call_id": tool_id,
                        "tool_name": payload.get("tool_name") or "tool",
                        "input": None,
                    }
                tool_calls.append(
                    FakeToolCallStream(
                        tool_call_id=buf["tool_call_id"],
                        tool_name=buf["tool_name"],
                        input=buf.get("input"),
                        output=payload.get("output"),
                        output_deltas=buf.get("deltas") or [],
                    )
                )
            elif tool_event == "tool-error":
                buf = tool_call_buf.pop(tool_id, None) or {
                    "tool_call_id": tool_id,
                    "tool_name": payload.get("tool_name") or "tool",
                    "input": None,
                }
                tool_calls.append(
                    FakeToolCallStream(
                        tool_call_id=buf["tool_call_id"],
                        tool_name=buf["tool_name"],
                        input=buf.get("input"),
                        error=str(payload.get("message") or "tool error"),
                    )
                )

        elif kind == "values":
            final_messages = list(event.get("messages") or [])
            for itp in event.get("interrupts") or ():
                itp_id = str(getattr(itp, "id", "") or "")
                if itp_id and itp_id not in seen_interrupt_ids:
                    seen_interrupt_ids.add(itp_id)
                    interrupts.append(itp)

    _flush_open()
    return FakeTypedRun(
        messages=messages,
        tool_calls=tool_calls,
        interrupts=interrupts,
        output={"messages": final_messages},
    )


class ScriptedV3Agent:
    """Stub agent whose ``astream_events`` returns a typed-projection mock."""

    def __init__(
        self,
        events: Iterable[dict[str, Any]] | Callable[[Any], Iterable[dict[str, Any]]],
    ) -> None:
        self._events = events
        self.inputs: list[Any] = []
        # Interrupts left pending in the (mock) graph state, for aget_state: a prompt
        # that interrupts records them, a resume clears them. Lets a test exercise the
        # reconnect path that re-surfaces a pending approval.
        self._pending_interrupts: list[Any] = []

    @property
    def resume_inputs(self) -> list[Command]:
        return [inp for inp in self.inputs if isinstance(inp, Command)]

    async def astream_events(self, stream_input, **_kwargs):
        self.inputs.append(stream_input)
        source = self._events(stream_input) if callable(self._events) else self._events
        events = list(source)
        if isinstance(stream_input, Command):
            self._pending_interrupts = []
        else:
            self._pending_interrupts = [
                itp
                for event in events
                if event.get("kind") == "values"
                for itp in (event.get("interrupts") or [])
            ]
        return _build_typed_run(events)

    async def aget_state(self, _config):
        """Mock state snapshot exposing any interrupts left pending (see astream_events)."""
        return SimpleNamespace(interrupts=list(self._pending_interrupts), tasks=())


def interrupt_agent(
    *,
    tool_name: str = "execute",
    tool_args: dict[str, Any] | None = None,
    allowed_decisions: list[str] | None = None,
    description: str | None = None,
    interrupt_id: str = "interrupt-1",
) -> ScriptedV3Agent:
    args = tool_args or {"command": "ls -la"}
    allowed = allowed_decisions or ["approve", "reject", "respond"]
    desc = description or "Review the requested tool action."
    interrupt = Interrupt(
        id=interrupt_id,
        value={
            "action_requests": [{"name": tool_name, "args": args, "description": desc}],
            "review_configs": [{"action_name": tool_name, "allowed_decisions": allowed}],
        },
    )

    def _events(stream_input):
        if hasattr(stream_input, "resume"):
            final = AIMessage(content="approval handled")
            return [
                v3_message_event(final),
                v3_values_event([final]),
            ]
        human = HumanMessage(content="approval please")
        return [v3_values_event([human], interrupts=[interrupt])]

    return ScriptedV3Agent(_events)


def echo_agent() -> ScriptedV3Agent:
    def _events(stream_input):
        human = HumanMessage(content=str(stream_input))
        final = AIMessage(content=f"Echo: {stream_input}")
        return [
            v3_message_event(human),
            v3_message_event(final),
            v3_values_event([human, final]),
        ]

    return ScriptedV3Agent(_events)


def streaming_agent() -> ScriptedV3Agent:
    def _events(stream_input):
        human = HumanMessage(content=str(stream_input))
        final = AIMessage(content="Hello world")
        return [
            v3_message_event(AIMessageChunk(content="Hello ")),
            v3_message_event(AIMessageChunk(content="world", chunk_position="last")),
            v3_values_event([human, final]),
        ]

    return ScriptedV3Agent(_events)


def reasoning_agent() -> ScriptedV3Agent:
    def _events(stream_input):
        human = HumanMessage(content=str(stream_input))
        final = AIMessage(content="Answer ready")
        return [
            v3_values_event([human]),
            v3_message_event(
                {
                    "event": "content-block-delta",
                    "delta": {
                        "type": "reasoning-delta",
                        "reasoning": "Checking simulator state.",
                    },
                }
            ),
            v3_message_event(
                {
                    "event": "content-block-delta",
                    "delta": {"type": "text-delta", "text": "Answer ready"},
                }
            ),
            v3_message_event({"event": "message-finish"}),
            v3_values_event([human, final]),
        ]

    return ScriptedV3Agent(_events)


class FakeJulia:
    """Synchronous-stub ``JuliaSession`` for tests that don't need real Julia."""

    def __init__(
        self,
        *,
        pkgdir: dict[str, Path] | None = None,
        answers: dict[str, str] | None = None,
        eval_handler: Callable[[str], EvalResult | Awaitable[EvalResult]] | None = None,
        stream_chunks: Sequence[str] | None = None,
    ) -> None:
        self._pkgdir = pkgdir or {}
        self._answers = answers or {}
        self._eval_handler = eval_handler
        # When set, eval emits these as live stdout fragments through ``on_chunk``
        # before returning, which lets a test exercise the streaming path.
        self._stream_chunks = list(stream_chunks or [])
        self.calls: list[str] = []
        self.reset_count: int = 0
        self.restart_count: int = 0

    async def __aenter__(self) -> FakeJulia:
        return self

    async def __aexit__(self, *_) -> None:
        return None

    async def eval(self, code: str, on_chunk=None) -> EvalResult:
        self.calls.append(code)
        if on_chunk is not None:
            for text in self._stream_chunks:
                on_chunk(OutputChunk(text=text, stream="stdout"))
        if self._eval_handler is not None:
            result = self._eval_handler(code)
            if inspect.iscoroutine(result):
                return await result
            return result
        if "pkgdir" in code:
            for name, path in self._pkgdir.items():
                if f"pkgdir({name})" in code:
                    return EvalResult(output=str(path))
        if code in self._answers:
            return EvalResult(output=self._answers[code])
        return EvalResult(output="")

    async def reset(self) -> EvalResult:
        self.reset_count += 1
        return EvalResult(output="reset")

    async def restart(self) -> None:
        self.restart_count += 1


def make_fake_adapter(
    tmp_path: Path,
    *,
    name: str = "fakesim",
    display_name: str = "FakeSim",
    package: str = "FakePkg",
    warm_package: str = "",
) -> SimulatorAdapter:
    module_dir = tmp_path / "sim"
    (module_dir / "skills").mkdir(parents=True, exist_ok=True)
    return SimulatorAdapter(
        name=name,
        display_name=display_name,
        module_dir=module_dir,
        package_imports=(package,),
        primary_package=package,
        domain_hints="",
        warm_package=warm_package,
    )
