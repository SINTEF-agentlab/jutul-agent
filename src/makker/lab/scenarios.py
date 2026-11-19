"""Named TUI scenarios: a recipe of (agent, interactions) the lab can render.

Each scenario builds a scripted agent and a short sequence of interactions that put
the TUI into a state worth looking at: a tool call, a streamed answer, an approval
prompt, an error, a tiny terminal, and so on. The lab (:mod:`makker.lab.tui`)
drives each one headlessly and captures what it renders, so an agent can see the UI
and iterate without a live session.

Add a scenario by calling :func:`scenario`. Keep them small and deterministic.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from makker.lab.fakes import (
    ScriptedV3Agent,
    echo_agent,
    interrupt_agent,
    reasoning_agent,
    streaming_agent,
    tool_call_events,
    v3_message_event,
    v3_tool_event,
    v3_values_event,
)

# A step is either a prompt to submit (str) or an async callable taking the pilot.
Step = str | Callable[[Any], Any]


@dataclass(frozen=True)
class Scenario:
    """One renderable TUI state: an agent plus the interactions to reach it."""

    name: str
    description: str
    build_agent: Callable[[], Any] | None
    steps: tuple[Step, ...] = ()
    size: tuple[int, int] = (110, 32)
    model_label: str = "openai:gpt-5.4-mini"
    tags: tuple[str, ...] = field(default_factory=tuple)


SCENARIOS: dict[str, Scenario] = {}


def scenario(
    name: str,
    description: str,
    *,
    build_agent: Callable[[], Any] | None,
    steps: tuple[Step, ...] = (),
    size: tuple[int, int] = (110, 32),
    tags: tuple[str, ...] = (),
) -> None:
    SCENARIOS[name] = Scenario(
        name=name,
        description=description,
        build_agent=build_agent,
        steps=steps,
        size=size,
        tags=tags,
    )


def all_scenarios() -> list[Scenario]:
    return list(SCENARIOS.values())


def get(name: str) -> Scenario:
    if name not in SCENARIOS:
        raise KeyError(f"unknown scenario {name!r}; known: {', '.join(SCENARIOS)}")
    return SCENARIOS[name]


# ---- agent builders --------------------------------------------------------


def _tool_agent(output: str, final: str = "Done.") -> ScriptedV3Agent:
    return ScriptedV3Agent(
        tool_call_events(
            tool_name="run_julia",
            tool_call_id="call_demo",
            args={"code": "[1, 2, 3] .+ 1"},
            output=output,
            final_text=final,
        )
    )


def _tool_error_agent() -> ScriptedV3Agent:
    human = HumanMessage(content="run a broken cell")
    ai = AIMessage(
        content="",
        tool_calls=[{"id": "call_err", "name": "run_julia", "args": {"code": "sqrt(-1)"}}],
    )
    final = AIMessage(content="That errored. `sqrt` of a negative needs a complex input.")
    return ScriptedV3Agent(
        [
            v3_message_event(human),
            v3_message_event(ai),
            v3_tool_event(
                {
                    "event": "tool-started",
                    "tool_call_id": "call_err",
                    "tool_name": "run_julia",
                    "input": {"code": "sqrt(-1)"},
                }
            ),
            v3_tool_event(
                {
                    "event": "tool-error",
                    "tool_call_id": "call_err",
                    "message": (
                        "DomainError with -1.0:\nsqrt was called with a negative real argument."
                    ),
                }
            ),
            v3_message_event(final),
            v3_values_event([human, ai, final]),
        ]
    )


def _empty_answer_agent() -> ScriptedV3Agent:
    """The model returns an empty final message: the UI must not show a blank card."""

    def _events(stream_input):
        human = HumanMessage(content=str(stream_input))
        final = AIMessage(content="")
        return [
            v3_message_event(human),
            v3_message_event(final),
            v3_values_event([human, final]),
        ]

    return ScriptedV3Agent(_events)


async def _approve(pilot) -> None:
    """Approve the pending request: Enter confirms the default-selected "Yes"."""
    await pilot.press("enter")


_LONG_OUTPUT = "\n".join(f" {i:>3}  reservoir cell value = {i * 1.5:.3f}" for i in range(60))
_WIDE_LINE = "result vector = [" + ", ".join(str(i) for i in range(80)) + "]"
# Deliberately exercises unicode the renderer must not mangle.
_UNICODE_OUTPUT = "ρ = 1000 kg/m³ · μ = 1e-3 Pa·s · ΔP = 12.5 bar · ∇·v ≈ 0\n안녕 — Δt = 30 d"  # noqa: RUF001


# ---- the registry ----------------------------------------------------------

scenario(
    "welcome",
    "The first screen: welcome card and an empty prompt.",
    build_agent=echo_agent,
    tags=("ui",),
)
scenario(
    "answer",
    "A plain question and a prose answer.",
    build_agent=echo_agent,
    steps=("What is a multisegment well?",),
    tags=("ui",),
)
scenario(
    "tool_call",
    "A run_julia tool card with code and a short result.",
    build_agent=lambda: _tool_agent("3-element Vector{Int64}:\n 2\n 3\n 4"),
    steps=("add one to [1, 2, 3]",),
    tags=("ui",),
)
scenario(
    "streaming",
    "A streamed assistant answer (token by token).",
    build_agent=streaming_agent,
    steps=("say hello",),
    tags=("ui",),
)
scenario(
    "reasoning",
    "A reasoning trace followed by the answer.",
    build_agent=reasoning_agent,
    steps=("think then answer",),
    tags=("ui",),
)
scenario(
    "long_output",
    "A long tool result that the UI must summarise or scroll.",
    build_agent=lambda: _tool_agent(_LONG_OUTPUT),
    steps=("dump 60 lines",),
    tags=("ui", "robustness"),
)
scenario(
    "unicode_output",
    "A tool result with scientific unicode and non-Latin text.",
    build_agent=lambda: _tool_agent(_UNICODE_OUTPUT),
    steps=("show units",),
    tags=("ui", "robustness"),
)
scenario(
    "tool_error",
    "A tool that raises: the error must read cleanly, not dump a stack soup.",
    build_agent=_tool_error_agent,
    steps=("run a broken cell",),
    tags=("ui", "robustness"),
)
scenario(
    "approval",
    "A human-in-the-loop approval prompt for a shell command.",
    build_agent=interrupt_agent,
    steps=("delete the old run",),
    tags=("ui",),
)
scenario(
    "narrow",
    "A tool card in a cramped 80x24 terminal.",
    build_agent=lambda: _tool_agent("3-element Vector{Int64}:\n 2\n 3\n 4"),
    steps=("add one to [1, 2, 3]",),
    size=(80, 24),
    tags=("robustness",),
)
scenario(
    "empty_answer",
    "The model returns nothing: the UI must degrade, not render a blank card.",
    build_agent=_empty_answer_agent,
    steps=("answer with silence",),
    tags=("robustness",),
)
scenario(
    "multi_turn",
    "Two exchanges in a row, to render conversation history.",
    build_agent=echo_agent,
    steps=("first question", "second question"),
    tags=("ui", "robustness"),
)
scenario(
    "wide_line",
    "A single very long line with no breaks: it must wrap, not overflow.",
    build_agent=lambda: _tool_agent(_WIDE_LINE),
    steps=("print a wide vector",),
    tags=("robustness",),
)
scenario(
    "approval_resolved",
    "Approve a pending request and capture the resumed turn.",
    build_agent=interrupt_agent,
    steps=("delete the old run", _approve),
    tags=("robustness",),
)
