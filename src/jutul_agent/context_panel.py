"""Render context-usage figures for the ``/context`` panel and status bar.

Numbers come from the provider's ``usage_metadata`` on each model call (also
traced as ``model_usage`` events), so they are measured, not estimated: the
last call's input tokens are exactly what the conversation costs to send —
system prompt, tools, skills, memory index, and history together.
"""

from __future__ import annotations

from typing import Any

_BAR_CELLS = 24
_WARN_FRACTION = 0.7
_HIGH_FRACTION = 0.9


def format_tokens(count: int) -> str:
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    if count >= 10_000:
        return f"{count / 1000:.0f}k"
    if count >= 1_000:
        return f"{count / 1000:.1f}k"
    return str(count)


def context_tokens(usage: dict[str, Any]) -> int:
    """Tokens the context holds after a model call (its input plus output)."""
    return int(usage.get("input_tokens") or 0) + int(usage.get("output_tokens") or 0)


def context_fraction(usage: dict[str, Any], window: int | None) -> float | None:
    if not window or window <= 0:
        return None
    return min(1.0, context_tokens(usage) / window)


def usage_alert(usage: dict[str, Any] | None, window: int | None) -> str:
    """``ok`` / ``warn`` / ``high`` for status-bar styling."""
    fraction = context_fraction(usage, window) if usage else None
    if fraction is None:
        return "ok"
    if fraction >= _HIGH_FRACTION:
        return "high"
    if fraction >= _WARN_FRACTION:
        return "warn"
    return "ok"


def status_label(usage: dict[str, Any] | None, window: int | None) -> str | None:
    """Short ``ctx …`` figure for the status bar, or ``None`` before any turn."""
    if not usage:
        return None
    fraction = context_fraction(usage, window)
    if fraction is None:
        return f"ctx {format_tokens(context_tokens(usage))}"
    return f"ctx {fraction:.0%}"


def _bar(fraction: float) -> str:
    filled = round(fraction * _BAR_CELLS)
    return "█" * filled + "░" * (_BAR_CELLS - filled)


def _detail(value: Any, label: str) -> str:
    return f" ({label} {format_tokens(int(value))})" if value else ""


def _pct(tokens: int, window: int | None) -> str:
    if not window:
        return ""
    return f" ({tokens / window:.1%})"


def render_context_panel(
    *,
    model_label: str | None,
    usage: dict[str, Any] | None,
    window: int | None,
    first_usage: dict[str, Any] | None = None,
    model_calls: int = 0,
    system_prompt_tokens: int | None = None,
    memory_index_tokens: int | None = None,
    memory_notes: int = 0,
    compact_trigger_tokens: int | None = None,
    clear_trigger_tokens: int | None = None,
) -> str:
    """The ``/context`` card body (markdown).

    The headline and the last-call figures are measured (``usage_metadata``).
    ``system_prompt_tokens`` / ``memory_index_tokens`` are caller-supplied
    approximations (marked ``~``) used to split the measured first-call input
    into categories; the remainder is the tool definitions, skill index, and
    framework prompt, which only the provider can count exactly.
    """
    if not usage:
        return (
            "No model calls yet — context usage appears after the first reply.\n\n"
            "The context holds the system prompt, tool definitions, the memory "
            "index, and the conversation; every model call sends all of it."
        )

    held = context_tokens(usage)
    estimated = bool(usage.get("estimated"))
    fraction = context_fraction(usage, window)
    if fraction is None:
        headline = f"**{format_tokens(held)} tokens** in context (window size unknown)"
    else:
        headline = (
            f"`{_bar(fraction)}` **{fraction:.0%}** — "
            f"{format_tokens(held)} of {format_tokens(window or 0)} tokens"
        )

    lines = [headline, ""]
    if estimated:
        # After /compact no model call has measured the new size yet; the
        # figures are an estimate until the next reply.
        lines += ["_estimated after compaction — refined on your next reply_", ""]
    baseline = int(first_usage.get("input_tokens") or 0) if first_usage else 0
    lines += _category_lines(
        held=held,
        baseline=baseline,
        window=window,
        system_prompt_tokens=system_prompt_tokens,
        memory_index_tokens=memory_index_tokens,
        memory_notes=memory_notes,
        compact_trigger_tokens=compact_trigger_tokens,
        clear_trigger_tokens=clear_trigger_tokens,
    )

    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    cache_read = (usage.get("input_token_details") or {}).get("cache_read")
    reasoning = (usage.get("output_token_details") or {}).get("reasoning")
    lines += [
        "",
        f"Last call: input {format_tokens(input_tokens)}{_detail(cache_read, 'cache-read')}"
        f" · output {format_tokens(output_tokens)}{_detail(reasoning, 'reasoning')}",
    ]
    if baseline and model_calls > 1 and not estimated:
        # Growth tracks the trajectory across model calls; after a compaction
        # the baseline predates the summary, so the comparison is meaningless
        # until the next measured turn.
        growth = held - baseline
        lines.append(
            f"Conversation growth: {'+' if growth >= 0 else '-'}{format_tokens(abs(growth))}"
            f" over {model_calls} model calls"
        )
    if model_label:
        lines.append(f"Model: `{model_label}`")
    lines.append("`/compact` summarizes older turns to free space now")
    return "\n".join(lines)


def _category_lines(
    *,
    held: int,
    baseline: int,
    window: int | None,
    system_prompt_tokens: int | None,
    memory_index_tokens: int | None,
    memory_notes: int,
    compact_trigger_tokens: int | None,
    clear_trigger_tokens: int | None = None,
) -> list[str]:
    """Estimated usage by category, Σ = the window when it is known."""
    lines = ["Estimated usage by category:"]

    if baseline and system_prompt_tokens is not None:
        lines.append(
            f"- system prompt: ~{format_tokens(system_prompt_tokens)}"
            f"{_pct(system_prompt_tokens, window)}"
        )
        memory = memory_index_tokens or 0
        if memory:
            notes = f" — {memory_notes} notes load on demand" if memory_notes else ""
            lines.append(f"- memory index: ~{format_tokens(memory)}{_pct(memory, window)}{notes}")
        framework = max(0, baseline - system_prompt_tokens - memory)
        lines.append(
            f"- tools, skills & framework: ~{format_tokens(framework)}{_pct(framework, window)}"
        )
        conversation = max(0, held - baseline)
        lines.append(f"- conversation: ~{format_tokens(conversation)}{_pct(conversation, window)}")
    elif baseline:
        lines.append(f"- fixed per call: {format_tokens(baseline)}{_pct(baseline, window)}")
        conversation = max(0, held - baseline)
        lines.append(f"- conversation: ~{format_tokens(conversation)}{_pct(conversation, window)}")
    else:
        lines.append(f"- in context: {format_tokens(held)}{_pct(held, window)}")

    trigger = compact_trigger_tokens
    if window and trigger and trigger <= window:
        free = max(0, trigger - held)
        buffer = window - trigger
        lines.append(f"- free space: {format_tokens(free)}{_pct(free, window)}")
        # Clearing fires before summarization, so note it first when it applies.
        if clear_trigger_tokens and clear_trigger_tokens < trigger:
            lines.append(
                f"- old tool results start clearing at {format_tokens(clear_trigger_tokens)}"
                " (before any summary)"
            )
        lines.append(
            f"- auto-compact buffer: {format_tokens(buffer)}{_pct(buffer, window)}"
            f" — summarization triggers at {format_tokens(trigger)}"
        )
    elif trigger:
        lines.append(f"- auto-compact triggers at {format_tokens(trigger)} tokens")
    return lines


def context_component_estimates(session, memory_dir) -> tuple[int | None, int | None]:
    """Approximate token counts for the fixed context cost the harness owns.

    Covers the base session prompt and the memory index (``None`` for a part
    that cannot be assembled). An approximation by design: capability prompt
    fragments and tool schemas ride on top, and the measured ``usage_metadata``
    figures remain the ground truth.
    """
    from langchain_core.messages.utils import count_tokens_approximately

    from jutul_agent.agent.memory import MEMORY_INDEX_FILENAME
    from jutul_agent.agent.prompts import assemble_session_prompt

    try:
        prompt = assemble_session_prompt(
            session.simulator,
            open_windows=session.open_windows,
            resumed=session.resumed,
        )
        system_tokens = int(count_tokens_approximately([prompt]))
    except Exception:
        system_tokens = None
    try:
        index = (memory_dir / MEMORY_INDEX_FILENAME).read_text(encoding="utf-8")
        memory_tokens = int(count_tokens_approximately([index]))
    except OSError:
        memory_tokens = None
    return system_tokens, memory_tokens
