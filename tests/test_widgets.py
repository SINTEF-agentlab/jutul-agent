"""Pure-function tests for tool-rendering helpers in ``interfaces.widgets``."""

from __future__ import annotations

from makker.interfaces.tui.tool_display import display_tool_body
from makker.interfaces.tui.widgets import (
    ToolBlock,
    _render_request_preview,
    _render_tool_body,
    _render_tool_output,
)


def test_write_todos_preview_is_compact() -> None:
    preview = _render_request_preview(
        "write_todos",
        {
            "todos": [
                {"content": "Inspect the TUI event path", "status": "in_progress"},
                {"content": "Reduce startup noise", "status": "pending"},
            ]
        },
    )
    output = _render_tool_output(
        "write_todos",
        "Updated todo list to [{'content': 'Inspect the TUI event path', 'status': 'in_progress'}]",
        expanded=False,
    )

    assert "plan" in preview
    assert "[~] Inspect the TUI event path" in preview
    assert '"status"' not in output
    assert "plan updated" in output


def test_read_file_renders_compact_by_default() -> None:
    body = _render_tool_output(
        "read_file",
        "     1\t# title\n     2\tmore",
        expanded=False,
        args={"file_path": "candidate.jl"},
        is_error=False,
    )
    # Path renders as a clickable link (short display text, full file:// target).
    assert "Read [candidate.jl](file://" in body
    assert "· 2 lines" in body
    assert "content=" not in body


def test_run_julia_running_body_shows_code_section() -> None:
    body = _render_tool_body(
        "run_julia",
        {"code": "1 + 1"},
        output="",
        expanded=False,
        reject_reason=None,
        is_error=False,
    )
    assert "**Code**" in body
    assert "1 + 1" in body
    assert "running" in body.lower()
    assert 'run_julia("' not in body


async def test_tool_block_append_output_then_set_result() -> None:
    block = ToolBlock("run_julia", {"code": "run()"}, tool_call_id="call-1")

    class _FakeMarkdown:
        def __init__(self) -> None:
            self.body = ""

        async def update(self, body: str) -> None:
            self.body = body

        def refresh(self, *, layout: bool = False) -> None:
            return None

    block._body_widget = _FakeMarkdown()
    await block.append_output("progress\n")
    await block.append_output("→ 42\n")
    assert block._streamed == "progress\n→ 42\n"

    await block.set_result("progress\n→ 42\n", is_error=False)
    assert block._streamed == ""
    expected = display_tool_body(
        "run_julia",
        {"code": "run()"},
        output="progress\n→ 42\n",
        expanded=False,
        is_error=False,
    )
    assert block._body_widget.body == expected


async def test_tool_block_collapses_carriage_return_progress() -> None:
    """Streamed deltas are rendered like a terminal: a progress bar that
    overwrites itself with carriage returns collapses to a single line instead
    of stacking, matching the kernel's final rendered output."""

    block = ToolBlock("run_julia", {"code": "run()"}, tool_call_id="call-2")

    class _FakeMarkdown:
        def __init__(self) -> None:
            self.body = ""

        async def update(self, body: str) -> None:
            self.body = body

        def refresh(self, *, layout: bool = False) -> None:
            return None

    block._body_widget = _FakeMarkdown()
    # Three in-place updates of one bar, arriving as separate deltas.
    await block.append_output("Progress   0%|        |\r")
    await block.append_output("Progress  50%|####    |\r")
    await block.append_output("Progress 100%|########|\n")

    assert block._output == "Progress 100%|########|"
    assert block._output.count("Progress") == 1


async def test_tool_block_coalesces_streamed_renders() -> None:
    """Mounted blocks defer the terminal render to a timer so a chatty tool
    can't re-render the card per delta; the flush renders the latest state and
    a final result cancels any pending flush."""
    from textual.app import App
    from textual.containers import VerticalScroll

    class _Host(App[None]):
        def compose(self):
            yield VerticalScroll(id="log")

    app = _Host()
    async with app.run_test() as pilot:
        log = app.query_one("#log", VerticalScroll)
        block = ToolBlock("run_julia", {"code": "run()"}, tool_call_id="call-3")
        await log.mount(block)
        await pilot.pause()

        await block.append_output("Progress   0%|        |\r")
        await block.append_output("Progress  50%|####    |\r")
        # Not rendered yet: the flush timer is pending.
        assert block._output == ""
        assert block.has_output  # streamed text still counts as output

        await block._flush_streamed()
        assert block._output.count("Progress") == 1
        assert "50%" in block._output

        await block.append_output("Progress 100%|########|\n")
        await block.set_result("Progress 100%|########|\n")
        assert "100%" in block._output
        # A late flush after the result is a no-op.
        await block._flush_streamed()
        assert "100%" in block._output


async def test_reasoning_block_streams_tail_then_collapses() -> None:
    from makker.interfaces.tui.widgets import ReasoningBlock

    block = ReasoningBlock()

    class _FakeStatic:
        def __init__(self) -> None:
            self.body = ""

        def update(self, body: str) -> None:
            self.body = body

        def refresh(self, *, layout: bool = False) -> None:
            return None

    block._body_widget = _FakeStatic()
    await block.append_content("**Heading**\nline1\nline2\nline3\nline4")
    # While thinking, only a rolling tail is rendered (stable height, O(1) cost).
    assert block._body_widget.body == "line2\nline3\nline4"

    await block.finish()
    assert block._body_widget.body == "**Heading**"  # one-line preview
    assert "thought for" in block.border_subtitle
    assert block.content_text.startswith("**Heading**")  # full text retained

    await block.set_expanded(True)
    assert "line4" in block._body_widget.body  # verbose shows everything
    await block.set_expanded(False)
    assert block._body_widget.body == "**Heading**"
