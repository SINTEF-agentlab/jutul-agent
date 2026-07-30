"""Tests for compact TUI tool summaries."""

from __future__ import annotations

from jutul_agent.agent.tool_output import strip_read_file_gutter
from jutul_agent.interfaces.tui.tool_display import (
    compact_tool_summary,
    display_path,
    display_tool_body,
)


def test_display_path_keeps_filename_for_long_paths() -> None:
    # Short paths render whole.
    assert display_path("test_folder/alphabet.jl") == "test_folder/alphabet.jl"

    # Long paths drop leading parents (marked with …/) but keep the filename,
    # so the click target's name stays visible and the label stays on one line.
    out = display_path("testbed/jutuldarcy/deep/nested/dirs/here/advanced_simulation.jl")
    assert out.startswith("…/")
    assert out.endswith("/advanced_simulation.jl")
    assert len(out) <= 46

    # A single over-long filename is middle-truncated, keeping its extension.
    out2 = display_path("a_really_really_really_really_really_long_simulation_name.jl")
    assert out2.endswith(".jl")
    assert "…" in out2
    assert len(out2) <= 44


def test_read_file_compact_summary() -> None:
    output = "     1\t# title\n     2\tbody"
    summary = compact_tool_summary(
        "read_file",
        {"file_path": "/skills/shared/julia-and-repl/SKILL.md"},
        output,
        is_error=False,
    )
    # The path renders as a clickable file link: short display text, full target.
    assert summary.startswith("Read [")
    assert "SKILL.md](file://" in summary
    assert summary.endswith("· 2 lines")


def test_read_file_compact_body_collapsed() -> None:
    output = "     1\t# title\n     2\tbody"
    body = display_tool_body(
        "read_file",
        {"file_path": "/experiments/candidate.jl"},
        output=output,
        expanded=False,
        is_error=False,
    )
    # The summary embeds a clickable link; display text stays short, target is full.
    assert "Read [experiments/candidate.jl](file://" in body
    assert "· 2 lines" in body
    assert "Ctrl+O" not in body
    assert "content=" not in body


def test_write_file_running_body_shows_growing_char_count() -> None:
    # While the content streams in, the card shows the target and a live count
    # so a long write isn't silent. The path stays a clickable link.
    body = display_tool_body(
        "write_file",
        {"file_path": "candidate.jl", "content": "x" * 1234},
        output="",
        expanded=False,
        is_error=False,
    )
    assert "Writing [candidate.jl](file://" in body
    assert "1,234 chars" in body


def test_write_file_completed_summary_shows_char_count() -> None:
    summary = compact_tool_summary(
        "write_file",
        {"file_path": "candidate.jl", "content": "x" * 50},
        "Wrote candidate.jl",
        is_error=False,
    )
    assert summary is not None
    assert summary.startswith("Wrote [candidate.jl](file://")
    assert summary.endswith("· 50 chars")


def test_edit_file_running_body_counts_replacement_text() -> None:
    body = display_tool_body(
        "edit_file",
        {"file_path": "candidate.jl", "old_string": "a", "new_string": "y" * 7},
        output="",
        expanded=False,
        is_error=False,
    )
    assert "Editing [candidate.jl](file://" in body
    assert "7 chars" in body


def test_strip_read_file_gutter() -> None:
    from deepagents.backends.utils import format_content_with_line_numbers

    # The legacy `cat -n` gutter, still present in stored transcripts.
    assert strip_read_file_gutter("     1\talpha\n     2\tbeta") == "alpha\nbeta"
    # The gutter `read_file` emits today, built with the framework's formatter
    # so a change to its separator fails here instead of showing up on a card.
    numbered = format_content_with_line_numbers(["alpha", "beta"])
    assert strip_read_file_gutter(numbered) == "alpha\nbeta"


def test_run_julia_body_shows_code_then_output() -> None:
    body = display_tool_body(
        "run_julia",
        {"code": "1 + 1"},
        output="2",
        expanded=False,
        is_error=False,
    )
    # Both the code the agent ran and the simulator output are visible; the
    # output isn't allowed to replace the code, and Jutul's iteration tables
    # would survive untouched if they were present.
    assert "**Code**" in body
    assert "1 + 1" in body
    assert "2" in body
    code_index = body.find("**Code**")
    output_index = body.find("output")
    assert code_index >= 0 and output_index > code_index


def test_run_julia_running_state_shows_code() -> None:
    body = display_tool_body(
        "run_julia",
        {"code": "simulate(model)"},
        output="",
        expanded=False,
        is_error=False,
    )
    assert "**Code**" in body
    assert "simulate(model)" in body
    assert "running" in body.lower()


def test_run_julia_tail_truncation_keeps_return_value() -> None:
    lines = [f"progress line {i}" for i in range(60)]
    lines.extend(["→ 42", "[1.23s]"])
    output = "\n".join(lines)
    body = display_tool_body(
        "run_julia",
        {"code": "run()"},
        output=output,
        expanded=False,
        is_error=False,
    )
    assert "output truncated above" in body
    assert "→ 42" in body
    assert "[1.23s]" in body
    assert "progress line 0" not in body


def test_run_julia_preview_fits_jutul_timing_summary() -> None:
    """The simulation summary tables that follow a Jutul ``simulate(...)``
    call are ~30 lines. The collapsed preview must show all of them along
    with the trailing progress bar, return value, and elapsed marker so
    the user gets the real result without having to expand the card.
    """

    output = "\n".join(
        [
            "╭───────────────┬─────────┬───────────────┬─────────╮",
            "│ Iteration type │ Avg/step │ Avg/ministep │   Total │",
            "│                │ 18 steps │   21 ministeps │ (wasted)│",
            "├───────────────┼─────────┼───────────────┼─────────┤",
            "│ Newton         │  1.22222 │      1.04762 │ 22 (0)  │",
            "│ Linearization  │  2.38889 │      2.04762 │ 43 (0)  │",
            "│ Linear solver  │  2.66667 │      2.28571 │ 48 (0)  │",
            "│ Precond apply  │  5.33333 │      4.57143 │ 96 (0)  │",
            "╰───────────────┴─────────┴───────────────┴─────────╯",
            "",
            "╭──────────────┬─────────┬───────────┬────────╮",
            "│ Timing type  │    Each │  Relative │  Total │",
            "│              │      ms │         % │    s   │",
            "├──────────────┼─────────┼───────────┼────────┤",
            "│ Properties   │  0.3348 │   2.37 %  │ 0.1135 │",
            "│ Equations    │  2.8068 │  28.42 %  │ 1.3613 │",
            "│ Assembly     │  0.8855 │   8.97 %  │ 0.4295 │",
            "│ Linear solve │  0.4221 │   2.99 %  │ 0.1431 │",
            "│ Linear setup │  0.0000 │   0.00 %  │ 0.0000 │",
            "│ Precond apply│  0.0000 │   0.00 %  │ 0.0000 │",
            "│ Update       │  0.3636 │   2.57 %  │ 0.1233 │",
            "│ Convergence  │  2.7461 │  27.81 %  │ 1.3319 │",
            "│ Input/Output │  0.3804 │   1.16 %  │ 0.0555 │",
            "│ Other        │  3.6336 │  25.72 %  │ 1.2318 │",
            "│ Total        │ 14.1293 │ 100.00 %  │ 4.7898 │",
            "╰──────────────┴─────────┴───────────┴────────╯",
            "",
            "[stderr]",
            "Progress 100%|████████████████████████| Time: 0:00:05",
            "→ (145, (2.41708, 4.15396), (0.150268, 2.54521), 7048.44)",
            "[30.96s]",
        ]
    )
    body = display_tool_body(
        "run_julia",
        {"code": 'include("candidate.jl"); run_candidate()'},
        output=output,
        expanded=False,
        is_error=False,
    )
    assert "Iteration type" in body
    assert "Timing type" in body
    assert "Total" in body
    assert "Progress 100%" in body
    assert "→ (145" in body
    assert "[30.96s]" in body


def test_task_delegate_summary() -> None:
    summary = compact_tool_summary(
        "task",
        {
            "subagent_type": "report",
            "description": "Write the calibration narrative markdown.",
        },
        "",
        is_error=False,
    )
    assert summary.startswith("Delegated → report ·")


def test_display_tool_body_handles_unparseable_todos() -> None:
    """write_todos output that isn't a todo list still renders as text, not None."""
    body = display_tool_body(
        "write_todos",
        {},
        output="Updated the plan.",
        expanded=False,
        is_error=False,
    )
    assert isinstance(body, str)
    assert "Updated the plan." in body


def test_empty_ls_and_glob_report_no_entries() -> None:
    # `ls` and `glob` render a sentence rather than an empty list when nothing
    # matched; counting its lines would report an empty directory as "1 entry".
    from deepagents.middleware.filesystem import _format_file_paths

    empty = _format_file_paths([])
    assert compact_tool_summary("ls", {"path": "/ws"}, empty, is_error=False).endswith("0 entries")
    assert compact_tool_summary("glob", {"pattern": "*.jl"}, empty, is_error=False).endswith(
        "0 matches"
    )
