"""Compact summaries and display bodies for TUI tool cards."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

from jutul_agent.agent.tool_output import is_interrupt_payload, strip_read_file_gutter
from jutul_agent.paths import workspace_root
from jutul_agent.preview import (
    fenced_block,
    shorten,
    shorten_single_line,
    truncate_preview,
)

_COMPACT_TOOLS = frozenset(
    {
        "read_file",
        "write_file",
        "edit_file",
        "ls",
        "glob",
        "grep",
        "task",
        "record_attempt",
        "write_report",
        "plot_julia",
    }
)


def uses_compact_display(tool_name: str, *, is_error: bool, output: str = "") -> bool:
    if tool_name not in _COMPACT_TOOLS:
        return False
    if not is_error:
        return True
    return is_interrupt_payload(output)


def compact_tool_summary(
    tool_name: str,
    args: dict[str, Any],
    output: str,
    *,
    is_error: bool,
) -> str | None:
    if is_error and is_interrupt_payload(output):
        if tool_name == "task":
            agent = args.get("subagent_type") or "subagent"
            return f"Delegated → {agent} · waiting for approval"
        return "Waiting for approval"

    if tool_name == "read_file":
        return _read_file_summary(args, output)
    if tool_name == "ls":
        return _ls_summary(args, output)
    if tool_name == "glob":
        return _glob_summary(args, output)
    if tool_name == "grep":
        return _grep_summary(args, output)
    if tool_name == "write_file":
        return _write_file_summary(args)
    if tool_name == "edit_file":
        return _edit_file_summary(args)
    if tool_name == "task":
        return _task_summary(args, output, is_error=is_error)
    if tool_name == "record_attempt":
        return _record_attempt_summary(output)
    if tool_name == "write_report":
        return _render_report_summary(args, output)
    if tool_name == "plot_julia":
        return _plot_julia_summary(args, output)
    return None


def display_tool_body(
    tool_name: str,
    args: dict[str, Any],
    *,
    output: str,
    expanded: bool,
    is_error: bool,
) -> str:
    # Running state: show the Code section while julia tools are in flight so
    # the user can see what is being executed.  This check must come before the
    # compact-summary path because plot_julia is in _COMPACT_TOOLS and would
    # otherwise return "_plot_" with no code context.
    if not output:
        code_section = _julia_code_section(tool_name, args)
        if code_section:
            return _join_nonempty(code_section, "_running…_")
        # File writes show the target and a character count that grows as the
        # model streams the content, so a long write isn't silent dead air.
        if tool_name == "write_file":
            return f"_{_write_file_summary(args, running=True)}_"
        if tool_name == "edit_file":
            return f"_{_edit_file_summary(args, running=True)}_"

    summary = compact_tool_summary(tool_name, args, output, is_error=is_error)
    if (
        summary
        and not expanded
        and uses_compact_display(tool_name, is_error=is_error, output=output)
    ):
        return f"_{summary}_"

    # write_todos normally renders as a checklist in widgets._render_todo_output;
    # reaching here means its output didn't parse, so fall through to the
    # generic rendering rather than returning nothing.
    code_section = _julia_code_section(tool_name, args)
    full = strip_read_file_gutter(output) if tool_name == "read_file" else output
    language = _TOOL_LANGUAGES.get(tool_name, "")

    summary_meta = _summarize_output(full)

    if expanded:
        return _join_nonempty(
            code_section,
            f"_result · {summary_meta} · full output_",
            "",
            fenced_block(full, language=language),
        )

    expandable = is_expandable(full, tool_name=tool_name)
    if expandable:
        preview = (
            _truncate_preview_tail(full, tool_name=tool_name)
            if tool_name == "run_julia"
            else _truncate_preview(full, tool_name=tool_name)
        )
    else:
        preview = full
    rendered_preview = (
        fenced_block(preview, language=language)
        if _prefer_fenced_preview(tool_name)
        else _quote_block(preview)
    )
    if summary and uses_compact_display(tool_name, is_error=is_error, output=output):
        header = f"_{summary}_ · {summary_meta} · details_"
    else:
        header = (
            f"_output · {summary_meta}{' · preview' if expandable else ''}_"
            if code_section
            else f"_result · {summary_meta}{' · preview' if expandable else ''}_"
        )
    return _join_nonempty(code_section, header, "", rendered_preview)


def _julia_code_section(tool_name: str, args: dict[str, Any]) -> str:
    """Return a fenced ``Code`` block for run_julia / plot_julia args.

    Empty string for any other tool. The TUI uses this so the agent's code
    stays visible alongside the simulator output instead of being replaced
    by it (and so something is shown while the call is still running).
    """

    if tool_name not in {"run_julia", "plot_julia"}:
        return ""
    code = args.get("code")
    if not isinstance(code, str) or not code.strip():
        return ""
    label = "**Code**"
    return "\n".join([label, "", fenced_block(code, language="julia"), ""])


def _truncate_preview_tail(text: str, *, tool_name: str | None = None) -> str:
    """Like ``_truncate_preview`` but keeps the trailing lines.

    Julia outputs put the meaningful bit (the ``→ return value`` and the
    ``[X.YZs]`` timing) at the bottom. Head-truncation hides exactly the
    part the agent; and the human reading the TUI; care about.
    """

    line_limit, char_limit = _preview_limits(tool_name)
    lines = text.splitlines()
    if len(lines) <= line_limit and len(text) <= char_limit:
        return text
    kept: list[str] = []
    chars = 0
    for line in reversed(lines):
        new_chars = chars + len(line) + (1 if kept else 0)
        if len(kept) >= line_limit or new_chars > char_limit:
            break
        kept.append(line)
        chars = new_chars
    kept.reverse()
    return "… [output truncated above]\n" + "\n".join(kept)


def _join_nonempty(*parts: str) -> str:
    return "\n".join(part for part in parts if part)


def _read_file_summary(args: dict[str, Any], output: str) -> str:
    path = _path_arg(args)
    text = strip_read_file_gutter(output)
    lines = [line for line in text.splitlines() if line.strip()]
    return f"Read {path} · {len(lines)} lines"


def _ls_summary(args: dict[str, Any], output: str) -> str:
    root = path_link(str(args.get("path") or "."))
    paths = _parse_path_list(output)
    label = "entry" if len(paths) == 1 else "entries"
    return f"Listed {root} · {len(paths)} {label}"


def _glob_summary(args: dict[str, Any], output: str) -> str:
    pattern = str(args.get("pattern") or args.get("glob") or "*")
    paths = _parse_path_list(output)
    label = "match" if len(paths) == 1 else "matches"
    return f"Glob `{pattern}` · {len(paths)} {label}"


def _grep_summary(args: dict[str, Any], output: str) -> str:
    pattern = str(args.get("pattern") or args.get("regex") or "")
    scope = args.get("path") or args.get("glob") or args.get("include")
    lines = [line for line in output.splitlines() if line.strip()]
    label = "line" if len(lines) == 1 else "lines"
    if scope:
        return (
            f"Grep `{shorten_single_line(pattern, 40)}` in "
            f"{path_link(str(scope))} · {len(lines)} {label}"
        )
    return f"Grep `{shorten_single_line(pattern, 48)}` · {len(lines)} {label}"


def _content_len(value: Any) -> int:
    return len(value) if isinstance(value, str) else 0


def _write_file_summary(args: dict[str, Any], *, running: bool = False) -> str:
    chars = _content_len(args.get("content"))
    suffix = f" · {chars:,} chars" if chars else ""
    verb = "Writing" if running else "Wrote"
    return f"{verb} {_path_arg(args)}{suffix}"


def _edit_file_summary(args: dict[str, Any], *, running: bool = False) -> str:
    chars = _content_len(args.get("new_string"))
    suffix = f" · {chars:,} chars" if chars else ""
    verb = "Editing" if running else "Edited"
    return f"{verb} {_path_arg(args)}{suffix}"


def _task_summary(args: dict[str, Any], output: str, *, is_error: bool) -> str:
    agent = str(args.get("subagent_type") or "subagent")
    if is_error:
        return f"Delegated → {agent} · failed"
    if output.strip():
        return f"Delegated → {agent} · {shorten_single_line(output, 72)}"
    description = shorten_single_line(str(args.get("description") or ""), 72)
    return f"Delegated → {agent} · {description}"


def _record_attempt_summary(output: str) -> str:
    # ``record_attempt`` returns ``"attempt #N (parent #M) · key=val · id=<uuid>"``;
    # keep the leading "attempt #N..." part so the card body shows the index
    # and primary metric inline.
    head = output.strip().splitlines()[0] if output.strip() else ""
    if head.startswith("attempt"):
        return shorten_single_line(head, 80)
    return "Recorded attempt"


def _render_report_summary(args: dict[str, Any], output: str) -> str:
    raw = output.strip() or str(args.get("output_path") or args.get("path") or "report.html")
    path = Path(raw)
    if path.is_absolute():
        try:
            rel = path.relative_to(workspace_root()).as_posix()
        except ValueError:
            rel = path.as_posix()
        return f"Rendered {path_link(rel)}"
    return f"Rendered {path_link(raw)}"


def _plot_julia_summary(args: dict[str, Any], output: str) -> str:
    cleaned = output.strip()
    if cleaned.startswith("ERROR"):
        return "plot · failed"
    slot = args.get("slot")
    caption = args.get("caption")
    head = f"plot `{shorten(str(slot), 28)}`" if slot else "plot"
    m = re.match(r"saved plot to (\S+) \(", cleaned)
    if m:
        return f"{head} · {path_link(m.group(1))}"
    if caption:
        return f"{head} · {shorten_single_line(str(caption), 48)}"
    return head


# What `ls` and `glob` render instead of an empty list. Without this the card
# counts the sentence itself and an empty directory reads as "1 entry".
_NO_MATCHES = "No files found"


def _parse_path_list(output: str) -> list[str]:
    candidate = output.strip()
    if not candidate or candidate == _NO_MATCHES:
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(candidate)
        except (ValueError, SyntaxError, json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    return [line for line in candidate.splitlines() if line.strip()]


def _path_arg(args: dict[str, Any]) -> str:
    path_value = args.get("path") or args.get("file_path")
    return path_link(str(path_value or "file"))


_PATH_DISPLAY_BUDGET = 40


def display_path(path_str: str) -> str:
    """A short path label: the trailing segments that fit, filename first.

    The filename matters most, so long paths keep their tail and mark dropped
    parents with a leading ``…/``; a single over-long name is truncated in the
    middle so its extension survives. Keeping the label short and on one line
    matters because a wrapped link can lose its click target past the first row.
    """

    normalized = str(path_str or "file").replace("\\", "/")
    if normalized in {"", "."}:
        return "."
    if normalized == "/":
        return "workspace"
    segments = [s for s in normalized.split("/") if s]
    if not segments:
        return "/"
    tail: list[str] = []
    width = 0
    for segment in reversed(segments):
        extra = len(segment) + (1 if tail else 0)
        if tail and width + extra > _PATH_DISPLAY_BUDGET:
            break
        tail.insert(0, segment)
        width += extra
    label = "/".join(tail)
    if len(tail) < len(segments):
        label = "…/" + label
    if len(label) > _PATH_DISPLAY_BUDGET + 4:
        label = _middle_truncate(segments[-1], _PATH_DISPLAY_BUDGET)
    return label


def _middle_truncate(text: str, limit: int) -> str:
    """Shorten ``text`` to ``limit`` keeping both ends (so a file extension shows)."""
    if len(text) <= limit:
        return text
    keep = limit - 1
    head = keep // 2
    return f"{text[:head]}…{text[-(keep - head) :]}"


def path_link(path_str: str) -> str:
    """A clickable file link: short display text, full absolute path as target.

    Cards render markdown and a ``file://`` link routes to the OS opener (see
    ``on_markdown_link_clicked``), so the visible text stays short while a click
    opens the real file, so the link and the text need not match.
    """
    short = display_path(path_str)
    normalized = str(path_str or "").replace("\\", "/")
    if normalized in {"", ".", "/"}:
        return short
    try:
        path = Path(normalized)
        full = path if path.is_absolute() else workspace_root() / normalized
        return f"[{short}]({full.resolve().as_uri()})"
    except (ValueError, OSError):
        return short


_TOOL_LANGUAGES: dict[str, str] = {
    "run_julia": "julia",
    "execute": "sh",
}
_PREVIEW_LINES = 6
_PREVIEW_CHARS = 600
_TOOL_PREVIEW_LIMITS: dict[str, tuple[int, int]] = {
    "execute": (12, 2000),
    # run_julia limits are sized so a typical Jutul summary table plus the
    # return value and elapsed marker fit without a click-to-expand.
    "run_julia": (40, 6000),
    "read_file": (14, 1600),
}


def _preview_limits(tool_name: str | None) -> tuple[int, int]:
    if tool_name is None:
        return (_PREVIEW_LINES, _PREVIEW_CHARS)
    return _TOOL_PREVIEW_LIMITS.get(tool_name, (_PREVIEW_LINES, _PREVIEW_CHARS))


def _truncate_preview(text: str, *, tool_name: str | None = None) -> str:
    line_limit, char_limit = _preview_limits(tool_name)
    return truncate_preview(
        text,
        max_lines=line_limit,
        max_chars=char_limit,
        marker="\n... [output truncated]",
    )


def is_expandable(text: str, *, tool_name: str | None = None) -> bool:
    """Whether ``text`` exceeds the tool's preview budget (so expanding shows more).

    The single source of the preview limits: ``ToolBlock.expandable`` and the
    body renderer both call this, so the toggle is offered exactly when the
    expanded body actually differs from the preview.
    """
    line_limit, char_limit = _preview_limits(tool_name)
    return len(text) > char_limit or text.count("\n") + 1 > line_limit


def _summarize_output(text: str) -> str:
    line_count = max(1, text.count("\n") + 1)
    line_label = "line" if line_count == 1 else "lines"
    return f"{line_count} {line_label}"


def _quote_block(text: str) -> str:
    if not text:
        return ">"
    return "\n".join(f"> {line}" if line else ">" for line in text.splitlines())


def _prefer_fenced_preview(tool_name: str) -> bool:
    return tool_name in {"run_julia", "read_file"}
