"""Deep Agents tools that wrap jutul-agent core operations.

Only tools that bridge something the stock deep-agents tools cannot do
live here. Generic file/shell access is handled by deep-agents'
``read_file`` / ``write_file`` / ``edit_file`` / ``execute`` against the
workspace backend.
"""

from __future__ import annotations

import re
import uuid
from typing import Any

from langchain_core.tools import tool

from jutul_agent.juliakernel.result import OnChunk, OutputChunk
from jutul_agent.paths import resolve_in_workspace, workspace_root
from jutul_agent.session import Session
from jutul_agent.trace.schema import ARTIFACT, ATTEMPT, artifact_payload

# langgraph exposes the active tool call's output-delta writer only through this
# ContextVar. Reading it directly keeps the tool's plain ``code: str`` signature (a
# ``ToolRuntime`` parameter would break standalone tool calls in tests). Guarded so
# a langgraph internals change just disables streaming.
try:  # pragma: no cover - import guard
    from langgraph.pregel._tools import _tool_call_writer
except Exception:  # pragma: no cover - langgraph internals moved
    _tool_call_writer = None  # type: ignore[assignment]


def _capture_delta_writer() -> OnChunk | None:
    """Return an ``on_chunk`` that streams kernel output as tool-output deltas.

    The writer is captured here, in the tool's context where langgraph set the
    ContextVar, so it stays valid when the kernel's pump task (a different task)
    invokes it. Returns ``None`` outside a streaming graph (e.g. unit tests).
    """

    if _tool_call_writer is None:
        return None
    writer = _tool_call_writer.get()
    if writer is None:
        return None

    def on_chunk(chunk: OutputChunk) -> None:
        if chunk.text:
            # Raw fragment (carriage returns / ANSI intact); the UI renders it.
            writer(chunk.text)

    return on_chunk


# Julia can't reload a module already loaded this session, so a new version only
# takes effect in a fresh process. This is the one case where a restart is unavoidable.
_STALE_LOAD_RE = re.compile(r"restart julia to access the new version", re.IGNORECASE)
_STALE_LOAD_HINT = (
    "\n\n[harness] The new version won't load until Julia restarts (`reset_julia`), "
    "which clears all REPL state, so do it only when you need the new version."
)

# A just-added package failing to precompile is usually the env holding it at an
# old version; re-resolving the whole env normally lifts that.
_PRECOMPILE_FAIL_RE = re.compile(r"failed to precompile", re.IGNORECASE)
_PRECOMPILE_FAIL_HINT = (
    "\n\n[harness] If you just added the package, run `Pkg.update()` to re-resolve "
    "the env to newer compatible versions and retry. Treat it as a real "
    "incompatibility only if it still fails after that."
)

# `PkgId(...) not found` means the session's loaded modules no longer match the
# env, typically after a mid-session package change.
_MODULE_DESYNC_RE = re.compile(r"PkgId\(.*?\) not found", re.IGNORECASE)
_MODULE_DESYNC_HINT = (
    "\n\n[harness] The session's loaded modules are out of sync with the "
    "environment, usually after a mid-session install/update. `reset_julia` gives "
    "a fresh session that loads cleanly from the manifest. Reset clears all REPL "
    "state, so use it deliberately."
)


def make_run_julia_tool(session: Session):
    @tool
    async def run_julia(code: str) -> str:
        """Evaluate Julia code in a persistent REPL session.

        State (variables, functions, loaded packages) persists across calls.
        Compilation cost is paid once per package per session.

        Args:
            code: Julia code to evaluate. Can be multi-line.

        Returns:
            REPL-style output: printed output plus the return value, or an
            error description if the evaluation failed.
        """
        try:
            result = await session.julia.eval(code, on_chunk=_capture_delta_writer())
        except Exception as exc:
            # A transport-level failure (not a Julia error): the session died,
            # e.g. the process was killed or crashed. Surface it as a recoverable
            # error rather than a raw traceback, and point at the way out. The
            # kernel explains what it can about the death; keep that intact, since
            # a crash and an out-of-memory kill call for different next steps.
            return (
                f"ERROR: the Julia session is unavailable: {exc or type(exc).__name__}\n"
                "Call `reset_julia` to start a fresh session. A session that dies "
                "repeatedly on the same step is running out of memory or hitting a "
                "crash, so retrying it unchanged will not help."
            )
        if result.error:
            # Keep anything the code printed before it threw, then the error.
            text = f"ERROR: {result.error}"
            if result.output.strip():
                text = f"{result.output}\n{text}"
        else:
            text = result.output
        # No model-facing truncation here: an oversized result is offloaded
        # recoverably by the context-editing middleware (the agent can re-read
        # it selectively), which beats destroying the middle of the output.
        # The kernel's own buffers bound what can arrive (see juliakernel).
        if _STALE_LOAD_RE.search(text):
            text += _STALE_LOAD_HINT
        if _PRECOMPILE_FAIL_RE.search(text):
            text += _PRECOMPILE_FAIL_HINT
        if _MODULE_DESYNC_RE.search(text):
            text += _MODULE_DESYNC_HINT
        return text

    return run_julia


def make_reset_julia_tool(session: Session):
    @tool
    async def reset_julia() -> str:
        """Restart Julia with a fresh, empty session.

        A recovery tool, not a routine one. It clears ALL state (loaded packages,
        variables, and anything you built like models or simulation results) and the
        next run pays compilation again. Use it deliberately, mainly when a module
        must be reloaded (Julia can't swap a module already loaded this session,
        e.g. after installing or updating one), or to recover when the session has
        died (a killed or crashed process). Prefer installing packages before
        building up expensive state.

        Returns:
            Confirmation that the session was restarted, or an error description.
        """
        # Try the cheap cooperative reset first. If the session is unresponsive or
        # already gone (e.g. the process was killed), fall back to a full restart
        # of the subprocess, which doesn't depend on the old session at all.
        try:
            reset_ok = not (await session.julia.reset()).error
        except Exception:
            reset_ok = False
        if not reset_ok:
            try:
                await session.julia.restart()
            except Exception as exc:
                return f"ERROR: failed to restart Julia: {exc}"
        return (
            "Julia restarted with a fresh session. All previous state "
            "(loaded packages, variables, results) was cleared."
        )

    return reset_julia


def make_record_attempt_tool(session: Session):
    # Session-scoped attempt counter; the tool returns 1-based indices so the
    # agent can refer to "attempt #N" in user-visible output. We track ids ↦
    # index in a dict so the parent lookup is O(1) per call.
    indices: dict[str, int] = {}

    @tool
    async def record_attempt(
        rationale: str,
        metrics: dict[str, float] | None = None,
        parameters_changed: dict[str, Any] | None = None,
        parent_attempt_id: str | None = None,
        candidate_path: str | None = None,
        plot_artifact_path: str | None = None,
        notes: str | None = None,
    ) -> str:
        """Log one step of an iterative investigation to the session trace.

        Call this once **per tried configuration**, not once at the end with
        everything bundled in. The baseline is itself an attempt (the
        "root"); each hypothesis you actually evaluate is its own call with
        the previous one as ``parent_attempt_id``.

        Pair every call with ``plot_julia`` and pass the resulting
        ``artifacts/<slot>.<format>`` path as ``plot_artifact_path`` so the
        report can embed one figure per attempt. See the
        ``investigation-loop`` skill.

        Args:
            rationale: Short reason for this step.
            metrics: Named scalar metrics (optional).
            parameters_changed: Mapping of param paths to new values or
                ``(old, new)`` pairs (optional).
            parent_attempt_id: Id returned by an earlier call. Omit for
                the root (baseline) attempt.
            candidate_path: Workspace path to the file being edited (optional).
            plot_artifact_path: Workspace-relative path of the comparison
                plot created with ``plot_julia`` for this attempt, usually
                ``artifacts/<slot>.png``.
            notes: Free-form short label (optional).

        Returns ``"attempt #N (parent #M) · key=value · id=<uuid>"``. The last
        token is the id to pass back as ``parent_attempt_id`` next time.
        """
        attempt_id = str(uuid.uuid4())
        index = len(indices) + 1
        indices[attempt_id] = index
        parent_index = indices.get(parent_attempt_id) if parent_attempt_id else None

        session.trace.append(
            ATTEMPT,
            {
                "id": attempt_id,
                "parent_id": parent_attempt_id,
                "rationale": rationale,
                "parameters_changed": dict(parameters_changed or {}),
                "metrics": dict(metrics or {}),
                "candidate_path": candidate_path,
                "plot_artifact_path": plot_artifact_path,
                "notes": notes,
            },
        )

        metric_str = (
            ", ".join(f"{k}={_fmt_metric(v)}" for k, v in (metrics or {}).items()) or "no metrics"
        )
        parent_str = f" (parent #{parent_index})" if parent_index else ""
        return f"attempt #{index}{parent_str} · {metric_str} · id={attempt_id}"

    return record_attempt


def _fmt_metric(value: Any) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{f:.4g}"


def make_write_report_tool(session: Session, *, surface: str | None = None):
    web = (surface or session.surface) == "web"

    @tool
    async def write_report(
        narrative: str = "",
        title: str | None = None,
        blocks: list[dict] | None = None,
        custom_css: str = "",
        output_path: str | None = None,
    ) -> str:
        """Write an HTML report for the session and open it in the browser.

        Use this when the user asks for a written summary of an investigation.
        There are two ways to build the report; choose what fits the work.

        Simple (default): pass ``narrative``, the Markdown prose you write
        yourself. ``![caption](artifacts/plot.png)`` embeds a figure. Any
        attempts you logged with ``record_attempt`` are appended as a
        metric/exploration section. With nothing logged, the report is just your
        write-up.

        Composed: pass ``blocks``, an ordered list of typed building blocks you
        arrange yourself, with your own titles; nothing is auto-added. Types:
          - ``{"type": "prose", "markdown": "## Heading\\n..."}``
          - ``{"type": "figure", "path": "artifacts/x.png", "caption": "..."}``
          - ``{"type": "metrics", "title": "Best fit", "rows": {"RMSE (mV)": 15.9}}``
          - ``{"type": "table", "title": "...", "headers": [...], "rows": [[...]]}``
          - ``{"type": "exploration"}``: the attempt tree and metric chart, shown
            when you logged attempts.
          - ``{"type": "html", "html": "<div>...</div>"}``: raw HTML for a fully
            custom block; pair it with ``custom_css`` to style it.

        Args:
            narrative: Markdown prose for the simple layout (ignored if ``blocks``
                is given).
            title: Optional page title, shown as the page heading (defaults to
                the simulator name). Your first block need not repeat it.
            blocks: Optional ordered building blocks to compose the report body.
            custom_css: Optional extra CSS appended to the report's stylesheet,
                for restyling or styling your own ``html`` blocks.
            output_path: Where to write the HTML. Defaults to the session output
                directory (``jutul-agent-output/sessions/<id>/report.html``).
        """
        from jutul_agent.open_file import open_path
        from jutul_agent.transcript.report import render_report

        note = ""
        # On the web surface the report is pinned into the app's canvas, so it
        # must live under the session's artifacts dir (where the server serves
        # it from) rather than open in a desktop browser.
        if web:
            out = session.output_dir / "artifacts" / "report.html"
            if output_path:
                note = " (shown in the app; the report lives in the session artifacts)"
        else:
            out = resolve_in_workspace(output_path) if output_path else None
            if out is None:
                out = session.output_dir / "report.html"
                if output_path:
                    note = (
                        f" (requested path {output_path!r} is outside the workspace; "
                        "wrote to the session output directory instead)"
                    )
        out.parent.mkdir(parents=True, exist_ok=True)

        ws = workspace_root()
        artifact_dirs = [
            ws,
            ws / "artifacts",
            session.output_dir / "artifacts",
            session.state_dir / "artifacts",
        ]

        render_report(
            session.trace.iter_events(),
            out,
            narrative_markdown=narrative,
            title=title,
            session_id=session.session_id,
            simulator=session.simulator.display_name,
            artifact_dirs=artifact_dirs,
            blocks=blocks,
            custom_css=custom_css,
        )
        session.note_report(out)
        if web:
            # Surface it to the app as a report view (pinned to the canvas) rather
            # than opening a desktop browser window the web user would never see.
            rel = out.relative_to(session.output_dir).as_posix()
            session.trace.append(
                ARTIFACT,
                artifact_payload(
                    path=rel,
                    mime="text/html",
                    caption=title or f"{session.simulator.display_name} report",
                    format="html",
                    kind="report",
                    slot="report",
                ),
            )
            return f"wrote the report and showed it in the app ({rel})" + note
        open_path(out)
        return str(out) + note

    return write_report
