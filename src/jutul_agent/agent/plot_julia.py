"""Julia plotting tools: capture Makie figures as session artifacts.

Plotting always runs on GLMakie. The tool opens a live window for the user only
when the session can show one (an interactive run with a display); otherwise it
renders offscreen to a PNG. Headless Linux still renders, via the xvfb-wrapped
Julia process (see the kernel). If GLMakie cannot load at all, the tool
returns a clear error rather than degrading to a backend where the native
plotters do not work.

When ``view=True`` the saved PNG is downscaled and returned to the model as a
multimodal image block, so the agent can see the plot it just made.

On the web surface, figures render in the browser with WGLMakie. They are served
live from a per-session Bonito server when it can start, so a figure's in-figure
widgets (a timestep slider, a field selector) run their Julia callbacks and update
the view; if the server can't start, a self-contained static HTML export is
embedded instead (camera control still works, the widgets do not).

This module is the orchestration: it loads the backend, runs the generated Julia
(see ``plot_julia_src``), records the artifact, and builds the model-facing reply.
"""

from __future__ import annotations

import base64
import io
import re
import socket
import sys
import uuid
from pathlib import Path
from typing import Annotated, Any

from langchain_core.tools import InjectedToolCallId, tool

from jutul_agent.agent import plot_julia_src as jl
from jutul_agent.paths import workspace_root
from jutul_agent.session import Session
from jutul_agent.simulators.base import SimulatorAdapter

_SLOT_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}$")

# Longest-edge cap (px) for the downscaled image fed back to the model. Keeps
# per-image token cost bounded across an investigation loop.
_VIEW_MAX_EDGE = 1024

_INVALID_SLOT = "ERROR: invalid slot name (use letters, digits, '.', '_', '-'; max 64 characters)."


def _resolve_slot(slot: str | None) -> tuple[str | None, str | None]:
    """Validate an optional slot, returning ``(clean slot, error)``.

    A missing slot is fine (``(None, None)``); a malformed one returns the error
    string the tool should reply with, so each tool does one check instead of two.
    """
    if not slot:
        return None, None
    slot = slot.strip()
    if not slot or not _SLOT_RE.match(slot):
        return None, _INVALID_SLOT
    return slot, None


def _truncate(text: str | None, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit] + "..."


async def _load_plot_backend(session: Session, adapter: SimulatorAdapter) -> str | None:
    """Load GLMakie and the capture helpers into the REPL once per session.

    Returns an error string, or ``None`` on success. The error is actionable when
    GLMakie can't load: the tool drives GLMakie for everything, so without it
    plotting is unavailable here.
    """

    gl = await session.julia.eval("using GLMakie")
    if gl.error:
        return (
            f"ERROR: GLMakie could not load in the {adapter.name} Julia environment, so "
            "plotting is unavailable here. On a headless Linux server install xvfb "
            "(jutul-agent auto-detects it) or a GL driver; otherwise rebuild the env with "
            f"`jutul-agent init --sim {adapter.name} --force`. "
            f"Julia said: {_truncate(gl.error, 300)}"
        )

    # The capture helpers (JutulAgent.JutulAgentPlots) ship precompiled in the
    # JutulAgent package, so this just loads it rather than eval-ing a script.
    helper = await session.julia.eval("using JutulAgent")
    if helper.error:
        return f"ERROR: failed to load JutulAgent plot helpers: {helper.error}"

    return None


async def _load_web_plot_backend(session: Session, adapter: SimulatorAdapter) -> str | None:
    """Load the web plotting backends once: WGLMakie + Bonito, plus GLMakie.

    The web surface renders figures into the browser with WebGL (WGLMakie) instead
    of a native window. GLMakie is also imported, not to render, but so the
    simulator's native plotters (whose methods live in Jutul/JutulDarcy's GLMakie
    extension, e.g. ``plot_reservoir``'s 3D mesh and well trajectories) are defined;
    those methods dispatch on backend-agnostic Makie types, so WGLMakie renders
    them to the browser. CairoMakie gives a static PNG for the record.

    GLMakie needs a GL context to load (a real display, or the Xvfb the server
    starts); if it can't load, native plotters are unavailable but inline Makie
    figures (built by the agent) still render interactively, so its failure is a
    warning, not an error.
    """

    loaded = await session.julia.eval("import CairoMakie, WGLMakie, Bonito")
    if loaded.error:
        return (
            f"ERROR: interactive web plots need WGLMakie + Bonito in the {adapter.name} "
            f"env, which did not load. Rebuild the env with "
            f"`jutul-agent init --sim {adapter.name} --force`. Julia said: "
            f"{_truncate(loaded.error, 300)}"
        )
    # Best-effort: enables native plotters when a GL context is available.
    await session.julia.eval(jl.IMPORT_GLMAKIE_OFFSCREEN)
    return None


def _free_port() -> int:
    """Pick a free localhost TCP port for the session's Bonito server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _encode_view_image(png_path: Path, max_edge: int = _VIEW_MAX_EDGE) -> str:
    """Downscale png_path to max_edge on its longest side and return base64 PNG."""
    from PIL import Image

    with Image.open(png_path) as im:
        im = im.convert("RGB")
        im.thumbnail((max_edge, max_edge))  # only shrinks; preserves aspect
        buf = io.BytesIO()
        im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _reply(summary: str, png_path: Path, view: bool) -> str | list[dict[str, Any]]:
    """The model-facing reply: the summary alone, or summary plus the downscaled
    image when ``view`` asked to see it. Vision is best-effort and never fails the
    plot, so an image that can't be encoded degrades to a noted text reply.
    """

    if not view:
        return summary
    try:
        b64 = _encode_view_image(png_path)
    except Exception as exc:
        return f"{summary}; (could not attach image for viewing: {exc})"
    return [
        {"type": "text", "text": summary},
        {"type": "image", "mime_type": "image/png", "base64": b64},
    ]


def _finalize(
    session: Session,
    *,
    abs_path: Path,
    rel_path: str,
    caption: str,
    tool_call_id: str,
    size: list[int] | None,
    dpi: int | None,
    slot: str | None,
    source_code: str,
    view: bool,
    lead: str,
    extra_parts: list[str],
) -> str | list[dict[str, Any]]:
    """Record the PNG artifact and build the reply (text, or text plus image when view).

    Shared by plot_julia and recapture_plot. The PNG artifact is always recorded
    for the transcript and report; the live Makie window or the TUI's open-artifact
    action is how the user actually sees it.
    """

    session.trace.append(
        "artifact",
        {
            "path": rel_path,
            "mime": "image/png",
            "caption": caption or slot or rel_path.rsplit("/", 1)[-1],
            "tool_call_id": tool_call_id,
            "format": "png",
            "size_px": list(size) if size is not None else None,
            "dpi": dpi,
            "slot": slot,
            "source_code": source_code,
        },
    )
    try:
        shown = abs_path.relative_to(workspace_root()).as_posix()
    except ValueError:
        shown = abs_path.as_posix()
    summary = "; ".join(p for p in [f"{lead} {shown}", *extra_parts] if p)
    return _reply(summary, abs_path, view)


def _finalize_web(
    session: Session,
    *,
    png_abs: Path,
    png_rel: str,
    html_rel: str,
    caption: str,
    tool_call_id: str,
    slot: str | None,
    source_code: str,
    view: bool,
    live_url: str | None = None,
) -> str | list[dict[str, Any]]:
    """Record the interactive plot artifact and build the reply.

    The artifact becomes the browser ``viz``: when ``live_url`` is set the figure
    is served live from the session's Bonito server (its widgets work) and the
    durable record is the PNG; otherwise the self-contained HTML export is the
    record and what's embedded. The PNG is also the poster/thumbnail and ``view``.
    """

    # A CairoMakie PNG is saved when the figure renders under Cairo (2D, and most
    # 3D scenes); it is the poster/thumbnail and the durable record. A GL-only
    # scene may yield none, and the interactive view still carries the figure.
    has_poster = png_abs.exists()
    # The durable record: a live plot's PNG poster when Cairo produced one, else the
    # static HTML export (a non-live plot always exports one; the live path exports a
    # WebGL fallback only when Cairo can't render the scene). Recording the PNG when
    # none was written would leave a dead path that 404s on resume.
    if live_url and has_poster:
        rec_path, mime, fmt = png_rel, "image/png", "png"
    else:
        rec_path, mime, fmt = html_rel, "text/html", "html"
    session.trace.append(
        "artifact",
        {
            "path": rec_path,
            "mime": mime,
            "caption": caption or slot or rec_path.rsplit("/", 1)[-1],
            "tool_call_id": tool_call_id,
            "format": fmt,
            "kind": "plot",
            "poster": png_rel if has_poster else None,
            "slot": slot,
            "live_url": live_url,
            "source_code": source_code,
        },
    )
    summary = "served a live interactive plot" if live_url else "rendered an interactive plot"
    summary += f" ({rec_path})"
    if slot:
        summary += f"; slot={slot}"
    return _reply(summary, png_abs, view and has_poster)


def make_plot_julia_tool(session: Session, *, surface: str = "tui"):
    artifacts_dir = session.output_dir / "artifacts"
    adapter = session.simulator
    web = surface == "web"
    backend_loaded = False  # one-shot memo: the backend loads once per session
    live_base: str | None = None  # the session's Bonito base URL once it's serving
    warned_no_live = False  # so a persistent server failure warns once, not per plot

    async def ensure_ready() -> str | None:
        """Load the backend on first use; return an error string if it can't load.

        On the web surface this also starts the session's Bonito server, so plots
        are served live (their in-figure widgets work). Starting the server is
        retried on each plot until it succeeds, so a transient failure (e.g. the
        picked port was grabbed in the race before Bonito bound it) doesn't disable
        live serving for the rest of the session; until it succeeds, plots render
        via the static export (a warning, not an error)."""
        nonlocal backend_loaded, live_base, warned_no_live
        if not backend_loaded:
            loader = _load_web_plot_backend if web else _load_plot_backend
            err = await loader(session, adapter)
            if err is not None:
                return err  # don't latch: a fixable load failure can be retried
            backend_loaded = True
        if web and live_base is None:
            started = await session.julia.eval(
                jl.web_server_start(_free_port(), session.session_id)
            )
            # The server prints its bound port on a uniquely-tagged line; read that
            # rather than scanning for digits (a startup log line could carry others).
            match = re.search(r"__JUTUL_WEB_PORT__=(\d+)", started.output or "")
            if started.error or match is None:
                if not warned_no_live:
                    warned_no_live = True
                    reason = started.error or "the server did not report a port"
                    print(
                        f"warning: live plot serving unavailable ({_truncate(reason, 200)}); "
                        "falling back to static interactive exports.",
                        file=sys.stderr,
                    )
            else:
                # A site-relative path, not the raw ``127.0.0.1:<port>`` Bonito
                # actually listens on: the browser reaches it through this app
                # server's ``/live/...`` reverse proxy (see interfaces/server/app.py),
                # which is the one port whatever forwarded this session's connection
                # (an SSH tunnel, VS Code/Cursor remote, Docker) already knows about.
                # ``session.web_plot_port`` is what tells that proxy where to send it.
                live_base = f"/live/{session.session_id}"
                session.web_plot_port = int(match.group(1))
        return None

    @tool
    async def plot_julia(
        code: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
        caption: str = "",
        size: list[int] | None = None,
        dpi: int | None = None,
        slot: str | None = None,
        view: bool = False,
        window: bool = True,
    ) -> str | list[dict[str, Any]]:
        """Run Julia plotting code and turn the figure into something the user can see.

        `plot_julia` is the bridge between a figure drawn in the REPL and a shareable
        result: it saves the figure as a PNG artifact (recorded in the transcript and
        report) and, in an interactive session, opens a live Makie window. Build
        figures only here, never in `run_julia` (that draws a figure nobody can see).

        Prefer your simulator's documented native plotters (the `plotting-basics` and
        per-simulator skills name them); otherwise build a `Figure` inline. Just run
        the code: you don't need to return a `Figure` or avoid `display`, since the
        tool captures whatever figure your code produced. Plotting runs on GLMakie
        like normal Julia.

        Give related plots a stable `slot`: the same `slot` refreshes one window
        in place (good for iterating), distinct slots get distinct windows, and
        `recapture_plot(slot=...)` / `close_plots(slot=...)` address that window.

        Args:
            code: Julia plotting code (a native plotter call or inline figure).
            caption: Optional caption shown in the transcript.
            size: Optional `(width, height)` in pixels.
            dpi: Optional DPI for the PNG.
            slot: Stable name (`artifacts/<slot>.png`) and window key; reuse it to
                refresh the same plot/window.
            view: Also return the downscaled image so you can see it, to verify a
                fit or diagnose. Not needed for every plot.
            window: Open a live window for the user (default true). Set false to
                compute/inspect a plot without opening a window for them.

        Returns:
            A confirmation string, or, when `view`, a text+image content list.
        """
        err = await ensure_ready()
        if err is not None:
            return err

        safe_slot, slot_err = _resolve_slot(slot)
        if slot_err:
            return slot_err

        if safe_slot:
            rel_path = f"artifacts/{safe_slot}.png"
            plot_id = safe_slot
        else:
            plot_id = uuid.uuid4().hex[:12]
            rel_path = f"artifacts/plot-{plot_id}.png"

        abs_path = session.output_dir / rel_path
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        if web:
            html_rel = rel_path[:-4] + ".html"
            html_abs = session.output_dir / html_rel
            # Serve live (in-figure widgets work) when the session's Bonito server
            # is up; otherwise fall back to a self-contained static export.
            if live_base:
                route = f"/viz/{plot_id}"
                call = jl.web_live_call(
                    user_code=code, png_path=abs_path, html_path=html_abs, route=route
                )
                live_url = f"{live_base}{route}"
            else:
                call = jl.web_render_call(user_code=code, png_path=abs_path, html_path=html_abs)
                live_url = None
            result = await session.julia.eval(call)
            if result.error:
                return f"ERROR: {result.error}"
            return _finalize_web(
                session,
                png_abs=abs_path,
                png_rel=rel_path,
                html_rel=html_rel,
                live_url=live_url,
                caption=caption,
                tool_call_id=tool_call_id,
                slot=safe_slot,
                source_code=code,
                view=view,
            )

        open_window = window and session.open_windows
        result = await session.julia.eval(
            jl.render_call(
                user_code=code,
                abs_path=abs_path,
                size=size,
                dpi=dpi,
                open_window=open_window,
                window_key=safe_slot or plot_id,
            )
        )
        if result.error:
            return f"ERROR: {result.error}"

        extra: list[str] = []
        if safe_slot:
            extra.append(f"slot={safe_slot}")
        if size is not None:
            extra.append(f"size={size[0]}x{size[1]}")
        if open_window:
            extra.append("opened window")
        return _finalize(
            session,
            abs_path=abs_path,
            rel_path=rel_path,
            caption=caption,
            tool_call_id=tool_call_id,
            size=size,
            dpi=dpi,
            slot=safe_slot,
            source_code=code,
            view=view,
            lead="saved plot to",
            extra_parts=extra,
        )

    return plot_julia


def make_recapture_tool(session: Session):
    artifacts_dir = session.output_dir / "artifacts"

    @tool
    async def recapture_plot(
        tool_call_id: Annotated[str, InjectedToolCallId],
        caption: str = "",
        view: bool = True,
        slot: str | None = None,
        size: list[int] | None = None,
    ) -> str | list[dict[str, Any]]:
        """Snapshot an open plot window at its CURRENT view and show it to you.

        Use this when the user has rotated/zoomed/stepped a live window and asks
        what it looks like now. It re-renders that window's figure at its current
        camera/timestep and (by default) returns the downscaled image so you can
        describe the new view.

        `slot` selects **which** window: the slot you gave that plot in
        `plot_julia`. Omit it for the most recently opened/refreshed window. You
        can't drive the window (advance its timestep yourself); you only snapshot
        what the user currently has. Errors if there's no such open window.

        Args:
            caption: Optional caption shown in the transcript.
            view: Return the image so you can see it (default true; that's the point).
            slot: Which window to recapture (its slot); omit for the most recent.
            size: Optional `(width, height)` in pixels.

        Returns:
            A confirmation string, or, when `view`, a text+image content list.
        """
        safe_slot, slot_err = _resolve_slot(slot)
        if slot_err:
            return slot_err
        rel_path = f"artifacts/recapture-{uuid.uuid4().hex[:12]}.png"
        abs_path = session.output_dir / rel_path
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        result = await session.julia.eval(
            jl.recapture_call(key=safe_slot or "", png_path=abs_path, size=size)
        )
        if result.error:
            return f"ERROR: {result.error}"

        extra: list[str] = []
        if safe_slot:
            extra.append(f"window={safe_slot}")
        if size is not None:
            extra.append(f"size={size[0]}x{size[1]}")
        return _finalize(
            session,
            abs_path=abs_path,
            rel_path=rel_path,
            caption=caption,
            tool_call_id=tool_call_id,
            size=size,
            dpi=None,
            slot=None,
            source_code=f"recapture_plot(slot={slot!r})" if slot else "recapture_plot()",
            view=view,
            lead="recaptured view to",
            extra_parts=extra,
        )

    return recapture_plot


def make_close_plots_tool(session: Session):
    @tool
    async def close_plots(slot: str | None = None) -> str:
        """Close interactive plot windows.

        Pass a `slot` to close that one window; omit to close all of them. Use it
        when the user asks to close/clear plot windows, or to tidy up.

        Args:
            slot: The window to close (its slot); omit to close all.

        Returns:
            A short confirmation.
        """
        safe_slot, slot_err = _resolve_slot(slot)
        if slot_err:
            return slot_err
        result = await session.julia.eval(jl.close_windows_call(safe_slot or ""))
        if result.error:
            return f"ERROR: {result.error}"
        return f"closed plot window: {safe_slot}" if safe_slot else "closed all plot windows"

    return close_plots
