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
from jutul_agent.simulators import warmup
from jutul_agent.simulators.base import SimulatorAdapter
from jutul_agent.trace.schema import ARTIFACT, artifact_payload

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

    helper = await _load_agent_runtime(session)
    if helper is not None:
        return helper

    gl = await session.julia.eval("using GLMakie")
    if gl.error:
        return (
            f"ERROR: GLMakie could not load in the {adapter.name} Julia environment, so "
            "plotting is unavailable here. On a headless Linux server install xvfb "
            "(jutul-agent auto-detects it) or a GL driver; otherwise rebuild the env with "
            f"`jutul-agent init --sim {adapter.name} --force`. "
            f"Julia said: {_truncate(gl.error, 300)}"
        )

    return None


async def _load_agent_runtime(session: Session) -> str | None:
    """Load the capture helpers (``JutulAgent.JutulAgentPlots``), through the same
    statement the session's warm-up uses.

    They ship precompiled, so this only pays load latency, but the order the packages
    load in decides how much of that precompilation survives: a pkgimage is valid only
    for the world it was baked in, and pulling the shared package in ahead of the
    simulator's rebuilds the parts of the bake that were inferred through a method the
    simulator brings. Warm-up normally gets here first; a plot requested before it
    finishes must not establish a different order.
    """

    loaded = await session.julia.eval(warmup.load_statement(session.simulator.warm_package))
    if loaded.error:
        return f"ERROR: failed to load JutulAgent plot helpers: {loaded.error}"
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

    # The live figures are held by JutulAgent.JutulAgentPlots, so this surface needs the
    # helpers just as the native one does, and before the backends so that the order
    # the packages load in is the one they were baked in.
    helper = await _load_agent_runtime(session)
    if helper is not None:
        return helper

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
    # Both guards patch over upstream bugs, once per session and before any figure is
    # built. Each reports whether it took: when the method it replaces moves upstream it
    # stops applying silently, and the only symptom is a mouse button that no longer works
    # in the browser.
    prefer = await session.julia.eval(jl.PREFER_OPEN_SCREEN_GUARD)
    if f"{jl.SCREEN_PREFERENCE_MARKER}=ok" not in (prefer.output or ""):
        print(
            "warning: could not make screen resolution prefer an open screen; clicking "
            "a plot (selecting a cell) may not respond "
            f"({_truncate(prefer.output or prefer.error, 200)})",
            file=sys.stderr,
        )
    guard = await session.julia.eval(jl.PICK_EMPTY_BUFFER_GUARD)
    if f"{jl.PICK_GUARD_MARKER}=ok" not in (guard.output or ""):
        print(
            "warning: could not guard WGLMakie's pick against an empty buffer; a plot "
            "whose plotter picks on click may not respond to that mouse button "
            f"({_truncate(guard.output or guard.error, 200)})",
            file=sys.stderr,
        )
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
    kind: str = "plot",
) -> str | list[dict[str, Any]]:
    """Record the PNG artifact and build the reply (text, or text plus image when view).

    Shared by plot_julia and recapture_plot. The PNG artifact is always recorded
    for the transcript and report; the live Makie window or the TUI's open-artifact
    action is how the user actually sees it.

    ``kind`` routes the browser: ``"plot"`` pins a canvas view, anything else
    (a recapture's ``"snapshot"``) shows inline.
    """

    session.trace.append(
        ARTIFACT,
        artifact_payload(
            path=rel_path,
            mime="image/png",
            caption=caption or slot or rel_path.rsplit("/", 1)[-1],
            tool_call_id=tool_call_id,
            format="png",
            kind=kind,
            size_px=size,
            dpi=dpi,
            slot=slot,
            source_code=source_code,
        ),
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
    tool_call_id: str | None,
    slot: str | None,
    source_code: str,
    view: bool,
    live_url: str | None = None,
    size_px: list[int] | None = None,
) -> str | list[dict[str, Any]]:
    """Record the interactive plot artifact and build the reply.

    The artifact becomes the browser ``viz``: when ``live_url`` is set the figure
    is served live from the session's Bonito server (its widgets work) and the
    durable record is the PNG; otherwise the self-contained HTML export is the
    record and what's embedded. The PNG is also the poster/thumbnail and ``view``.
    """

    # A PNG is saved when some rasteriser can render the figure offscreen; it is the
    # poster/thumbnail and the durable record. A scene none of them can draw yields
    # none, and the interactive view still carries the figure.
    has_poster = png_abs.exists()
    # The durable record: a live plot's PNG poster when one was rendered, else the
    # static HTML export (a non-live plot always exports one; the live path exports a
    # WebGL fallback only when no rasteriser could render the scene). Recording the PNG
    # when none was written would leave a dead path that 404s on resume.
    if live_url and has_poster:
        rec_path, mime, fmt = png_rel, "image/png", "png"
    else:
        rec_path, mime, fmt = html_rel, "text/html", "html"
    session.trace.append(
        ARTIFACT,
        artifact_payload(
            path=rec_path,
            mime=mime,
            caption=caption or slot or rec_path.rsplit("/", 1)[-1],
            tool_call_id=tool_call_id,
            format=fmt,
            kind="plot",
            size_px=size_px,
            poster=png_rel if has_poster else None,
            slot=slot,
            live_url=live_url,
            source_code=source_code,
        ),
    )
    summary = "served a live interactive plot" if live_url else "rendered an interactive plot"
    summary += f" ({rec_path})"
    if slot:
        summary += f"; slot={slot}"
    return _reply(summary, png_abs, view and has_poster)


def _parse_fig_size(output: str | None) -> list[int] | None:
    """The figure's real pixel size, read back from the tagged echo line."""
    match = re.search(rf"{jl.FIG_SIZE_MARKER}=(\d+)x(\d+)", output or "")
    if match is None:
        return None
    return [int(match.group(1)), int(match.group(2))]


def _panel_aspect(session: Session) -> float | None:
    """The browser canvas panel's height/width, from the client's size hint.

    ``None`` without a plausible hint (no client connected yet, or a degenerate
    measurement), which leaves the figure's authored size untouched. Clamped so
    a freak measurement can never author an absurdly tall or flat figure.
    """
    hint = session.web_canvas_hint
    if not hint:
        return None
    w, h = hint
    if w < 100 or h < 100:
        return None
    return max(0.3, min(h / w, 3.0))


def _plot_id_of(payload: dict[str, Any]) -> str:
    """A recorded plot's route identity: its slot, else its file stem.

    The tool routes a slotted plot on its slot and an unslotted one on
    ``plot-<id>`` (the artifact file stem), so replaying through the same rule
    lands on the same route and the browser view revives in place.
    """
    slot = payload.get("slot")
    if slot:
        return str(slot)
    rec = str(payload.get("path") or "")
    return rec.rsplit("/", 1)[-1].rsplit(".", 1)[0]


async def replot_web(
    session: Session,
    payload: dict[str, Any],
    *,
    route_suffix: str = "",
    record: bool = True,
    size: list[int] | None = None,
) -> tuple[str | None, str | None, list[int] | None]:
    """Re-run a recorded plot's code through the live web path; ``(error, url, size)``.

    ``payload`` is the plot's artifact payload from the trace (the server looks
    it up there — recorded code only, never code a client sent). With ``record``
    the figure re-serves on its original route and the artifact is re-finalized,
    so the side-output flush delivers a fresh ``viz`` and the browser view
    revives in place (the regenerate button). With ``record=False`` the figure
    serves on ``route_suffix``'s separate route and only the live URL is
    returned — an independent, ephemeral view for a popout window.

    ``size`` re-fits the figure to a target the client measured — the canvas
    stage on a regenerate, the popup window on a popout — so the replay is laid
    out for the rectangle it will actually land in. The returned size is the
    figure's *echoed* real size, which can differ when its own layout refuses
    part of the resize; presentation must trust the echo, never the request.

    The code replays into the current kernel state: variables it used may have
    changed or be gone after a restart, so a failure is reported, not hidden.
    """

    code = str(payload.get("source_code") or "")
    if not code:
        return "this plot has no recorded code to re-run", None, None
    err = await _load_web_plot_backend(session, session.simulator)
    if err is not None:
        return err, None, None
    # Idempotent: an existing server answers with its real port, a fresh kernel
    # gets one started. Either way the reverse proxy learns where to dial.
    started = await session.julia.eval(jl.web_server_start(_free_port(), session.session_id))
    match = re.search(r"__JUTUL_WEB_PORT__=(\d+)", started.output or "")
    if started.error or match is None:
        reason = _truncate(started.error or "the server did not report a port", 200)
        return f"the session's live plot server is unavailable ({reason})", None, None
    session.web_plot_port = int(match.group(1))

    plot_id = _plot_id_of(payload)
    route = jl.viz_route(f"{plot_id}{route_suffix}")
    live_url = f"/live/{session.session_id}{route}"
    png_rel = f"artifacts/{plot_id}.png"
    png_abs = session.output_dir / png_rel
    html_rel = png_rel[:-4] + ".html"

    result = await session.julia.eval(
        jl.web_live_call(
            user_code=code,
            png_path=png_abs,
            html_path=session.output_dir / html_rel,
            route=route,
            size=size,
            poster=record,
        )
    )
    if result.error:
        return f"the plot code failed when re-run: {_truncate(result.error, 300)}", None, None
    size_px = _parse_fig_size(result.output)
    if record:
        _finalize_web(
            session,
            png_abs=png_abs,
            png_rel=png_rel,
            html_rel=html_rel,
            live_url=live_url,
            caption=str(payload.get("caption") or ""),
            tool_call_id=None,
            slot=payload.get("slot"),
            source_code=code,
            view=False,
            size_px=size_px,
        )
    return None, live_url, size_px


def make_plot_julia_tool(session: Session, *, surface: str | None = None):
    artifacts_dir = session.output_dir / "artifacts"
    adapter = session.simulator
    web = (surface or session.surface) == "web"
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
            plot_id = safe_slot
        else:
            # The artifact file stem, so the plot's live route and its recorded
            # path agree — a replay derives the route from the record, and only
            # the same route revives the browser view in place.
            plot_id = f"plot-{uuid.uuid4().hex[:12]}"
        rel_path = f"artifacts/{plot_id}.png"

        abs_path = session.output_dir / rel_path
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        if web:
            html_rel = rel_path[:-4] + ".html"
            html_abs = session.output_dir / html_rel
            # Without an explicit size, extend the figure's height toward the
            # browser panel's shape (the client keeps the session's hint fresh):
            # the authored width — what the layout was designed for — is kept,
            # so the panel fills vertically instead of letterboxing a wide figure.
            aspect = _panel_aspect(session) if size is None else None
            # Serve live (in-figure widgets work) when the session's Bonito server
            # is up; otherwise fall back to a self-contained static export.
            if live_base:
                route = jl.viz_route(plot_id)
                call = jl.web_live_call(
                    user_code=code,
                    png_path=abs_path,
                    html_path=html_abs,
                    route=route,
                    size=size,
                    aspect=aspect,
                )
                live_url = f"{live_base}{route}"
            else:
                call = jl.web_render_call(
                    user_code=code, png_path=abs_path, html_path=html_abs, size=size, aspect=aspect
                )
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
                size_px=_parse_fig_size(result.output),
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


def make_recapture_tool(session: Session, *, surface: str | None = None):
    artifacts_dir = session.output_dir / "artifacts"
    web = (surface or session.surface) == "web"

    @tool
    async def recapture_plot(
        tool_call_id: Annotated[str, InjectedToolCallId],
        caption: str = "",
        view: bool = True,
        slot: str | None = None,
    ) -> str | list[dict[str, Any]]:
        """Snapshot an open plot at its CURRENT view and show it to you.

        Use this when the user has rotated/zoomed/stepped a live plot (a native
        window, or a live view in the web canvas) and asks what it looks like now.
        It captures that plot at its current camera/timestep and (by default)
        returns the downscaled image so you can describe the new view.

        `slot` selects **which** plot: the slot you gave it in `plot_julia`. Omit
        it for the most recently opened/refreshed one. You can't drive the plot
        (advance its timestep yourself) or resize it; you only snapshot what the
        user currently has, at the size they have it. Errors if there's no such
        open plot.

        Args:
            caption: Optional caption shown in the transcript.
            view: Return the image so you can see it (default true; that's the point).
            slot: Which plot to recapture (its slot); omit for the most recent.

        Returns:
            A confirmation string, or, when `view`, a text+image content list.
        """
        safe_slot, slot_err = _resolve_slot(slot)
        if slot_err:
            return slot_err
        rel_path = f"artifacts/recapture-{uuid.uuid4().hex[:12]}.png"
        abs_path = session.output_dir / rel_path
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        if web:
            # The browser owns the current view, so the snapshot comes from the plot's
            # connected live session.
            call = jl.web_recapture_call(slot=safe_slot or "", png_path=abs_path)
        else:
            call = jl.recapture_call(key=safe_slot or "", png_path=abs_path)
        result = await session.julia.eval(call)
        if result.error:
            return f"ERROR: {result.error}"
        if jl.WEB_RECAPTURE_REFUSED in result.output:
            return f"ERROR: {result.output.split(jl.WEB_RECAPTURE_REFUSED, 1)[1].strip()}"

        extra: list[str] = []
        if safe_slot:
            extra.append(f"plot={safe_slot}" if web else f"window={safe_slot}")
        return _finalize(
            session,
            abs_path=abs_path,
            rel_path=rel_path,
            caption=caption,
            tool_call_id=tool_call_id,
            size=None,
            dpi=None,
            slot=None,
            source_code=f"recapture_plot(slot={slot!r})" if slot else "recapture_plot()",
            view=view,
            lead="recaptured view to",
            extra_parts=extra,
            kind="snapshot",
        )

    return recapture_plot


def make_close_plots_tool(session: Session, *, surface: str | None = None):
    web = (surface or session.surface) == "web"

    @tool
    async def close_plots(slot: str | None = None) -> str:
        """Close interactive plots (native windows, or live views in the web canvas).

        Pass a `slot` to close that one plot; omit to close all of them. Use it
        when the user asks to close/clear plots, or to tidy up a long session
        (each live plot holds its data in memory until closed or superseded).

        Args:
            slot: The plot to close (its slot); omit to close all.

        Returns:
            A short confirmation.
        """
        safe_slot, slot_err = _resolve_slot(slot)
        if slot_err:
            return slot_err
        make_call = jl.close_web_plots_call if web else jl.close_windows_call
        result = await session.julia.eval(make_call(safe_slot or ""))
        if result.error:
            return f"ERROR: {result.error}"
        return f"closed plot: {safe_slot}" if safe_slot else "closed all plots"

    return close_plots
