"""Julia source the plotting tools send to the REPL.

These are pure string builders: given Python values (paths, sizes, the user's
code) they return the Julia snippet ``plot_julia.py`` evaluates. Keeping the
code generation here, apart from the tool orchestration (tracing, the artifact
record, the model-facing reply), means every piece of Julia the plotting tools
run lives in one place and the Python side reads as plain control flow.

Two render paths share one figure-resolution preamble:

- the native **GLMakie** path (``render_call``) used by the terminal, which
  captures the figure to a PNG and optionally opens a live window, and
- the **web** path (``web_render_call`` / ``web_live_call``) used by the browser
  UI, which renders with WGLMakie and either exports self-contained HTML or
  routes the live figure on the session's Bonito server.
"""

from __future__ import annotations

from pathlib import Path

# Import GLMakie offscreen so the simulator's native plotters (their methods live
# in GLMakie's Makie extension) are defined, without ever popping a desktop window.
# Best-effort: with no GL context GLMakie can't load, so native plotters are
# unavailable, but inline WGLMakie figures still render.
IMPORT_GLMAKIE_OFFSCREEN = (
    "try; @eval import GLMakie; GLMakie.activate!(visible = false); catch; end"
)


def _size_tuple(size: list[int] | None) -> str:
    if size is None:
        return "nothing"
    return f"({int(size[0])}, {int(size[1])})"


def _optional_int(value: int | None) -> str:
    if value is None:
        return "nothing"
    return str(int(value))


def render_call(
    *,
    user_code: str,
    abs_path: Path,
    size: list[int] | None,
    dpi: int | None,
    open_window: bool,
    window_key: str,
) -> str:
    """Julia to activate GLMakie, evaluate the user code, and capture the figure.

    One begin/end block so the backend is active before the plotter runs (native
    plotters dispatch on the active backend) and the figure is captured whether the
    code returns it or opens a window. A window is keyed by window_key (the plot's
    slot) so it can be refreshed, recaptured, or closed later.
    """

    visible = "true" if open_window else "false"
    return (
        "begin\n"
        f"    GLMakie.activate!(visible = {visible})\n"
        "    local _jap_prev = JutulAgent.JutulAgentPlots._current_fig()\n"
        "    local _jap_value = begin\n"
        f"{user_code}\n"
        "    end\n"
        "    JutulAgent.JutulAgentPlots.capture(_jap_value;\n"
        f'        path = raw"{abs_path.as_posix()}",\n'
        f"        size = {_size_tuple(size)},\n"
        f"        dpi = {_optional_int(dpi)},\n"
        f"        open_window = {visible},\n"
        f'        window_key = raw"{window_key}",\n'
        "        prev_figure = _jap_prev,\n"
        "    )\n"
        "end"
    )


def _web_figure_block(user_code: str) -> str:
    """Shared Julia preamble for the web render paths: activate an offscreen
    backend, run the user code, and resolve its result to a Makie ``_fig`` (error
    if none).

    An *offscreen* backend is active while the user code runs, because a native
    plotter may call ``display(fig)`` internally and with WGLMakie active that pops
    a browser tab (the figure then also lands in the canvas a moment later, via the
    route below). GLMakie offscreen — loaded for its native-plotter methods anyway —
    absorbs the display the way the terminal path does; CairoMakie is the fallback
    when there is no GL context. WGLMakie is activated by the caller, after the
    figure is built, only to route/export it. The Figure is backend-agnostic, so it
    still renders as an interactive WebGL view once routed. Both the static-export
    and live-serve builders start from this, so the figure-resolution logic lives in
    one place.

    Absorbing the display costs a real GL screen, so the screens are closed once the
    figure is in hand: an invisible one is still a live context whose render loop
    keeps running for the rest of the session, against the same GPU driver the next
    solve shares. The figure is unaffected, being data rather than the window that
    was showing it.
    """

    return (
        "    import CairoMakie, WGLMakie, Bonito\n"
        # Load native-plotter GLMakie methods, offscreen so no desktop window pops.
        f"    {IMPORT_GLMAKIE_OFFSCREEN}\n"
        # Keep an offscreen backend active for the user code so an internal display()
        # cannot open a browser tab (GLMakie offscreen if it loaded, else CairoMakie).
        "    try; GLMakie.activate!(visible = false); catch; CairoMakie.activate!(); end\n"
        "    local _M = WGLMakie.Makie\n"
        "    local _val = begin\n"
        f"{user_code}\n"
        "    end\n"
        "    local _fig = if _val isa _M.Figure\n"
        "        _val\n"
        "    elseif _val isa _M.FigureAxisPlot\n"
        "        _val.figure\n"
        "    elseif _val isa Tuple && length(_val) >= 1 && _val[1] isa _M.Figure\n"
        "        _val[1]\n"
        "    else\n"
        "        _M.current_figure()\n"
        "    end\n"
        "    _fig === nothing && error(\n"
        '        "plot_julia: the code did not produce a Makie figure. Return a Figure, "  *\n'
        '        "or call a plotter that builds one."\n'
        "    )\n"
        # Release the GL screen the display absorption opened; the web surface never
        # shows it, and its render loop would outlive the plot.
        "    try; GLMakie.closeall(); catch; end\n"
    )


def _cairo_poster_block(png_path: Path, *, restore_wgl: bool) -> str:
    """Julia to save the figure's CairoMakie PNG poster best-effort (2D and most 3D
    scenes; a GL-only scene yields none). When ``restore_wgl`` the live path puts
    WGLMakie back as the active backend so later client connections render with it.
    """

    restore = (
        "    finally\n        WGLMakie.activate!(resize_to = :parent)\n" if restore_wgl else ""
    )
    return (
        "    try\n"
        "        CairoMakie.activate!()\n"
        f'        CairoMakie.save(raw"{png_path.as_posix()}", _fig)\n'
        "    catch\n"
        f"{restore}"
        "    end\n"
    )


def web_render_call(*, user_code: str, png_path: Path, html_path: Path) -> str:
    """Julia to evaluate the user code and export the figure for the browser.

    Bonito exports the resolved figure to a self-contained, responsive HTML file
    the web UI embeds (the static fallback when no live server is running).
    """

    return (
        "begin\n"
        + _web_figure_block(user_code)
        + "    WGLMakie.activate!(resize_to = :parent)\n"
        + f'    Bonito.export_static(raw"{html_path.as_posix()}",\n'
        '        Bonito.App(() -> Bonito.DOM.div(_fig; style = "width:100%; height:100%;")))\n'
        + _cairo_poster_block(png_path, restore_wgl=False)
        + '    "ok"\n'
        "end"
    )


def web_server_start(port: int, session_id: str) -> str:
    """Julia to start the session's Bonito server once (idempotent), returning the
    actual port it is bound to.

    The server lives in the Julia process for the session's lifetime and holds
    the live figures, so their in-figure widgets (a timestep slider, a field
    selector) run their Julia callbacks over the WebSocket and update the view —
    interactivity a static export cannot provide.

    It is created once and reused: if it already exists (e.g. the plot tool was
    rebuilt mid-session by a model switch, which resets the Python-side memo), the
    existing server stands and we return *its* port, not the freshly-requested one.
    Returning the real port is what keeps the advertised live URL pointing at the
    server the figures are actually routed on — a mismatch here is a dead "refused
    to connect" embed.

    ``proxy_url`` tells Bonito to write every URL it hands the browser (asset
    links and the widget websocket) as ``/live/<session_id>/...`` instead of an
    absolute ``127.0.0.1:<port>``, so they resolve through the app server's own
    ``/live/...`` reverse proxy (see ``interfaces/server/app.py``) rather than a
    raw port the browser may have no route to (an SSH/VS Code/Docker port
    forward that only knows about the app's own port). This is the same
    site-relative-prefix mechanism Bonito already uses for JupyterHub/Binder.
    """

    # ``global`` (not ``Main.X = ``) so the assignment defines the Main global even
    # under Julia 1.12's stricter check, which rejects assigning to a qualified
    # global that doesn't exist yet. ``__JUTUL_WEB_PORT__`` reads back
    # ``__JUTUL_WEB_SERVER__.port`` rather than echoing the requested port: if the
    # requested port lost the free/rebind race (grabbed by someone else between
    # Python releasing it and Julia binding), Bonito's `start` silently retries on
    # port+1, port+2, ... and updates `.port` to whatever it actually bound —
    # echoing the request instead would advertise a dead port nothing listens on.
    return (
        "begin\n"
        "    import WGLMakie, Bonito\n"
        "    if !isdefined(Main, :__JUTUL_WEB_SERVER__)\n"
        f'        global __JUTUL_WEB_SERVER__ = Bonito.Server("127.0.0.1", {int(port)};\n'
        f'            proxy_url = raw"/live/{session_id}/")\n'
        "        global __JUTUL_WEB_FIGS__ = Dict{String,Any}()\n"
        "        global __JUTUL_WEB_PORT__ = __JUTUL_WEB_SERVER__.port\n"
        "    end\n"
        # Print the bound port on a uniquely-tagged line so the Python side reads it
        # back unambiguously — taking "the last run of digits" from the output would
        # pick up a wrong number if Bonito/HTTP.jl logged its address (also digits)
        # on startup.
        '    println("__JUTUL_WEB_PORT__=", __JUTUL_WEB_PORT__)\n'
        "    __JUTUL_WEB_PORT__\n"
        "end"
    )


def web_live_call(*, user_code: str, png_path: Path, html_path: Path, route: str) -> str:
    """Julia to build the figure, keep it alive, and serve it on the live route.

    WGLMakie is active while the user code runs, so native plotters build WebGL
    scenes. The figure is stored (keeping its Observables alive) and routed on the
    session's Bonito server, so the browser gets a *live* view whose in-figure
    widgets run their Julia callbacks. A CairoMakie PNG is saved best-effort as the
    poster/record/``view``; if Cairo cannot render the scene (a GL-only figure),
    a self-contained WebGL HTML is exported instead, so the figure still has a
    durable record that resumes to a viewable plot rather than a dead PNG. WGLMakie
    is restored afterwards so client connections render with it.
    """

    return (
        "begin\n"
        + _web_figure_block(user_code)
        + "    WGLMakie.activate!(resize_to = :parent)\n"
        + f'    Main.__JUTUL_WEB_FIGS__[raw"{route}"] = _fig\n'
        f'    Bonito.route!(Main.__JUTUL_WEB_SERVER__, raw"{route}" => Bonito.App(() ->\n'
        f'        Bonito.DOM.div(Main.__JUTUL_WEB_FIGS__[raw"{route}"];\n'
        '            style = "width:100%; height:100%;")))\n'
        + _cairo_poster_or_export(png_path, html_path)
        + '    "ok"\n'
        "end"
    )


def _cairo_poster_or_export(png_path: Path, html_path: Path) -> str:
    """Julia for the live path's durable record: save the CairoMakie PNG poster, and
    if Cairo can't render the scene (GL-only), fall back to exporting a self-contained
    WebGL HTML. Either way WGLMakie is restored as the active backend so later client
    connections to the live route render with it.
    """

    return (
        "    try\n"
        "        CairoMakie.activate!()\n"
        f'        CairoMakie.save(raw"{png_path.as_posix()}", _fig)\n'
        "    catch\n"
        "        try\n"
        "            WGLMakie.activate!(resize_to = :parent)\n"
        f'            Bonito.export_static(raw"{html_path.as_posix()}", Bonito.App(() ->\n'
        '                Bonito.DOM.div(_fig; style = "width:100%; height:100%;")))\n'
        "        catch\n"
        "        end\n"
        "    finally\n"
        "        WGLMakie.activate!(resize_to = :parent)\n"
        "    end\n"
    )


def recapture_call(*, key: str, png_path: Path, size: list[int] | None) -> str:
    """Julia to re-render a stored window's figure at its current view to a PNG.

    Re-activates GLMakie offscreen first; the try-guard lets a session with no open
    window report cleanly rather than throwing.
    """

    return (
        "begin\n"
        "    try; GLMakie.activate!(visible = false); catch; end\n"
        "    JutulAgent.JutulAgentPlots.recapture(;\n"
        f'        key = raw"{key}",\n'
        f'        path = raw"{png_path.as_posix()}",\n'
        f"        size = {_size_tuple(size)},\n"
        "    )\n"
        "end"
    )


def close_windows_call(key: str) -> str:
    """Julia to close one window by key, or all windows when ``key`` is empty."""
    return f'JutulAgent.JutulAgentPlots.close_windows(raw"{key}")'
