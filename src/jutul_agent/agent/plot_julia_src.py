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


# ``Makie.getscreen`` hands out closed screens: it takes the *first* screen matching the
# active backend and, while it does ask ``isopen(screen)``, returns it either way. A
# figure accumulates screens. WGLMakie makes a fresh one for every Bonito session that
# displays it, and nothing takes the old ones off, so the one it picks can be a corpse.
# Picking then has no session to ask the browser through, gets an empty buffer, and the
# click falls through to the camera instead of selecting.
#
# Every legitimate re-render pushes another screen: reconnecting to a session, popping the
# view into its own window, closing a view and reopening it. Removing the accumulation
# would be the better fix, but it has to key on a session that is *definitively* closed,
# pruning on ``isopen`` also deletes screens that are merely still connecting, which was
# measured to take the live view down. So this prefers rather than prunes, and falls back
# to the first match when nothing is open.
SCREEN_PREFERENCE_MARKER = "__JUTUL_SCREEN_PREF__"

PREFER_OPEN_SCREEN_GUARD = (
    "try\n"
    # Reached through WGLMakie: the web preamble imports that, not Makie itself.
    "    @eval WGLMakie.Makie function getscreen(scene::Scene, backend = current_backend())\n"
    "        isempty(scene.current_screens) && return nothing\n"
    "        local matches = filter(scene.current_screens) do screen\n"
    "            parentmodule(typeof(screen)) === backend\n"
    "        end\n"
    "        isempty(matches) && return nothing\n"
    "        for screen in matches\n"
    "            isopen(screen) && return screen\n"
    "        end\n"
    "        return first(matches)\n"
    "    end\n"
    # Resolve the call the way a scene lookup does. If dispatch still lands in Makie's
    # own file, the signature moved and the replacement is inert.
    "    local _sm = which(WGLMakie.Makie.getscreen,\n"
    "        Tuple{WGLMakie.Makie.Scene, Module})\n"
    '    local _sinert = occursin("display.jl", String(_sm.file))\n'
    f'    println("{SCREEN_PREFERENCE_MARKER}=", _sinert ? "inert" : "ok")\n'
    "catch err\n"
    f'    println("{SCREEN_PREFERENCE_MARKER}=error: ", err)\n'
    "end"
)

# WGLMakie's ``pick`` indexes the pick buffer without checking there is anything in it:
# ``pick_native`` returns a 0x0 matrix when the screen has no usable Bonito session, and
# ``plot_matrix[1, 1]`` on that throws.
#
# A plotter that picks on click turns that throw into a dead mouse button. Jutul's 3D
# explorer picks from an ``events(fig).mousebutton`` handler at priority 2, so the throw
# lands *inside* the Observables notify and every listener behind it is skipped; on the
# web that is the whole left button: rotation, buttons, sliders, toggles and menus. The
# right button survives because that handler only looks at the left and middle ones.
#
# ``Makie.pick`` already answers ``(nothing, 0)`` when a scene has no screen at all; this
# restores that answer when it has one whose buffer is empty. Evaluated *inside* WGLMakie
# so it replaces that package's own method rather than pirating it from here.
PICK_GUARD_MARKER = "__JUTUL_PICK_GUARD__"

PICK_EMPTY_BUFFER_GUARD = (
    "try\n"
    "    @eval WGLMakie function Makie.pick(::Scene, screen::Screen, xy)\n"
    "        plot_matrix = pick_native(screen, Rect2i(xy..., 1, 1))\n"
    "        isempty(plot_matrix) && return (nothing, 0)\n"
    "        return plot_matrix[1, 1]\n"
    "    end\n"
    # Resolve the call the way a picking handler does. If dispatch still lands in
    # WGLMakie's own file, the signature moved and the replacement is inert.
    "    local _m = which(WGLMakie.Makie.pick,\n"
    "        Tuple{WGLMakie.Makie.Scene, WGLMakie.Screen, WGLMakie.Makie.Vec{2, Float64}})\n"
    '    local _inert = occursin("picking.jl", String(_m.file))\n'
    f'    println("{PICK_GUARD_MARKER}=", _inert ? "inert" : "ok")\n'
    "catch err\n"
    f'    println("{PICK_GUARD_MARKER}=error: ", err)\n'
    "end"
)


# Live web figures the session keeps at most. A safety net, not a working limit:
# the browser releases views it can no longer show long before this (its own cap
# is the WebGL context budget), so this only catches a session that plots in a
# loop without ever showing the results.
WEB_LIVE_ROUTE_CAP = 24

# Printed after the figure is built so the tool records the figure's real pixel
# size. The browser sizes the view's stage from it: the served page renders at
# exactly this size and is scaled to fit, which is what keeps a wide dashboard's
# layout intact in a narrow panel.
FIG_SIZE_MARKER = "__JUTUL_FIG_SIZE__"


# How much of its authored width a figure may lose when fitted to a narrower
# panel. The floor is the squash guard: a wide dashboard's layout was designed
# for its width, and past this much compression its controls start colliding
# (measured with the reservoir explorer: fine at ~2/3, crowded at ~2/5). When
# the floor binds, the figure stays wider than the panel and is scaled down —
# still matching the panel's aspect, so the panel is filled either way.
FIT_MIN_WIDTH_FRACTION = "2 / 3"


def _fig_size_block(size: list[int] | None, fit: list[int] | None = None) -> str:
    """Julia to apply an explicit figure size and echo the size the figure has.

    An explicit ``size`` wins. Otherwise ``fit`` — the panel's pixel size — fits
    the figure to the panel it will land in: the width becomes the panel's
    (floored at ``FIT_MIN_WIDTH_FRACTION`` of the authored width, so a wide
    layout is never crushed), and the height follows the panel's aspect but
    never drops below the authored height at the new width (the layout only
    ever gains relative room, it is never squashed). At the panel's own size
    the figure shows at scale 1 — full-size text — and when the width floor
    binds it still matches the panel's aspect, so the panel is filled without
    bands at a mild scale instead.

    The echo reports the *resulting* size (viewport widths), not the request, so
    the recorded ``size_px`` is what the browser will actually receive — even
    when the figure's own layout refused part of the resize."""

    resize = ""
    if size is not None:
        resize = f"    _M.resize!(_fig, {int(size[0])}, {int(size[1])})\n"
    elif fit is not None:
        resize = (
            "    let _vp = _fig.scene.viewport[]\n"
            f"        local _pw, _ph = {int(fit[0])}, {int(fit[1])}\n"
            "        local _w0 = _vp.widths[1]\n"
            "        local _h0 = _vp.widths[2]\n"
            f"        local _wt = max(_pw, ceil(Int, _w0 * {FIT_MIN_WIDTH_FRACTION}))\n"
            "        local _ht = max(round(Int, _wt * _ph / _pw), round(Int, _h0 * _wt / _w0))\n"
            "        (_wt != round(Int, _w0) || _ht != round(Int, _h0)) &&\n"
            "            _M.resize!(_fig, _wt, _ht)\n"
            "    end\n"
        )
    return (
        resize + "    let _vp = _fig.scene.viewport[]\n"
        f'        println("{FIG_SIZE_MARKER}=", _vp.widths[1], "x", _vp.widths[2])\n'
        "    end\n"
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
    route below). GLMakie offscreen, loaded for its native-plotter methods anyway,
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
        "    try; GLMakie.closeall(); catch; end\n" + RESTORE_OPAQUE_OVERLAY_DEPTH_WRITES
    )


# GLMakie reads ``transparency = true`` as "composite me after all opaque geometry", and
# has an order-independent pass that does it. WGLMakie has none: the attribute reaches
# three.js as nothing but ``depthWrite: false`` (``create_material`` in ThreeHelper.js),
# so the plot writes colour and no depth, and whatever renders after it paints over.
# Upstream calls this a bug, MakieOrg/Makie.jl#4673, open and untouched since 2024-12, so
# so this compensates for it rather than anticipating a fix.
# ``transparent`` is set unconditionally there, so clearing the attribute costs no alpha
# blending; it only gives the plot its depth writes back.
#
# Something does render after it: ``plot_explorer`` draws its gradient backdrop last and
# leans on already-written depth to stay behind the 3D content. JutulDarcy's
# ``plot_well!`` draws its well-name ``text!`` with ``transparency = true``, so the
# backdrop erased every label except where the mesh sat behind one and held the depth test
# off it; readable over the reservoir, gone over open background, fine in the terminal.
#
# Keyed on opacity, not on the plot being text: a plot with nothing to blend gains nothing
# from ``transparency`` under either backend, and loses the depth that keeps it on screen
# under this one. Anything not provably opaque keeps the attribute, and only the
# annotation primitives are considered; surfaces, images and volumes are where
# transparency does real work, so this doesn't get a vote there.
RESTORE_OPAQUE_OVERLAY_DEPTH_WRITES = (
    "    try\n"
    "        local _jap_scenes = Any[_fig.scene]\n"
    "        while !isempty(_jap_scenes)\n"
    "            local _jap_s = pop!(_jap_scenes)\n"
    "            local _jap_plots = Any[_jap_s.plots...]\n"
    # Recipes nest, so walk each plot's own children rather than the scene's top level.
    "            while !isempty(_jap_plots)\n"
    "                local _jap_p = pop!(_jap_plots)\n"
    "                if _jap_p isa _M.Text || _jap_p isa _M.Lines ||\n"
    "                        _jap_p isa _M.LineSegments || _jap_p isa _M.Scatter\n"
    "                    local _jap_opaque = false\n"
    # Fail closed: an attribute set we cannot read leaves the plot exactly as the
    # plotter asked for it.
    "                    try\n"
    "                        local _jap_c = _jap_p.color[]\n"
    # One colour, many colours, or per-element values whose alpha lives in the
    # colormap rather than in the values themselves.
    "                        local _jap_cols = _jap_c isa AbstractArray{<:Real} ?\n"
    "                            _M.to_colormap(_jap_p.colormap[]) :\n"
    "                            _jap_c isa AbstractArray ? _jap_c : (_jap_c,)\n"
    "                        _jap_opaque = _jap_p.alpha[] >= 1 &&\n"
    "                            all(x -> _M.RGBAf(_M.to_color(x)).alpha >= 1, _jap_cols)\n"
    "                    catch\n"
    "                        _jap_opaque = false\n"
    "                    end\n"
    # Best-effort per plot: one plot whose attribute will not take must not cost the
    # rest of the figure its overlays.
    "                    if _jap_opaque\n"
    "                        try; _jap_p.transparency[] = false; catch; end\n"
    "                    end\n"
    "                end\n"
    "                append!(_jap_plots, _jap_p.plots)\n"
    "            end\n"
    "            append!(_jap_scenes, _jap_s.children)\n"
    "        end\n"
    "    catch\n"
    "    end\n"
)


# Detach the screen ``CairoMakie.save`` registers on the figure while it renders the
# poster. Saving displays the figure to a Cairo screen, ``push_screen!`` puts it on the
# scene and recursively on every child, and nothing takes it off again; nothing closes a
# screen that only ever wrote a file. Only Cairo needs this: closing a GLMakie screen
# takes it off the scene tree as well, so ``closeall`` after the GL poster leaves nothing
# behind.
#
# It is not inert once left there: ``push!(scene, plot)`` inserts into *every* screen in
# ``current_screens``, so the live figure's later plot additions, as the explorer adds and
# deletes a mesh per cell click, are replayed into a screen that will never draw them.
#
# Census first, then drop: the filter mutates the very lists the walk reads. Best-effort
# throughout, and iterative, because a self-referential local function inside the
# generated block is fragile.
DETACH_CAIRO_SCREENS = (
    "    try\n"
    # Across the whole tree: push_screen! recurses into children, so filtering the
    # root's list alone leaves every sub-scene still holding it.
    "        local _jap_seen = Any[]\n"
    "        local _jap_stack = Any[_fig.scene]\n"
    "        while !isempty(_jap_stack)\n"
    "            local _jap_s = pop!(_jap_stack)\n"
    "            for _jap_scr in _jap_s.current_screens\n"
    "                if _jap_scr isa CairoMakie.Screen &&\n"
    "                        !any(x -> x === _jap_scr, _jap_seen)\n"
    "                    push!(_jap_seen, _jap_scr)\n"
    "                end\n"
    "            end\n"
    "            append!(_jap_stack, _jap_s.children)\n"
    "        end\n"
    "        for _jap_scr in _jap_seen\n"
    "            local _jap_drop = Any[_fig.scene]\n"
    "            while !isempty(_jap_drop)\n"
    "                local _jap_s = pop!(_jap_drop)\n"
    "                filter!(x -> x !== _jap_scr, _jap_s.current_screens)\n"
    "                append!(_jap_drop, _jap_s.children)\n"
    "            end\n"
    "        end\n"
    "    catch\n"
    "    end\n"
)


def _poster_block(png_path: Path, *, html_fallback: Path | None) -> str:
    """Julia to save the figure's PNG poster best-effort, and, when ``html_fallback``
    is given, to export a self-contained WebGL page instead if Cairo could not
    render the scene. WGLMakie is restored as the active backend either way, so later
    client connections to a live route render with it.

    The poster is CairoMakie's, deliberately never GLMakie's: the first backend
    to draw a plot registers backend-shaped computations on the figure, and a
    second backend meeting them fails (MakieOrg/Makie.jl#5228; fix unreleased
    as of Makie 0.24.13). A GL poster is that first backend, and the failure
    lands in the live view — WGLMakie's render dies inside Bonito's websocket
    handler, silently, and the browser spins forever; surface plots are the
    known trigger. Cairo registers the same computations WGLMakie uses, so the
    live render survives its poster, at the price of a slower save with no
    depth test.
    """

    export = (
        "        WGLMakie.activate!(resize_to = :parent)\n"
        f'        Bonito.export_static(raw"{html_fallback.as_posix()}", Bonito.App(() ->\n'
        f'            Bonito.DOM.div(_fig; style = "{_WRAP_STYLE}")))\n'
        if html_fallback is not None
        else ""
    )
    return (
        "    try\n"
        "        CairoMakie.activate!()\n"
        f'        CairoMakie.save(raw"{png_path.as_posix()}", _fig)\n'
        "    catch\n"
        f"{export}"
        "    finally\n"
        "        WGLMakie.activate!(resize_to = :parent)\n"
        "    end\n"
    ) + DETACH_CAIRO_SCREENS


def web_render_call(
    *,
    user_code: str,
    png_path: Path,
    html_path: Path,
    size: list[int] | None = None,
    fit: list[int] | None = None,
) -> str:
    """Julia to evaluate the user code and export the figure for the browser.

    Bonito exports the resolved figure to a self-contained, responsive HTML file
    the web UI embeds (the static fallback when no live server is running).
    """

    return (
        "begin\n"
        + _web_figure_block(user_code)
        + _fig_size_block(size, fit)
        + "    WGLMakie.activate!(resize_to = :parent)\n"
        + f'    Bonito.export_static(raw"{html_path.as_posix()}",\n'
        f'        Bonito.App(() -> Bonito.DOM.div(_fig; style = "{_WRAP_STYLE}")))\n'
        + _poster_block(png_path, html_fallback=None)
        + '    "ok"\n'
        "end"
    )


# The div Bonito serves the figure in. Fixed to the viewport, not sized by
# percentages: a percentage height resolves against the body's computed height,
# which nothing sets in the served page, so ``height:100%`` collapses to the
# content height and ``resize_to = :parent`` then tracks only the width — the
# squashed-canvas bug. ``position:fixed; inset:0`` is the iframe's viewport
# regardless of the body's own layout *and* its default margin (which otherwise
# offsets the canvas by 8px, a permanent band where widget renders land when the
# canvas and figure sizes disagree — the out-of-frame-toggles bug).
_WRAP_STYLE = "position:fixed; inset:0; margin:0; padding:0; overflow:hidden;"


def web_server_start(port: int, session_id: str) -> str:
    """Julia to start the session's Bonito server once (idempotent), returning the
    actual port it is bound to.

    The server lives in the Julia process for the session's lifetime and holds
    the live figures, so their in-figure widgets (a timestep slider, a field
    selector) run their Julia callbacks over the WebSocket and update the view,
    interactivity a static export cannot provide.

    It is created once and reused: if it already exists (e.g. the plot tool was
    rebuilt mid-session by a model switch, which resets the Python-side memo), the
    existing server stands and we return *its* port, not the freshly-requested one.
    Returning the real port is what keeps the advertised live URL pointing at the
    server the figures are actually routed on; a mismatch here is a dead "refused
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
    # port+1, port+2, ... and updates `.port` to whatever it actually bound;
    # echoing the request instead would advertise a dead port nothing listens on.
    return (
        "begin\n"
        "    import WGLMakie, Bonito\n"
        "    if !isdefined(Main, :__JUTUL_WEB_SERVER__)\n"
        f'        global __JUTUL_WEB_SERVER__ = Bonito.Server("127.0.0.1", {int(port)};\n'
        f'            proxy_url = raw"/live/{session_id}/")\n'
        "        global __JUTUL_WEB_FIGS__ = Dict{String,Any}()\n"
        # Routes in least-to-most recently served order. A recapture with no slot
        # has to name a figure, and the dict above is unordered, so recency is
        # tracked here rather than recovered from it.
        "        global __JUTUL_WEB_ORDER__ = String[]\n"
        "        global __JUTUL_WEB_PORT__ = __JUTUL_WEB_SERVER__.port\n"
        "    end\n"
        # Print the bound port on a uniquely-tagged line so the Python side reads it
        # back unambiguously; taking "the last run of digits" from the output would
        # pick up a wrong number if Bonito/HTTP.jl logged its address (also digits)
        # on startup.
        '    println("__JUTUL_WEB_PORT__=", __JUTUL_WEB_PORT__)\n'
        "    __JUTUL_WEB_PORT__\n"
        "end"
    )


def web_live_call(
    *,
    user_code: str,
    png_path: Path,
    html_path: Path,
    route: str,
    size: list[int] | None = None,
    fit: list[int] | None = None,
    poster: bool = True,
) -> str:
    """Julia to build the figure, keep it alive, and serve it on the live route.

    WGLMakie is active while the user code runs, so native plotters build WebGL
    scenes. The figure is stored in the session's live-figure registry (keeping
    its Observables alive) and routed on the session's Bonito server, so the
    browser gets a *live* view whose in-figure widgets run their Julia callbacks.
    A PNG is saved best-effort as the poster/record/``view``; if no rasteriser can
    render the scene, a self-contained WebGL HTML is exported instead, so the
    figure still has a durable record that resumes to a viewable plot rather than
    a dead PNG. WGLMakie is restored afterwards so client connections render with
    it.

    ``poster=False`` skips the durable record entirely, for a popout replay:
    that serves a second, ephemeral figure of a plot whose record already
    exists (and saves the poster's CairoMakie render time).
    """

    return (
        "begin\n"
        + _web_figure_block(user_code)
        + _fig_size_block(size, fit)
        + "    WGLMakie.activate!(resize_to = :parent)\n"
        + f'    Main.__JUTUL_WEB_FIGS__[raw"{route}"] = _fig\n'
        # Move the route to the end of the recency list (a re-plot on the same slot
        # reuses its route, so drop any earlier entry rather than double-listing it).
        "    let _order = Main.__JUTUL_WEB_ORDER__\n"
        f'        filter!(!=(raw"{route}"), _order)\n'
        f'        push!(_order, raw"{route}")\n'
        # The safety-net cap: release the oldest figures beyond it so a session
        # that plots unseen in a loop cannot grow the registry without bound. The
        # browser bounds what it *shows* far earlier and releases routes as it
        # downgrades views, so under normal use this never fires.
        f"        while length(_order) > {WEB_LIVE_ROUTE_CAP}\n"
        "            local _victim = popfirst!(_order)\n"
        "            delete!(Main.__JUTUL_WEB_FIGS__, _victim)\n"
        "            try\n"
        "                Main.Bonito.delete_route!(Main.__JUTUL_WEB_SERVER__, _victim)\n"
        "            catch\n"
        "            end\n"
        "        end\n"
        "    end\n"
        f'    Bonito.route!(Main.__JUTUL_WEB_SERVER__, raw"{route}" => Bonito.App(() ->\n'
        f'        Bonito.DOM.div(Main.__JUTUL_WEB_FIGS__[raw"{route}"];\n'
        f'            style = "{_WRAP_STYLE}")))\n'
        + (_poster_block(png_path, html_fallback=html_path) if poster else "")
        + '    "ok"\n'
        "end"
    )


def recapture_call(*, key: str, png_path: Path) -> str:
    """Julia to re-render a stored window's figure at its current view to a PNG.

    Re-activates GLMakie offscreen first; the try-guard lets a session with no open
    window report cleanly rather than throwing. No size is passed: the window is
    saved at the size the user has it, since asking Makie for another one resizes
    the window itself.
    """

    return (
        "begin\n"
        "    try; GLMakie.activate!(visible = false); catch; end\n"
        "    JutulAgent.JutulAgentPlots.recapture(;\n"
        f'        key = raw"{key}",\n'
        f'        path = raw"{png_path.as_posix()}",\n'
        "    )\n"
        "end"
    )


# Printed on the one line the Python side reads back, the way the other Julia/Python
# handshakes in this file signal a result. A refusal is an ordinary situation with
# an instruction attached, so it is reported as that instruction rather than raised,
# which would trail a Julia backtrace the model has to read past.
WEB_RECAPTURE_REFUSED = "__JUTUL_RECAPTURE_REFUSED__"

# A view the browser has never painted has no pixels to send back, so it answers
# the request with nothing. That is what happens to every plot but one when several
# are drawn at once, since only the one the canvas ends up showing gets painted; a
# view that has been displayed once stays readable afterwards.
_NOT_ON_SCREEN = (
    "this plot has not been drawn in the canvas, so it has no view to snapshot. Ask "
    "the user to select its tab there once, then try again."
)


def viz_route(plot_id: str) -> str:
    """The Bonito route a live figure is served on, given its slot or plot id.

    One spelling shared by the three sites that need it (serve, recapture, close):
    they have no way to fail loudly on each other if it drifts, since a route that
    does not match simply reads as a plot that is not there.
    """

    return f"/viz/{plot_id}"


def web_recapture_call(*, slot: str, png_path: Path) -> str:
    """Julia to snapshot a live web figure's browser view at its current state.

    The browser owns what the user currently sees, because WGLMakie runs the camera
    client-side, so the snapshot has to come from there: ``Makie.colorbuffer`` on a
    WGLMakie screen asks the connected Bonito session to render the scene and send
    the canvas pixels back. The GLMakie path this replaces looks in a registry that
    only native windows populate, and so reports "no interactive window" for every
    web figure.

    A view that cannot answer throws from deep inside the backend, so the read is
    guarded and reports instead. The capture is bounded by ``timedwait`` so a
    connection that is dead but not closed cannot stall the tool for the backend's
    own much longer timeout.
    """

    # Reached only past the guards below, which is also what proves WGLMakie is
    # loaded: importing it here would spend seconds of the kernel on a session that
    # has never plotted, just to answer that there is nothing to snapshot.
    refuse = f'return println("{WEB_RECAPTURE_REFUSED}", '
    if slot:
        resolve = (
            f'        local _route = raw"{viz_route(slot)}"\n'
            "        if !haskey(_figs, _route)\n"
            f"            {refuse}\n"
            f"                \"no live plot for slot '{slot}' (it may have been closed \" *\n"
            '                "or released); re-plot it with plot_julia and try again.")\n'
            "        end\n"
        )
    else:
        resolve = (
            "        local _order = isdefined(Main, :__JUTUL_WEB_ORDER__) ?"
            " Main.__JUTUL_WEB_ORDER__ : String[]\n"
            "        if isempty(_order)\n"
            f"            {refuse}\n"
            '                "no live plots in this session to recapture; " *\n'
            '                "make one with plot_julia first.")\n'
            "        end\n"
            "        local _route = last(_order)\n"
        )
    return (
        # A function so the checks can return early; a `begin` block cannot.
        "(function ()\n"
        "        local _figs = isdefined(Main, :__JUTUL_WEB_FIGS__) ?"
        " Main.__JUTUL_WEB_FIGS__ : Dict{String,Any}()\n"
        f"{resolve}"
        "        local _fig = _figs[_route]\n"
        "        local _screen = nothing\n"
        "        for s in _fig.scene.current_screens\n"
        "            if s isa Main.WGLMakie.Screen && isopen(s)\n"
        "                _screen = s\n"
        "                break\n"
        "            end\n"
        "        end\n"
        f'        _screen === nothing && {refuse}"{_NOT_ON_SCREEN}")\n'
        "        local _snap = @async Main.WGLMakie.Makie.colorbuffer(_screen)\n"
        "        if timedwait(() -> istaskdone(_snap), 20.0) !== :ok\n"
        f"            {refuse}\n"
        '                "the browser view did not respond to the snapshot request; " *\n'
        '                "is its tab still open?")\n'
        "        end\n"
        "        local _img = try\n"
        "            fetch(_snap)\n"
        "        catch\n"
        "            nothing\n"
        "        end\n"
        f'        (_img === nothing || isempty(_img)) && {refuse}"{_NOT_ON_SCREEN}")\n'
        f'        Main.WGLMakie.PNGFiles.save(raw"{png_path.as_posix()}", _img)\n'
        '        return "ok"\n'
        "end)()"
    )


def close_windows_call(key: str) -> str:
    """Julia to close one window by key, or all windows when ``key`` is empty."""
    return f'JutulAgent.JutulAgentPlots.close_windows(raw"{key}")'


def close_web_plots_call(slot: str) -> str:
    """Julia to release live web figures: one slot's route, or all when empty.

    The web counterpart of ``close_windows_call``, which closes native windows and
    so has nothing to close on this surface. Unrouting the figure and dropping it
    from the registry is what lets the kernel reclaim what a live figure holds. A
    tab already open in the browser keeps showing its last frame until reloaded.
    """

    routes = f'[raw"{viz_route(slot)}"]' if slot else "collect(keys(Main.__JUTUL_WEB_FIGS__))"
    return (
        "begin\n"
        # Guarded rather than imported, so closing costs nothing in a session that
        # never plotted; a server that exists is also what proves Bonito is loaded.
        "    if isdefined(Main, :__JUTUL_WEB_SERVER__)\n"
        f"        for r in {routes}\n"
        "            delete!(Main.__JUTUL_WEB_FIGS__, r)\n"
        "            filter!(!=(r), Main.__JUTUL_WEB_ORDER__)\n"
        "            try\n"
        "                Main.Bonito.delete_route!(Main.__JUTUL_WEB_SERVER__, r)\n"
        "            catch\n"
        "            end\n"
        "        end\n"
        "    end\n"
        '    "ok"\n'
        "end"
    )
