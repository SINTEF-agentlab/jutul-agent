"""Julia source generation for the plotting tools.

``web_server_start`` is the one place two connectivity bugs can hide:

- A port-mismatch: Bonito silently retries on ``port+1``, ``port+2``, ... when
  the requested port is already taken (a real risk given ``_free_port()``'s
  bind-then-release race), and updates the server object's own ``.port`` to
  wherever it actually bound. If the generated code echoed the requested port
  back instead of reading ``.port``, every plot's ``live_url`` for the rest of
  the session would point at a dead port.
- An unreachable raw port: the browser is handed Bonito's own ephemeral port
  directly unless ``proxy_url`` is set, which breaks under any port-forwarded
  setup (SSH tunnel, VS Code/Cursor remote, Docker) where only the app's own
  port is known to whatever is forwarding ports -- a persistent "connection
  refused" no client-side retry can recover from, since the port genuinely
  isn't reachable from where the browser sits.

``web_render_call`` and ``web_live_call`` share three more, each a way for a figure to
reach the browser looking or behaving wrong rather than failing outright:

- ``CairoMakie.save``, the poster's fallback rasteriser, registers a screen on the
  figure's scene and every child scene to render it, and nothing removes it -- so the
  live figure keeps replaying its plot additions into a render target that will never
  draw them.
- ``PICK_EMPTY_BUFFER_GUARD``: WGLMakie's ``pick`` throws on an empty buffer, and a
  plotter that picks on click turns that throw into a dead mouse button for every
  listener behind it.
- ``RESTORE_OPAQUE_OVERLAY_DEPTH_WRITES``: ``transparency = true`` costs a plot its depth
  writes under WGLMakie and buys it nothing else, so a plotter that paints its backdrop
  last erases every opaque overlay drawn that way.
"""

from __future__ import annotations

from pathlib import Path

from jutul_agent.agent.plot_julia_src import (
    FIG_SIZE_MARKER,
    PICK_EMPTY_BUFFER_GUARD,
    PICK_GUARD_MARKER,
    PREFER_OPEN_SCREEN_GUARD,
    RESTORE_OPAQUE_OVERLAY_DEPTH_WRITES,
    SCREEN_PREFERENCE_MARKER,
    WEB_LIVE_ROUTE_CAP,
    web_live_call,
    web_render_call,
    web_server_start,
)


def test_web_server_start_reports_the_actual_bound_port() -> None:
    code = web_server_start(12345, "2026-01-01-0000-abcd")
    assert "__JUTUL_WEB_PORT__ = __JUTUL_WEB_SERVER__.port" in code
    assert "__JUTUL_WEB_PORT__ = 12345" not in code


def test_web_server_start_sets_a_site_relative_proxy_url() -> None:
    code = web_server_start(12345, "2026-01-01-0000-abcd")
    assert 'proxy_url = raw"/live/2026-01-01-0000-abcd/"' in code


def _live_code() -> str:
    return web_live_call(
        user_code="plot_reservoir(model, states[end])",
        png_path=Path("/tmp/p.png"),
        html_path=Path("/tmp/p.html"),
        route="/viz/reservoir",
    )


def test_live_call_never_lets_glmakie_draw_the_figure() -> None:
    # The first backend to draw a figure claims it (MakieOrg/Makie.jl#5228): a
    # GLMakie poster saves fine and the live WGLMakie render then dies inside
    # the websocket handler, silently, so the browser spins forever — surface
    # plots are the trigger. The poster is Cairo's, whose computations WGLMakie
    # shares.
    code = _live_code()
    assert "GLMakie.save" not in code
    assert "CairoMakie.save" in code


def test_live_call_still_closes_the_absorption_screens() -> None:
    # The user code runs under offscreen GLMakie so an internal display() cannot
    # pop a window; the screens that absorption opens are still closed once the
    # figure is in hand — an invisible screen is a live GL context whose render
    # loop would otherwise run for the rest of the session.
    assert "GLMakie.closeall" in _live_code()


def test_live_call_detaches_the_screen_the_cairo_fallback_leaves_behind() -> None:
    code = _live_code()
    assert "CairoMakie.Screen" in code
    assert "current_screens" in code
    # It must run *after* the save, or it detaches a screen that is then re-added.
    assert code.index("CairoMakie.save") < code.index("CairoMakie.Screen")


def test_detach_walks_child_scenes_not_just_the_root() -> None:
    # Makie registers the screen on every child scene, not just the root, so filtering
    # the root's list alone would leave every sub-scene still holding it.
    code = _live_code()
    assert "children" in code


def test_static_render_call_also_detaches() -> None:
    code = web_render_call(
        user_code="lines(1:10)",
        png_path=Path("/tmp/p.png"),
        html_path=Path("/tmp/p.html"),
    )
    assert "CairoMakie.Screen" in code
    assert "current_screens" in code


def test_cairo_detach_censuses_the_whole_scene_tree() -> None:
    code = _live_code()
    detach = code[code.index("CairoMakie.Screen") :]
    assert "children" in detach[: detach.index("filter!")]


def test_pick_guard_returns_early_on_an_empty_buffer() -> None:
    # WGLMakie indexes pick_native's result unconditionally; on a 0x0 buffer that throws
    # inside the mousebutton notify, and every listener behind it is skipped -- which on
    # the web kills the whole left button (camera, buttons, sliders, menus).
    assert "isempty(plot_matrix)" in PICK_EMPTY_BUFFER_GUARD
    assert PICK_EMPTY_BUFFER_GUARD.index("isempty(plot_matrix)") < PICK_EMPTY_BUFFER_GUARD.index(
        "plot_matrix[1, 1]"
    )


def test_pick_guard_is_evaluated_inside_wglmakie() -> None:
    # Inside the owning module, so it replaces WGLMakie's own method instead of being
    # pirated from here -- and so Scene/Screen/pick_native/Rect2i resolve.
    assert "@eval WGLMakie" in PICK_EMPTY_BUFFER_GUARD
    # Best-effort: a WGLMakie whose internals differ must not fail web plotting.
    assert PICK_EMPTY_BUFFER_GUARD.lstrip().startswith("try")


def test_pick_guard_reports_whether_it_actually_applied() -> None:
    # The guard is pinned to a signature we do not own, so it can stop applying when
    # WGLMakie moves. Silence would be the worst outcome: the plot still renders and a
    # mouse button just stops working. It resolves the call and says which side won, so
    # the caller can warn. `picking.jl` appearing here is the upstream-side marker it
    # compares against, not a call into it.
    assert PICK_GUARD_MARKER in PICK_EMPTY_BUFFER_GUARD
    assert "which(" in PICK_EMPTY_BUFFER_GUARD
    assert "picking.jl" in PICK_EMPTY_BUFFER_GUARD
    # Failure to even evaluate has to be reported too, not swallowed by the try.
    assert f"{PICK_GUARD_MARKER}=error" in PICK_EMPTY_BUFFER_GUARD


def test_screen_preference_returns_an_open_screen() -> None:
    # A figure collects one screen per render and they are never removed, so a screen
    # that never finished initialising can sit ahead of the live one. Makie's own
    # getscreen takes the first backend match -- and computes isopen() only to return
    # the screen either way -- so it hands out the dead one and picking dies with it.
    assert "isopen(screen) && return screen" in PREFER_OPEN_SCREEN_GUARD


def test_screen_preference_never_removes_a_screen() -> None:
    # Preference, not pruning. A screen mid-connect reports isopen == false, so
    # deleting on that basis takes the live view down with it; falling back to the
    # first match keeps today's behaviour when nothing is open.
    assert "first(matches)" in PREFER_OPEN_SCREEN_GUARD
    for destructive in ("filter!", "delete_screen!", "deleteat!", "empty!"):
        assert destructive not in PREFER_OPEN_SCREEN_GUARD


def test_depth_writes_are_restored_on_both_web_paths() -> None:
    # The erasure is a render-order problem, so it costs the live view and the static
    # export alike -- both hand the same figure to WGLMakie.
    for code in (
        _live_code(),
        web_render_call(
            user_code="lines(1:10)", png_path=Path("/tmp/p.png"), html_path=Path("/tmp/p.html")
        ),
    ):
        assert "transparency[] = false" in code


def test_depth_writes_are_restored_before_the_figure_is_served() -> None:
    # WGLMakie reads transparency when it serializes the plot, so a pass that ran after
    # the route was registered would leave the first view -- the one the user sees --
    # still missing its overlays.
    code = _live_code()
    assert code.index("transparency[] = false") < code.index("Bonito.route!")


def test_translucency_is_what_decides_not_the_plot_being_text() -> None:
    # The mismatch is about opacity: a plot with nothing to blend gains nothing from
    # transparency under either backend, and loses its depth writes under this one. Text
    # was only where we happened to see it first -- well trajectories and markers reach
    # the same state.
    for kind in ("_M.Text", "_M.Lines", "_M.LineSegments", "_M.Scatter"):
        assert f"isa {kind}" in RESTORE_OPAQUE_OVERLAY_DEPTH_WRITES
    assert "alpha[] >= 1" in RESTORE_OPAQUE_OVERLAY_DEPTH_WRITES
    assert "_jap_opaque = false" in RESTORE_OPAQUE_OVERLAY_DEPTH_WRITES


def test_surfaces_images_and_volumes_are_never_touched() -> None:
    # Where transparency is load-bearing -- a volume's absorption rendering depends on
    # it -- the pass does not get a vote at all.
    for kind in ("_M.Mesh", "_M.Surface", "_M.Image", "_M.Heatmap", "_M.Volume", "_M.Voxels"):
        assert kind not in RESTORE_OPAQUE_OVERLAY_DEPTH_WRITES


def test_a_colour_that_cannot_be_read_keeps_its_transparency() -> None:
    # Fail closed. Guessing wrong towards "opaque" hides whatever sits behind the plot,
    # which is worse than the erasure this pass exists to undo.
    assert (
        "catch\n                        _jap_opaque = false" in RESTORE_OPAQUE_OVERLAY_DEPTH_WRITES
    )


def test_colour_mapped_overlays_are_judged_by_their_colormap() -> None:
    # Per-element values carry no alpha of their own; a colormap with a translucent stop
    # is what makes such a plot see-through.
    assert "AbstractArray{<:Real}" in RESTORE_OPAQUE_OVERLAY_DEPTH_WRITES
    assert "to_colormap" in RESTORE_OPAQUE_OVERLAY_DEPTH_WRITES


def test_depth_pass_reaches_nested_plots_and_child_scenes() -> None:
    # Overlays sit wherever the plotter put them: inside a recipe's own plot list, and in
    # the child scene an LScene keeps its 3D content in.
    assert "_jap_p.plots" in RESTORE_OPAQUE_OVERLAY_DEPTH_WRITES
    assert "_jap_s.children" in RESTORE_OPAQUE_OVERLAY_DEPTH_WRITES


def test_depth_pass_never_fails_the_plot() -> None:
    # Best-effort, per plot and overall: a Makie whose attributes differ should cost the
    # overlays their fix, not the user their figure.
    assert RESTORE_OPAQUE_OVERLAY_DEPTH_WRITES.lstrip().startswith("try")
    assert "try; _jap_p.transparency[] = false; catch; end" in RESTORE_OPAQUE_OVERLAY_DEPTH_WRITES


def test_screen_preference_reports_whether_it_actually_applied() -> None:
    # Same reason as the pick guard: it replaces a method we do not own, so it can stop
    # applying when Makie moves, and the only symptom would be clicks quietly not
    # landing. `display.jl` is the upstream-side marker it compares against.
    assert "which(" in PREFER_OPEN_SCREEN_GUARD
    assert "display.jl" in PREFER_OPEN_SCREEN_GUARD
    assert f"{SCREEN_PREFERENCE_MARKER}=error" in PREFER_OPEN_SCREEN_GUARD


def test_screen_preference_is_reached_through_wglmakie() -> None:
    # The web preamble imports WGLMakie, not Makie, so the module has to be qualified.
    assert "@eval WGLMakie.Makie" in PREFER_OPEN_SCREEN_GUARD
    assert SCREEN_PREFERENCE_MARKER in PREFER_OPEN_SCREEN_GUARD


def test_served_wrapper_is_fixed_to_the_viewport() -> None:
    # A percentage height resolves against the served page's body height, which
    # nothing sets, so it collapses to the content height and resize_to=:parent
    # then tracks only the width -- the figure keeps its own height and a wide
    # dashboard renders crushed in a narrow panel. Fixed positioning is the
    # iframe's viewport regardless of the body's layout AND its default margin,
    # whose 8px offset otherwise leaves a permanent band where widget renders
    # land whenever the canvas and figure sizes disagree.
    live = _live_code()
    static = web_render_call(
        user_code="lines(1:10)", png_path=Path("/tmp/p.png"), html_path=Path("/tmp/p.html")
    )
    for code in (live, static):
        assert "position:fixed" in code and "inset:0" in code
        assert "width:100%" not in code and "height:100%" not in code


def test_live_call_echoes_the_figure_size_and_applies_a_requested_one() -> None:
    # The echo reports the figure's *resulting* pixel size; the client sizes the
    # view's stage from it (the record is what keeps a dashboard's layout intact).
    code = _live_code()
    assert FIG_SIZE_MARKER in code
    assert "resize!" not in code  # no size requested: the figure keeps its own
    sized = web_live_call(
        user_code="lines(1:10)",
        png_path=Path("/tmp/p.png"),
        html_path=Path("/tmp/p.html"),
        route="/viz/x",
        size=[900, 600],
    )
    assert "resize!(_fig, 900, 600)" in sized
    # Applied before the echo, so what is echoed is what the browser will get.
    assert sized.index("resize!") < sized.index(FIG_SIZE_MARKER)


def test_live_call_fit_targets_the_panel_with_a_squash_floor() -> None:
    # With a panel fit (and no explicit size) the figure takes the panel's
    # shape — aspect clamped near the authored one, scaled up only by the
    # squash floors. The rule lives in the generated Julia so the figure's
    # *authored* size (known only after the code ran) is what it works from.
    code = web_live_call(
        user_code="lines(1:10)",
        png_path=Path("/tmp/p.png"),
        html_path=Path("/tmp/p.html"),
        route="/viz/x",
        fit=[990, 1230],
    )
    assert "local _pw, _ph = 990, 1230" in code
    assert "clamp(_pw / _ph, (_w0 / _h0) * 2 / 3, (_w0 / _h0) / (2 / 3))" in code
    assert "clamp(max((_w0 * 2 / 3) / _w1, (_h0 * 2 / 3) / _h1), 1.0, 3.0)" in code
    # Fitting happens before the echo, so the recorded size is the truth.
    assert code.index("resize!") < code.index(FIG_SIZE_MARKER)

    # An explicit size wins over the fit hint.
    sized = web_live_call(
        user_code="lines(1:10)",
        png_path=Path("/tmp/p.png"),
        html_path=Path("/tmp/p.html"),
        route="/viz/x",
        size=[900, 600],
        fit=[990, 1230],
    )
    assert "resize!(_fig, 900, 600)" in sized
    assert "_pw" not in sized


def test_size_block_echoes_the_authored_size_before_any_fit() -> None:
    # Both echoes ship: the authored size (before any resize — what a later
    # re-fit anchors its floors to) and the resulting size (what the browser
    # stages). The authored echo must come before the fit touches the figure.
    from jutul_agent.agent.plot_julia_src import FIG_AUTHORED_MARKER

    code = web_live_call(
        user_code="lines(1:10)",
        png_path=Path("/tmp/p.png"),
        html_path=Path("/tmp/p.html"),
        route="/viz/x",
        fit=[990, 1230],
    )
    assert FIG_AUTHORED_MARKER in code and FIG_SIZE_MARKER in code
    assert code.index(FIG_AUTHORED_MARKER) < code.index("resize!")
    assert code.index("resize!") < code.index(FIG_SIZE_MARKER)


def test_resize_call_resizes_the_routed_figure_and_echoes() -> None:
    # The refit path: an in-place resize! of a routed live figure, echoing the
    # resulting size; a route without a figure answers without touching anything.
    from jutul_agent.agent.plot_julia_src import resize_web_fig_call

    code = resize_web_fig_call("/viz/res", 1200, 1000)
    assert 'get(Main.__JUTUL_WEB_FIGS__, raw"/viz/res", nothing)' in code
    assert "resize!(_fig, 1200, 1000)" in code
    assert FIG_SIZE_MARKER in code
    assert '"missing"' in code


def test_live_call_caps_the_route_registry() -> None:
    # The safety net: beyond the cap the oldest route is unrouted and its figure
    # dropped, so a session plotting unseen in a loop cannot grow without bound.
    code = _live_code()
    assert f"length(_order) > {WEB_LIVE_ROUTE_CAP}" in code
    assert "popfirst!(_order)" in code
    assert "delete_route!" in code
    # Generous by design: a working session shows its plots and never gets here.
    assert WEB_LIVE_ROUTE_CAP >= 20


def test_live_call_without_poster_skips_the_durable_record() -> None:
    # A popout replay serves a second, ephemeral figure of a plot whose record
    # already exists; re-saving the poster would cost the CairoMakie render and
    # touch files the original owns.
    code = web_live_call(
        user_code="lines(1:10)",
        png_path=Path("/tmp/p.png"),
        html_path=Path("/tmp/p.html"),
        route="/viz/x--pop1",
        poster=False,
    )
    assert "CairoMakie.save" not in code
    assert "export_static" not in code
    assert "__JUTUL_WEB_FIGS__" in code  # still served live
