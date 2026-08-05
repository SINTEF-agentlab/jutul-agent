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

``web_render_call`` and ``web_live_call`` share two more, each a way for a figure to
reach the browser looking or behaving wrong rather than failing outright:

- ``CairoMakie.save`` registers a screen on the figure's scene and every child scene to
  render the poster, and nothing removes it -- so the live figure keeps replaying its
  plot additions into a render target that will never draw them.
- ``PICK_EMPTY_BUFFER_GUARD``: WGLMakie's ``pick`` throws on an empty buffer, and a
  plotter that picks on click turns that throw into a dead mouse button for every
  listener behind it.
"""

from __future__ import annotations

from pathlib import Path

from jutul_agent.agent.plot_julia_src import (
    PICK_EMPTY_BUFFER_GUARD,
    PICK_GUARD_MARKER,
    PREFER_OPEN_SCREEN_GUARD,
    SCREEN_PREFERENCE_MARKER,
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


def test_live_call_detaches_the_screen_the_poster_save_leaves_behind() -> None:
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
