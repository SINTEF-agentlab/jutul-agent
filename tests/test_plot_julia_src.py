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

``web_render_call`` and ``web_live_call`` share one more, each a way for a figure to
reach the browser looking or behaving wrong rather than failing outright:

- ``CairoMakie.save`` registers a screen on the figure's scene and every child scene to
  render the poster, and nothing removes it -- so the live figure keeps replaying its
  plot additions into a render target that will never draw them.
"""

from __future__ import annotations

from pathlib import Path

from jutul_agent.agent.plot_julia_src import (
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
