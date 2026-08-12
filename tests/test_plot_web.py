"""The web surface renders plots as interactive HTML (WGLMakie + Bonito)."""

from __future__ import annotations

from pathlib import Path

from jutul_agent.agent import plot_julia_src as jl_src
from jutul_agent.agent.plot_julia import (
    make_close_plots_tool,
    make_plot_julia_tool,
    make_recapture_tool,
)
from jutul_agent.julia.session import EvalResult
from jutul_agent.lab.fakes import FakeJulia, make_fake_adapter
from jutul_agent.session import Session
from jutul_agent.simulators.warmup import load_statement
from jutul_agent.trace import TraceLog


def _session(tmp_path: Path, julia: FakeJulia) -> Session:
    return Session.create(julia=julia, state_root=tmp_path, simulator=make_fake_adapter(tmp_path))


async def _call(tool, args: dict) -> str:
    msg = await tool.ainvoke({"type": "tool_call", "name": "plot_julia", "id": "c1", "args": args})
    return str(getattr(msg, "content", msg))


async def test_web_surface_serves_plot_live(tmp_path: Path) -> None:
    # With the session's Bonito server up, a plot is served live (its in-figure
    # widgets work); the recorded artifact is the PNG and a live URL is attached.
    seen: list[str] = []

    async def fake_eval(code: str) -> EvalResult:
        seen.append(code)
        if "CairoMakie.Makie === WGLMakie.Makie" in code:
            return EvalResult(output="JUTUL_MAKIE_MATCH")
        if "Bonito.Server" in code:
            # The server prints its bound port on a tagged line; a later log line
            # carries other digits (an address) that a "last run of digits" parse
            # would wrongly pick, so this guards the tagged-line parse.
            return EvalResult(output="__JUTUL_WEB_PORT__=51000\n[ Info: listening on :9999")
        if "Bonito.route!" in code:  # the live render: Cairo saves the poster PNG
            (session.output_dir / "artifacts" / "pres.png").write_bytes(b"PNG")
        return EvalResult(output="")  # render "succeeds"

    session = _session(tmp_path, FakeJulia(eval_handler=fake_eval))
    tool = make_plot_julia_tool(session, surface="web")

    result = await _call(tool, {"code": "lines(1:3, 1:3)", "caption": "field", "slot": "pres"})

    assert "live" in result
    assert any("Bonito.Server" in c for c in seen)  # the session server started
    assert any("Bonito.route!" in c for c in seen)  # the figure was routed live
    # The live render carries a static-export fallback for a GL-only scene Cairo
    # can't render, so the figure still has a durable record (not a dead PNG).
    assert any("Bonito.export_static" in c for c in seen)

    # An offscreen backend is active while the user code runs (a native plotter may
    # call display() internally, which with WGLMakie active pops a browser tab);
    # WGLMakie is activated only after the figure is built, to route it.
    render = next(c for c in seen if "Bonito.route!" in c)
    assert "GLMakie.activate!(visible = false)" in render
    assert "CairoMakie.activate!()" in render  # the no-GL-context fallback
    assert render.index("lines(1:3, 1:3)") < render.index("WGLMakie.activate!(resize_to = :parent)")

    log = TraceLog(session.state_dir / "trace.sqlite")
    try:
        artifact = next(ev for ev in log.iter_events() if ev.kind == "artifact")
        assert artifact.payload["mime"] == "image/png"
        assert artifact.payload["path"] == "artifacts/pres.png"
        assert artifact.payload["kind"] == "plot"
        # The live URL is a site-relative path through this session's reverse proxy
        # (see interfaces/server/app.py), not the raw port Bonito bound -- the browser
        # never needs to reach that port directly (it may not be forwarded at all,
        # e.g. behind an SSH tunnel). The proxy resolves the port from session state.
        assert f"/live/{session.session_id}/viz/" in (artifact.payload["live_url"] or "")
        assert session.web_plot_port == 51000
    finally:
        log.close()


async def test_web_plot_without_size_extends_toward_the_panel_hint(tmp_path: Path) -> None:
    # With a canvas-size hint on the session and no explicit size, the figure
    # keeps its authored width and grows its height toward the panel's aspect;
    # an explicit size from the model still wins outright.
    seen: list[str] = []

    async def fake_eval(code: str) -> EvalResult:
        seen.append(code)
        if "Bonito.Server" in code:
            return EvalResult(output="__JUTUL_WEB_PORT__=51000")
        return EvalResult(output="")

    session = _session(tmp_path, FakeJulia(eval_handler=fake_eval))
    session.web_canvas_hint = (800, 920)  # aspect 1.15
    tool = make_plot_julia_tool(session, surface="web")

    await _call(tool, {"code": "lines(1:3)", "slot": "a"})
    routed = next(c for c in seen if "Bonito.route!" in c)
    assert "_vp.widths[1] * 1.1500" in routed
    assert "_t > _vp.widths[2] && _M.resize!" in routed

    seen.clear()
    await _call(tool, {"code": "lines(1:3)", "slot": "b", "size": [640, 480]})
    routed = next(c for c in seen if "Bonito.route!" in c)
    assert "resize!(_fig, 640, 480)" in routed
    assert "1.1500" not in routed


async def test_web_plot_ignores_a_degenerate_panel_hint(tmp_path: Path) -> None:
    async def fake_eval(code: str) -> EvalResult:
        if "Bonito.Server" in code:
            return EvalResult(output="__JUTUL_WEB_PORT__=51000")
        return EvalResult(output="")

    session = _session(tmp_path, FakeJulia(eval_handler=fake_eval))
    session.web_canvas_hint = (40, 8000)  # nonsense measurement
    tool = make_plot_julia_tool(session, surface="web")
    await _call(tool, {"code": "lines(1:3)", "slot": "a"})
    routed = next(c for c in session.julia.calls if "Bonito.route!" in c)
    assert "resize!" not in routed


async def test_web_surface_live_gl_only_records_html_export(tmp_path: Path) -> None:
    # A GL-only scene Cairo can't render yields no poster PNG on the live path; the
    # durable record must then be the static HTML export, not a dead PNG path that
    # would 404 on resume. The live URL is still attached (the live view worked).
    async def fake_eval(code: str) -> EvalResult:
        if "CairoMakie.Makie === WGLMakie.Makie" in code:
            return EvalResult(output="JUTUL_MAKIE_MATCH")
        if "Bonito.Server" in code:
            return EvalResult(output="__JUTUL_WEB_PORT__=51000")
        return EvalResult(output="")  # render "succeeds" but writes no poster PNG

    session = _session(tmp_path, FakeJulia(eval_handler=fake_eval))
    tool = make_plot_julia_tool(session, surface="web")

    await _call(tool, {"code": "volume(rand(4, 4, 4))", "caption": "field", "slot": "pres"})

    log = TraceLog(session.state_dir / "trace.sqlite")
    try:
        artifact = next(ev for ev in log.iter_events() if ev.kind == "artifact")
        assert artifact.payload["mime"] == "text/html"
        assert artifact.payload["path"] == "artifacts/pres.html"
        assert f"/live/{session.session_id}/viz/" in (artifact.payload["live_url"] or "")
    finally:
        log.close()


async def test_web_surface_static_fallback_when_server_down(tmp_path: Path) -> None:
    # If the Bonito server can't start, plots fall back to a self-contained static
    # HTML export (the camera still works; the in-figure widgets don't).
    seen: list[str] = []

    async def fake_eval(code: str) -> EvalResult:
        seen.append(code)
        if "CairoMakie.Makie === WGLMakie.Makie" in code:
            return EvalResult(output="JUTUL_MAKIE_MATCH")
        if "Bonito.Server" in code:
            return EvalResult(output="", error="could not bind port")
        return EvalResult(output="")

    session = _session(tmp_path, FakeJulia(eval_handler=fake_eval))
    tool = make_plot_julia_tool(session, surface="web")

    result = await _call(tool, {"code": "lines(1:3, 1:3)", "caption": "field", "slot": "pres"})

    assert ".html" in result
    assert any("Bonito.export_static" in c and "resize_to = :parent" in c for c in seen)

    log = TraceLog(session.state_dir / "trace.sqlite")
    try:
        artifact = next(ev for ev in log.iter_events() if ev.kind == "artifact")
        assert artifact.payload["mime"] == "text/html"
        assert artifact.payload["format"] == "html"
        assert artifact.payload["path"] == "artifacts/pres.html"
        assert artifact.payload["live_url"] is None
    finally:
        log.close()


async def test_web_render_releases_the_gl_screen_it_opened(tmp_path: Path) -> None:
    # A native plotter displays internally, which with an offscreen GLMakie active
    # opens a real GL screen whose render loop then runs for the rest of the
    # session, sharing a GPU driver with every solve that follows. The web surface
    # never shows that screen, so it is closed once the figure is in hand.
    seen: list[str] = []

    async def fake_eval(code: str) -> EvalResult:
        seen.append(code)
        if "CairoMakie.Makie === WGLMakie.Makie" in code:
            return EvalResult(output="JUTUL_MAKIE_MATCH")
        if "Bonito.Server" in code:
            return EvalResult(output="__JUTUL_WEB_PORT__=51000")
        return EvalResult(output="")

    session = _session(tmp_path, FakeJulia(eval_handler=fake_eval))
    tool = make_plot_julia_tool(session, surface="web")

    await _call(tool, {"code": "plot_reservoir(case)", "slot": "setup"})

    render = next(c for c in seen if "Bonito.route!" in c)
    assert "GLMakie.closeall()" in render
    # After the figure is resolved: closing first would discard what was displayed.
    assert render.index("plot_reservoir(case)") < render.index("GLMakie.closeall()")
    assert render.index("GLMakie.closeall()") < render.index("Bonito.route!")


async def test_web_surface_reports_missing_backend(tmp_path: Path) -> None:
    async def fake_eval(code: str) -> EvalResult:
        if code.strip() == "import CairoMakie, WGLMakie, Bonito":
            return EvalResult(output="", error="ArgumentError: Package WGLMakie not found")
        return EvalResult(output="")

    session = _session(tmp_path, FakeJulia(eval_handler=fake_eval))
    tool = make_plot_julia_tool(session, surface="web")

    result = await _call(tool, {"code": "lines(1:3, 1:3)"})
    assert "WGLMakie" in result and "Bonito" in result


async def test_tui_surface_still_uses_glmakie(tmp_path: Path) -> None:
    session, seen = _recording(tmp_path)
    tool = make_plot_julia_tool(session)  # default surface = tui

    await _call(tool, {"code": "lines(1:3, 1:3)"})
    assert any(c.strip() == "using GLMakie" for c in seen)
    assert not any("Bonito.export_static" in c for c in seen)


def _recording(tmp_path: Path) -> tuple[Session, list[str]]:
    """A session whose Julia records each snippet it is given and answers emptily."""

    seen: list[str] = []

    async def fake_eval(code: str) -> EvalResult:
        seen.append(code)
        return EvalResult(output="")

    return _session(tmp_path, FakeJulia(eval_handler=fake_eval)), seen


async def _recapture(tool, args: dict) -> str:
    msg = await tool.ainvoke(
        {"type": "tool_call", "name": "recapture_plot", "id": "c1", "args": args}
    )
    return str(getattr(msg, "content", msg))


async def test_recapture_on_web_snapshots_the_live_view(tmp_path: Path) -> None:
    """A web recapture must read the browser's canvas, not the native-window registry.

    WGLMakie runs the camera client-side, so the browser is the only place the
    user's current view exists. The GLMakie path consults a registry that only
    native windows populate, so on this surface it finds nothing and reports "no
    interactive window is open" for a plot the user is looking at.
    """

    session, seen = _recording(tmp_path)
    tool = make_recapture_tool(session, surface="web")

    result = await _recapture(tool, {"slot": "pres", "view": False})
    assert "recaptured view" in result

    call = next(c for c in seen if "colorbuffer" in c)
    assert "/viz/pres" in call  # the slot's live route, not a window key
    assert "current_screens" in call  # captured from the figure's connected screen
    assert "PNGFiles.save" in call
    # The GLMakie registry lookup is what produced the bogus "no window" error.
    assert "JutulAgentPlots.recapture" not in call

    log = TraceLog(session.state_dir / "trace.sqlite")
    try:
        artifact = next(ev for ev in log.iter_events() if ev.kind == "artifact")
        # A snapshot, not a "plot": the browser shows it inline rather than pinning
        # another canvas view for every recapture of a plot it already displays.
        assert artifact.payload["kind"] == "snapshot"
    finally:
        log.close()


async def test_recapture_on_web_without_a_slot_takes_the_most_recent_plot(
    tmp_path: Path,
) -> None:
    """An unslotted plot's route is a random id, so recency has to be tracked."""

    session, seen = _recording(tmp_path)
    tool = make_recapture_tool(session, surface="web")

    await _recapture(tool, {"view": False})
    call = next(c for c in seen if "colorbuffer" in c)
    assert "last(_order)" in call


async def test_live_plot_records_its_route_as_most_recent(tmp_path: Path) -> None:
    """Serving a plot must put its route last, so an unslotted recapture finds it."""

    seen: list[str] = []

    async def fake_eval(code: str) -> EvalResult:
        seen.append(code)
        if "CairoMakie.Makie === WGLMakie.Makie" in code:
            return EvalResult(output="JUTUL_MAKIE_MATCH")
        if "Bonito.Server" in code:
            return EvalResult(output="__JUTUL_WEB_PORT__=51000")
        if "Bonito.route!" in code:
            (session.output_dir / "artifacts" / "pres.png").write_bytes(b"PNG")
        return EvalResult(output="")

    adapter = make_fake_adapter(tmp_path, warm_package="JutulAgentFakeSim")
    session = Session.create(
        julia=FakeJulia(eval_handler=fake_eval), state_root=tmp_path, simulator=adapter
    )
    tool = make_plot_julia_tool(session, surface="web")

    await _call(tool, {"code": "lines(1:3, 1:3)", "slot": "pres"})

    start = next(c for c in seen if "Bonito.Server" in c)
    assert "__JUTUL_WEB_ORDER__ = String[]" in start
    render = next(c for c in seen if "Bonito.route!" in c)
    # Re-plotting a slot reuses its route, so the earlier entry is dropped rather
    # than leaving the same route listed twice.
    assert 'filter!(!=(raw"/viz/pres"), _order)' in render
    assert 'push!(_order, raw"/viz/pres")' in render


async def test_web_recapture_reports_a_view_it_cannot_read(tmp_path: Path) -> None:
    """A hidden plot must report, not throw.

    Only the view the canvas currently shows is rendered by the browser; any other
    answers the snapshot request with nothing, and decoding that empty reply throws
    from deep inside the backend. Letting it escape puts a Julia backtrace where an
    instruction belongs, since the user just has to select that plot's tab.
    """

    session, seen = _recording(tmp_path)
    tool = make_recapture_tool(session, surface="web")

    await _recapture(tool, {"slot": "pres", "view": False})
    call = next(c for c in seen if "colorbuffer" in c)

    # The read is guarded and an empty frame is caught before it reaches the decoder.
    assert "try" in call and "catch" in call
    assert "isempty(_img)" in call
    # Both ways of not being readable say the same actionable thing.
    assert call.count("has not been drawn in the canvas") == 2
    assert "select its tab there once" in call
    # Printed, not raised, so no Julia backtrace rides along with the instruction.
    assert "error(" not in call
    assert f'println("{jl_src.WEB_RECAPTURE_REFUSED}"' in call


async def test_web_recapture_refusal_is_reported_without_a_backtrace(tmp_path: Path) -> None:
    """The refusal reaches the model as its instruction and nothing else."""

    async def fake_eval(code: str) -> EvalResult:
        if "colorbuffer" in code:
            return EvalResult(output=f"{jl_src.WEB_RECAPTURE_REFUSED}bring it up in the canvas.")
        return EvalResult(output="")

    session = _session(tmp_path, FakeJulia(eval_handler=fake_eval))
    tool = make_recapture_tool(session, surface="web")

    result = await _recapture(tool, {"slot": "pres", "view": False})
    assert result == "ERROR: bring it up in the canvas."
    assert jl_src.WEB_RECAPTURE_REFUSED not in result


async def test_close_plots_on_web_releases_the_live_route(tmp_path: Path) -> None:
    """Closing must unroute the figure, which is what frees what it holds.

    The native path closes GLMakie windows, of which the web surface has none, so
    it reports success while the figure stays routed and in memory.
    """

    session, seen = _recording(tmp_path)
    tool = make_close_plots_tool(session, surface="web")

    await tool.ainvoke(
        {"type": "tool_call", "name": "close_plots", "id": "c1", "args": {"slot": "pres"}}
    )
    call = seen[-1]
    assert "delete_route!" in call and "/viz/pres" in call
    assert "__JUTUL_WEB_FIGS__" in call  # dropped from the registry, not just unrouted
    assert "JutulAgentPlots.close_windows" not in call

    # Closing everything sweeps the registry rather than naming one route.
    await tool.ainvoke({"type": "tool_call", "name": "close_plots", "id": "c2", "args": {}})
    assert "keys(Main.__JUTUL_WEB_FIGS__)" in seen[-1]


async def test_close_plots_on_tui_still_closes_native_windows(tmp_path: Path) -> None:
    session, seen = _recording(tmp_path)
    tool = make_close_plots_tool(session)  # default surface = tui

    await tool.ainvoke({"type": "tool_call", "name": "close_plots", "id": "c1", "args": {}})
    assert "JutulAgentPlots.close_windows" in seen[-1]


async def test_recapture_on_tui_still_uses_the_native_window(tmp_path: Path) -> None:
    """The GLMakie path is the right one off the web surface and must stay put."""

    session, seen = _recording(tmp_path)
    tool = make_recapture_tool(session)  # default surface = tui

    await _recapture(tool, {"slot": "pres", "view": False})
    assert any("JutulAgentPlots.recapture" in c for c in seen)
    assert not any("colorbuffer" in c for c in seen)


async def test_web_surface_loads_the_warm_package_before_it_serves(tmp_path: Path) -> None:
    # Session warm-up loads the warm package too, but in the background: relying on
    # that makes an early plot race it. Loading through the same statement warm-up
    # uses means whichever gets there first establishes the baked load order.
    seen: list[str] = []

    async def fake_eval(code: str) -> EvalResult:
        seen.append(code)
        if "CairoMakie.Makie === WGLMakie.Makie" in code:
            return EvalResult(output="JUTUL_MAKIE_MATCH")
        if "Bonito.Server" in code:
            return EvalResult(output="__JUTUL_WEB_PORT__=51000")
        return EvalResult(output="")

    adapter = make_fake_adapter(tmp_path, warm_package="JutulAgentFakeSim")
    session = Session.create(
        julia=FakeJulia(eval_handler=fake_eval), state_root=tmp_path, simulator=adapter
    )
    tool = make_plot_julia_tool(session, surface="web")

    await _call(tool, {"code": "lines(1:3, 1:3)", "slot": "pres"})

    expected = load_statement("JutulAgentFakeSim")
    load = next(i for i, c in enumerate(seen) if c.strip() == expected)
    serve = next(i for i, c in enumerate(seen) if "Bonito.route!" in c)
    assert load < serve
