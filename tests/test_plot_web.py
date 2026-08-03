"""The web surface renders plots as interactive HTML (WGLMakie + Bonito)."""

from __future__ import annotations

from pathlib import Path

from fakes import FakeJulia, make_fake_adapter
from jutul_agent.agent.plot_julia import make_plot_julia_tool
from jutul_agent.julia.session import EvalResult
from jutul_agent.session import Session
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
            # would wrongly pick — so this guards the tagged-line parse.
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
        # The live URL uses the port the server reported (51000), not whatever the
        # tool requested — so it points at where the figures are actually served.
        assert "127.0.0.1:51000/viz/" in (artifact.payload["live_url"] or "")
    finally:
        log.close()


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
        assert "127.0.0.1:51000/viz/" in (artifact.payload["live_url"] or "")
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
    seen: list[str] = []

    async def fake_eval(code: str) -> EvalResult:
        seen.append(code)
        return EvalResult(output="")

    session = _session(tmp_path, FakeJulia(eval_handler=fake_eval))
    tool = make_plot_julia_tool(session)  # default surface = tui

    await _call(tool, {"code": "lines(1:3, 1:3)"})
    assert any(c.strip() == "using GLMakie" for c in seen)
    assert not any("Bonito.export_static" in c for c in seen)
