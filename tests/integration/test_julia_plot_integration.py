"""Integration tests: real Julia + GLMakie via plot_julia.

Three intentional layers:

- ``test_plot_julia_captures_figure_shapes`` and ``..._view_returns_image_blocks``
  exercise *our* capture/view machinery against plain Makie. They don't touch a
  simulator's plotters, so they stay stable across upstream changes; they guard
  the code we own.
- ``test_native_plotters_render_to_png`` is a *canary* over JutulDarcy's own
  plotters: it confirms the end-to-end wiring works today, and it embeds
  JutulDarcy's setup/plotter API, so it will need updating when that API drifts.

Skipped unless the JutulDarcy env is instantiated (the shipped template has no
Manifest). The shared JutulAgent package lives once in ``julia_runtime/`` and is
synced into the env at bootstrap; an in-place instantiate must do the same first.
To run locally:

    uv run python -c "from pathlib import Path; from jutul_agent.workspace import \
        sync_shared_julia_package as s; \
        s(Path('src/jutul_agent/simulators/jutuldarcy/julia_env'))"
    julia --project=src/jutul_agent/simulators/jutuldarcy/julia_env \
        -e 'using Pkg; Pkg.instantiate()'
    uv run pytest tests/integration/test_plot_julia_integration.py

On headless Linux GLMakie needs a virtual display; the ``plot_display`` fixture
starts a private Xvfb (as production does via ``managed_display``). CI runs these
in the `plot-integration` job.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from jutul_agent.agent import plot_julia_src as jl
from jutul_agent.agent.plot_julia import make_plot_julia_tool
from jutul_agent.display import has_display, managed_display, xvfb_available
from jutul_agent.juliakernel import JuliaKernel, KernelConfig
from jutul_agent.session import Session
from jutul_agent.simulators.env_setup import manifest_has_package
from jutul_agent.simulators.jutuldarcy import JUTULDARCY
from jutul_agent.workspace import sync_shared_julia_package

JUTULDARCY_ENV = JUTULDARCY.julia_env_template_path


@pytest.fixture(autouse=True)
def _materialize_shared_package() -> None:
    """Keep the shared JutulAgent package present in the in-place template env.

    It lives once in ``julia_runtime/`` and is copied into an env at bootstrap;
    these tests instantiate the template directly, so sync it here too (idempotent).
    Skipped when the env was never instantiated; those tests skip anyway.
    """
    if (JUTULDARCY_ENV / "Project.toml").exists():
        sync_shared_julia_package(JUTULDARCY_ENV)


@pytest.fixture
def plot_display():
    """A DISPLAY for GLMakie, faithful to production.

    GLMakie needs an OpenGL context; with none, its ``save`` *hard-crashes* the
    Julia process (a GL abort, not a catchable Julia error), surfaced as "the
    Julia kernel exited unexpectedly". Production gives the kernel a private Xvfb
    display via ``display.managed_display`` (passed through ``DISPLAY``);
    the test does the same, so it doesn't depend on the outer process owning one.
    Yields ``None`` when a real display is already present (use the ambient one).
    """

    if has_display():
        yield None
        return
    if not xvfb_available():
        pytest.skip("no display and Xvfb is not available")
    with managed_display() as display:
        yield display


def _plot_kernel_config(display: str | None) -> KernelConfig:
    env = {"DISPLAY": display} if display else None
    return KernelConfig(julia_project=JUTULDARCY_ENV, env=env)


def _julia_available() -> bool:
    return shutil.which("julia") is not None


def _gl_ready(env_dir: Path) -> bool:
    """Env is instantiated *and* actually resolved GLMakie."""
    return (
        (env_dir / "Project.toml").exists()
        and (env_dir / "Manifest.toml").exists()
        and manifest_has_package(env_dir, "GLMakie")
    )


def _wgl_ready(env_dir: Path) -> bool:
    """Env is instantiated *and* actually resolved WGLMakie (the web plotting stack)."""
    return (
        (env_dir / "Project.toml").exists()
        and (env_dir / "Manifest.toml").exists()
        and manifest_has_package(env_dir, "WGLMakie")
    )


def _session(julia: JuliaKernel, tmp_path: Path, sid: str) -> Session:
    return Session.create(julia=julia, state_root=tmp_path, simulator=JUTULDARCY, session_id=sid)


async def _plot(tool, code: str, slot: str):
    msg = await tool.ainvoke(
        {
            "type": "tool_call",
            "name": "plot_julia",
            "id": f"call_{slot}",
            "args": {"code": code, "slot": slot},
        }
    )
    return str(getattr(msg, "content", msg))


pytestmark = pytest.mark.integration


# Every Makie return shape that `_as_figure` must resolve, built from plain Makie so
# the test stays stable across simulator-package changes. `nonfig` is the negative
# case: a non-figure value must NOT capture the (now stale) current figure.
_FIGURE_SHAPES = {
    "returns_figure": "fig = Figure(); lines!(Axis(fig[1, 1]), 1:3); fig",
    "returns_figureaxisplot": "lines(1:3)",
    "returns_tuple": "fig = Figure(); ax = Axis(fig[1, 1]); (fig, ax, lines!(ax, 1:3))",
    "draws_then_nothing": "fig = Figure(); lines!(Axis(fig[1, 1]), 1:3); display(fig); nothing",
}


@pytest.mark.skipif(
    not _julia_available() or not _gl_ready(JUTULDARCY_ENV),
    reason="Julia and a GLMakie-instantiated jutuldarcy env are required",
)
async def test_plot_julia_captures_figure_shapes(tmp_path: Path, plot_display) -> None:
    """plot_julia captures every Makie return shape, and errors (not captures a stale
    figure) when the code draws nothing new. Guards our capture logic, not anyone's
    plotters."""
    config = _plot_kernel_config(plot_display)
    async with JuliaKernel(config) as julia:
        session = _session(julia, tmp_path, "capture-shapes")
        tool = make_plot_julia_tool(session)

        failures: list[str] = []
        for slot, code in _FIGURE_SHAPES.items():
            result = await _plot(tool, code, slot)
            png = session.output_dir / "artifacts" / f"{slot}.png"
            if "saved plot to" not in result:
                failures.append(f"{slot}: tool error: {result}")
            elif not (png.exists() and png.stat().st_size > 500):
                failures.append(f"{slot}: no/blank PNG")

        # A non-figure return must error and write nothing, even though a previous
        # plot (and warm-up) left a current figure behind.
        result = await _plot(tool, "1 + 1", "nonfig")
        if not result.startswith("ERROR"):
            failures.append(f"nonfig: expected an error, got: {result}")
        if (session.output_dir / "artifacts" / "nonfig.png").exists():
            failures.append("nonfig: wrote a PNG for a non-figure value")

        session.finalize()
        assert not failures, "capture-shape failures:\n" + "\n".join(failures)


@pytest.mark.skipif(
    not _julia_available() or not _gl_ready(JUTULDARCY_ENV),
    reason="Julia and a GLMakie-instantiated jutuldarcy env are required",
)
async def test_plot_julia_view_returns_image_blocks(tmp_path: Path, plot_display) -> None:
    """view=True returns text + image content blocks the model can see."""
    config = _plot_kernel_config(plot_display)
    async with JuliaKernel(config) as julia:
        session = _session(julia, tmp_path, "view-image")
        tool = make_plot_julia_tool(session)
        code = (
            "fig = Figure(size = (320, 240))\n"
            "lines!(Axis(fig[1, 1]), 1:4, [1.0, 4.0, 9.0, 16.0])\n"
            "fig"
        )
        msg = await tool.ainvoke(
            {
                "type": "tool_call",
                "name": "plot_julia",
                "id": "call_view",
                "args": {"code": code, "view": True},
            }
        )
        content = getattr(msg, "content", msg)
        assert isinstance(content, list)
        types = [b.get("type") for b in content]
        assert "text" in types and "image" in types
        image = next(b for b in content if b["type"] == "image")
        assert image["mime_type"] == "image/png"
        assert len(image["base64"]) > 100
        session.finalize()


# Canary over JutulDarcy's own native plotters: confirms the end-to-end wiring works
# today. These embed JutulDarcy's setup/plotter API (note the setup_reservoir_model
# return convention and the GraphMakie extension), so they may need updating when
# upstream changes; that is expected for a canary.
_NATIVE_PLOTTERS = {
    "reservoir_domain": "plot_reservoir(domain)",
    "reservoir_model": "plot_reservoir(model)",
    "cell_data": "plot_cell_data(physical_representation(domain), domain[:porosity])",
    "variable_graph": (
        "using GraphMakie, NetworkLayout, LayeredLayouts\n"
        "plot_variable_graph(reservoir_model(model))"
    ),
    "model_graph": ("using GraphMakie, NetworkLayout, LayeredLayouts\nplot_model_graph(model)"),
}


@pytest.mark.skipif(
    not _julia_available() or not _gl_ready(JUTULDARCY_ENV),
    reason="Julia and a GLMakie-instantiated jutuldarcy env are required",
)
async def test_native_plotters_render_to_png(tmp_path: Path, plot_display) -> None:
    """Every native plotter we document renders to a non-blank PNG through plot_julia."""
    config = _plot_kernel_config(plot_display)
    async with JuliaKernel(config) as julia:
        session = _session(julia, tmp_path, "native-plotters")
        # setup_reservoir_model returns the model directly (a tuple only with
        # extra_out=true), and takes a system, not an :immiscible shortcut.
        setup = await julia.eval(
            "using JutulDarcy, Jutul\n"
            "g = CartesianMesh((3, 3, 2), (30.0, 30.0, 6.0))\n"
            "domain = reservoir_domain(g; permeability = 1e-13, porosity = 0.2)\n"
            "Inj = setup_well(domain, (1, 1, 1); name = :Inj)\n"
            "Prod = setup_well(domain, (3, 3, 1); name = :Prod)\n"
            "sys = ImmiscibleSystem((LiquidPhase(), VaporPhase()); "
            "reference_densities = [1000.0, 100.0])\n"
            "model = setup_reservoir_model(domain, sys; wells = [Inj, Prod])\n"
            '"setup ok"'
        )
        assert not setup.error, f"model setup failed: {setup.error}"

        # Run every plotter, collecting failures so one broken plotter doesn't mask
        # which of the others render (the report then names exactly what failed).
        tool = make_plot_julia_tool(session)
        failures: list[str] = []
        for slot, code in _NATIVE_PLOTTERS.items():
            result = await _plot(tool, code, slot)
            png = session.output_dir / "artifacts" / f"{slot}.png"
            if "saved plot to" not in result:
                failures.append(f"{slot}: tool error: {result}")
            elif not png.exists():
                failures.append(f"{slot}: no PNG written")
            elif png.stat().st_size <= 1000:
                failures.append(f"{slot}: PNG looks blank ({png.stat().st_size} bytes)")
        session.finalize()
        assert not failures, "native plotters failed:\n" + "\n".join(failures)


@pytest.mark.skipif(
    not _julia_available() or not _wgl_ready(JUTULDARCY_ENV),
    reason="Julia and a WGLMakie-instantiated jutuldarcy env are required",
)
async def test_pick_guard_still_applies_to_the_installed_wglmakie() -> None:
    """The empty-buffer pick guard actually replaces the method it means to.

    This one is a *canary over someone else's code*. The guard patches an upstream
    bug -- WGLMakie's single-point `pick` indexes the pick buffer without checking it
    is non-empty -- by redefining that method. Which means it is pinned to a signature
    we do not own: rename `pick_native`, or resign `Makie.pick`, and the redefinition
    stops applying.

    Nothing would fail loudly if that happened. The plot still renders; the buffer is
    still empty on the first click; the BoundsError comes back, propagates out of the
    Observables notify, and every listener behind the picking handler is skipped -- so
    a mouse button silently stops working in the browser and nothing points at why.

    So assert dispatch actually lands on the replacement: if it resolves back into
    WGLMakie's own picking.jl, the guard is inert and this fails while it is still
    cheap to notice. When upstream checks the buffer itself, this test is the signal
    to drop the guard rather than keep carrying it.
    """

    config = KernelConfig(julia_project=JUTULDARCY_ENV)
    async with JuliaKernel(config) as julia:
        loaded = await julia.eval("import WGLMakie")
        assert not loaded.error, f"WGLMakie did not load: {loaded.error}"

        result = await julia.eval(jl.PICK_EMPTY_BUFFER_GUARD)
        assert not result.error, f"guard raised: {result.error}"
        assert f"{jl.PICK_GUARD_MARKER}=ok" in (result.output or ""), (
            "the pick guard no longer applies to this WGLMakie -- dispatch still lands "
            "in its own picking.jl, so the empty-buffer BoundsError is back and the "
            f"mouse button it kills is dead again. Julia said: {result.output!r}"
        )
