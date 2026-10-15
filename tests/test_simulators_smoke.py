"""Per-simulator smoke test: each env instantiates and its package loads.

This is the cheapest signal that an upstream change broke jutul-agent: if a
simulator (or Jutul underneath it) ships a breaking release, ``using <Sim>``
fails to precompile/load and this test goes red. It runs in the dedicated
``Simulators`` workflow (Linux matrix, every simulator on every PR), not in
the cross-OS lane; see ``.github/workflows/simulators.yml``.

Gating mirrors the other integration tests: a simulator runs only when its env
is instantiated, which we detect by the generated (gitignored) ``Manifest.toml``.
The CI job instantiates the matrix simulator's env first; locally, a simulator
participates once you've run ``Pkg.instantiate()`` on its env.
"""

from __future__ import annotations

import shutil

import pytest

from jutul_agent.juliakernel import JuliaKernel, KernelConfig
from jutul_agent.simulators import registry
from jutul_agent.simulators.base import SimulatorAdapter

pytestmark = pytest.mark.integration

_ADAPTERS = [registry.get(name) for name in registry.names()]


def _env_ready(adapter: SimulatorAdapter) -> bool:
    env = adapter.julia_env_template_path
    return (
        shutil.which("julia") is not None
        and (env / "Project.toml").exists()
        and (env / "Manifest.toml").exists()
    )


@pytest.mark.parametrize("adapter", _ADAPTERS, ids=[a.name for a in _ADAPTERS])
async def test_simulator_env_loads(adapter: SimulatorAdapter) -> None:
    env = adapter.julia_env_template_path
    if not _env_ready(adapter):
        pytest.skip(f"{adapter.name} env not instantiated (run `Pkg.instantiate()` on {env})")

    config = KernelConfig(julia_project=env)
    async with JuliaKernel(config) as julia:
        result = await julia.eval(f"using {adapter.primary_package}")
        assert result.error is None, f"{adapter.name}: {result.error}"
        loaded = await julia.eval(f"@isdefined({adapter.primary_package})")
        assert "true" in loaded.output, f"{adapter.name}: module not in scope ({loaded.output!r})"


@pytest.fixture
def warm_display():
    """A DISPLAY for the warm package, faithful to production.

    Warm packages load GLMakie, whose module init needs an OpenGL context.
    Production gives the kernel a private Xvfb display via
    ``display.managed_display``; the test does the same. Yields ``None`` when a
    real display is already present.
    """
    from jutul_agent.display import has_display, managed_display, xvfb_available

    if has_display():
        yield None
        return
    if not xvfb_available():
        pytest.skip("no display and Xvfb is not available")
    with managed_display() as display:
        yield display


@pytest.mark.parametrize(
    "adapter",
    [a for a in _ADAPTERS if a.warm_package],
    ids=[a.name for a in _ADAPTERS if a.warm_package],
)
async def test_simulator_warm_package_loads(adapter: SimulatorAdapter, warm_display) -> None:
    """The env's warm package (and so GLMakie) actually loads.

    ``using <Sim>`` alone stays green when the GL half of the env failed to
    precompile; loading the warm package is the check that the env works for
    the agent, plotting included.
    """
    env = adapter.julia_env_template_path
    if not _env_ready(adapter):
        pytest.skip(f"{adapter.name} env not instantiated (run `Pkg.instantiate()` on {env})")

    kernel_env = {"DISPLAY": warm_display} if warm_display else None
    config = KernelConfig(julia_project=env, env=kernel_env)
    async with JuliaKernel(config) as julia:
        result = await julia.eval(f"using {adapter.warm_package}")
        assert result.error is None, f"{adapter.name}: {result.error}"
        loaded = await julia.eval(f"@isdefined({adapter.warm_package})")
        assert "true" in loaded.output, f"{adapter.name}: warm package not in scope"


@pytest.mark.parametrize(
    "adapter",
    [a for a in _ADAPTERS if a.warm_package],
    ids=[a.name for a in _ADAPTERS if a.warm_package],
)
async def test_simulator_warm_workload_runs(adapter: SimulatorAdapter, warm_display) -> None:
    """The warm package's workload still runs, so its bake is not a silent no-op.

    ``@compile_workload`` has to swallow exceptions: a context-less precompile
    cannot plot, and must still bake the solve. The cost is that a workload which
    has drifted out of step with its simulator's API bakes nothing and says nothing.
    The env looks built, loads fine, and every first call is slow again, none of
    which any other test here can see.

    Each warm package therefore exposes ``_warm()``: the same workload without the
    swallow. Running it is cheap when the bake landed and slow-but-green when it did
    not, so what this catches is the workload erroring out.
    """
    env = adapter.julia_env_template_path
    if not _env_ready(adapter):
        pytest.skip(f"{adapter.name} env not instantiated (run `Pkg.instantiate()` on {env})")

    kernel_env = {"DISPLAY": warm_display} if warm_display else None
    config = KernelConfig(julia_project=env, env=kernel_env)
    async with JuliaKernel(config) as julia:
        result = await julia.eval(f"using {adapter.warm_package}")
        assert result.error is None, f"{adapter.name}: {result.error}"
        ran = await julia.eval(f"{adapter.warm_package}._warm()")
        assert ran.error is None, f"{adapter.name}: warm workload failed: {ran.error}"


# HYPRE's OpenMP thread count, or 0 where the simulator doesn't load HYPRE.
_HYPRE_THREADS = """let hypre = get(Base.loaded_modules,
        Base.PkgId(Base.UUID("b5ffcf37-a2bd-41ab-a3da-4bd9bc8ad771"), "HYPRE"), nothing)
    hypre === nothing ? 0 : hypre.NumThreads()
end"""


@pytest.mark.parametrize(
    "adapter",
    [a for a in _ADAPTERS if a.warm_package],
    ids=[a.name for a in _ADAPTERS if a.warm_package],
)
async def test_a_reset_comes_back_warm(adapter: SimulatorAdapter, tmp_path, monkeypatch) -> None:
    """A reset returns the session to how it started, not to a bare Julia.

    The warm-up binds GLMakie into ``Main``, which plotting code calls ``Figure``
    and ``lines`` through, and pins HYPRE's threads where the simulator loads it.
    A fresh process has neither until the warm-up runs on it too: plots fail with
    an ``UndefVarError`` and every solve runs HYPRE on one thread, unannounced.
    """
    from jutul_agent.display import has_display, xvfb_available
    from jutul_agent.julia.threads import HYPRE_THREADS_ENV_VAR
    from jutul_agent.session_core import start_core

    env = adapter.julia_env_template_path
    if not _env_ready(adapter):
        pytest.skip(f"{adapter.name} env not instantiated (run `Pkg.instantiate()` on {env})")
    if not (has_display() or xvfb_available()):
        pytest.skip("no display and Xvfb is not available")
    monkeypatch.setenv(HYPRE_THREADS_ENV_VAR, "2")  # not HYPRE's own default of one
    core = await start_core(
        simulator=adapter,
        surface="tui",
        workspace=tmp_path,
        state_root=tmp_path,
        julia_project=env,
        prepare_env=False,
        discover=False,
        sysimage=False,
    )

    async def warm_state() -> tuple[str, str]:
        await core.warmup_task
        figure = await core.session.julia.eval("isdefined(Main, :Figure)")
        threads = await core.session.julia.eval(_HYPRE_THREADS)
        return figure.output, threads.output

    try:
        core.start_warmup()
        started = await warm_state()
        assert started[0] == "true", f"{adapter.name}: the warm-up did not bind GLMakie"
        await core.session.julia.reset()
        assert await warm_state() == started
    finally:
        await core.aclose()
