"""Tests for the background Julia warmup task and ``JuliaSession.reset``."""

from __future__ import annotations

import asyncio

import pytest

from jutul_agent.julia.session import EvalResult
from jutul_agent.lab.fakes import FakeJulia
from jutul_agent.simulators.warmup import GL_CONTEXT_WARMUP, start_warmup


async def test_start_warmup_loads_both_runtime_packages() -> None:
    julia = FakeJulia()
    task = start_warmup(julia, "JutulAgentBattMo")
    assert task is not None
    await task
    # First eval loads the shared runtime + the per-sim warm package; then the GL
    # context warm-up runs.
    assert "using JutulAgent" in julia.calls[0]
    assert "using JutulAgentBattMo" in julia.calls[0]
    assert julia.calls[-1] == GL_CONTEXT_WARMUP


async def test_start_warmup_without_warm_package_still_loads_shared() -> None:
    julia = FakeJulia()
    task = start_warmup(julia, "")
    assert task is not None
    await task
    # A placeholder sim with no warm package still loads the shared runtime and
    # warms the GL context.
    assert "using JutulAgent" in julia.calls[0]
    assert "JutulAgentBattMo" not in julia.calls[0]
    assert julia.calls[-1] == GL_CONTEXT_WARMUP


async def test_start_warmup_swallows_errors_so_startup_does_not_break() -> None:
    def boom(_code: str) -> EvalResult:
        raise RuntimeError("simulated env-load failure")

    julia = FakeJulia(eval_handler=boom)
    task = start_warmup(julia, "JutulAgentBattMo")
    assert task is not None
    # The task should finish without re-raising — startup must not block on
    # a broken simulator env.
    await task


async def test_start_warmup_can_be_cancelled_during_shutdown() -> None:
    started = asyncio.Event()

    async def slow(_code: str) -> EvalResult:
        started.set()
        await asyncio.sleep(60)
        return EvalResult(output="never")

    julia = FakeJulia(eval_handler=slow)
    task = start_warmup(julia, "JutulAgentJutulDarcy")
    assert task is not None
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_fake_julia_reset_counts_invocations() -> None:
    julia = FakeJulia()
    assert julia.reset_count == 0
    result = await julia.reset()
    assert result.output == "reset"
    await julia.reset()
    assert julia.reset_count == 2


def test_all_real_adapters_name_a_warm_package() -> None:
    """Every simulator declares a per-sim warm package (placeholders too)."""

    from jutul_agent.simulators import registry

    for name in registry.names():
        adapter = registry.get(name)
        assert adapter.warm_package == "JutulAgent" + adapter.display_name, (
            f"adapter {name!r} warm_package {adapter.warm_package!r} should be "
            f"'JutulAgent{adapter.display_name}'"
        )


def test_gl_context_warmup_drives_the_offscreen_save_path() -> None:
    """The one irreducible per-session cost: GLMakie's offscreen render+save."""

    assert "GLMakie.activate!(visible = false)" in GL_CONTEXT_WARMUP
    assert "save(" in GL_CONTEXT_WARMUP
    # Self-contained: it binds GLMakie into Main itself (the packages load it only
    # inside their own modules).
    assert "using GLMakie" in GL_CONTEXT_WARMUP
