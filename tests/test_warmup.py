"""Tests for the background Julia warmup task and ``JuliaSession.reset``."""

from __future__ import annotations

import asyncio

import pytest

from jutul_agent.julia.session import EvalResult
from jutul_agent.lab.fakes import FakeJulia
from jutul_agent.simulators.warmup import (
    GL_CONTEXT_WARMUP,
    YieldsToWork,
    load_statement,
    start_warmup,
)


def test_load_statement_names_the_warm_package_first() -> None:
    # The warm package depends on the shared one, so naming it first loads both in
    # the order they were baked in. Naming the shared one first would put the session
    # in a different world than the bake, and the difference is rebuilt at first use.
    assert load_statement("JutulAgentBattMo") == "using JutulAgentBattMo, JutulAgent"


def test_load_statement_binds_the_shared_package_into_main() -> None:
    # Loading the warm package does not bring the shared package's own name into
    # scope, and the generated plot code calls it as `JutulAgent.JutulAgentPlots`.
    for warm_package in ("JutulAgentBattMo", ""):
        assert load_statement(warm_package).endswith("JutulAgent")


def test_load_statement_names_capability_packages_after_the_simulator() -> None:
    # A capability package is a sibling of the warm package, not a dependant, so the
    # order is settled by which bake survives rather than by depth. Measured on the
    # geoteric capability: first costs 6.08s of recompilation at load, second 1.56s.
    assert load_statement("JutulAgentJutulDarcy", ("GeotericAgenticDemo",)) == (
        "using JutulAgentJutulDarcy, GeotericAgenticDemo, JutulAgent"
    )


def test_load_statement_keeps_capability_packages_without_a_warm_package() -> None:
    # A simulator with no warm package still has to load the capability's, or the
    # first tool call that needs it pays the load at the user's expense.
    assert load_statement("", ("Geo", "Flow")) == "using Geo, Flow, JutulAgent"


async def test_start_warmup_loads_capability_packages_too() -> None:
    julia = FakeJulia()
    task = start_warmup(julia, "JutulAgentJutulDarcy", ["GeotericAgenticDemo"])
    assert task is not None
    await task
    assert load_statement("JutulAgentJutulDarcy", ["GeotericAgenticDemo"]) in julia.calls[0]


async def test_start_warmup_loads_the_packages_in_baked_order() -> None:
    julia = FakeJulia()
    task = start_warmup(julia, "JutulAgentBattMo")
    assert task is not None
    await task
    assert load_statement("JutulAgentBattMo") in julia.calls[0]
    assert julia.calls[-1] == GL_CONTEXT_WARMUP


async def test_start_warmup_without_warm_package_still_loads_shared() -> None:
    julia = FakeJulia()
    task = start_warmup(julia, "")
    assert task is not None
    await task
    # A placeholder sim with no warm package still loads the shared runtime, which is
    # where the capture helpers live, and warms the GL context.
    assert "using JutulAgent;" in julia.calls[0]
    assert julia.calls[-1] == GL_CONTEXT_WARMUP


async def test_start_warmup_swallows_errors_so_startup_does_not_break() -> None:
    def boom(_code: str) -> EvalResult:
        raise RuntimeError("simulated env-load failure")

    julia = FakeJulia(eval_handler=boom)
    task = start_warmup(julia, "JutulAgentBattMo")
    assert task is not None
    # The task should finish without re-raising; startup must not block on
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


# ---- YieldsToWork: real work drops the background warm-up -------------------


class _SlowJulia:
    """A kernel stand-in whose eval blocks until released, recording every call."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.gate = asyncio.Event()
        self.cancelled = 0
        self.attr = "kernel-attr"

    async def eval(self, code: str, on_chunk: object = None) -> EvalResult:
        self.calls.append(code)
        try:
            await self.gate.wait()
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        return EvalResult(output="")


async def test_real_work_cancels_a_running_warmup() -> None:
    inner = _SlowJulia()
    julia = YieldsToWork(inner)
    task = start_warmup(inner, "JutulAgentJutulDarcy")
    assert task is not None
    julia.set_warmup(task)
    await asyncio.sleep(0)  # let the warm-up reach its first eval

    inner.gate.set()  # so the caller's own eval can complete
    await julia.eval("real work")
    with pytest.raises(asyncio.CancelledError):
        await task
    assert "real work" in inner.calls
    # The in-flight load is NOT interrupted: cancelling a `using` half-initialises
    # Julia's module system and the kernel dies on the next use of it.
    assert inner.cancelled == 0


async def test_warmup_evals_do_not_cancel_themselves() -> None:
    # The warm-up is handed the kernel, not the wrapper, so its own evals never
    # reach the cancelling path. If they did it would kill itself on its first step.
    inner = _SlowJulia()
    inner.gate.set()
    julia = YieldsToWork(inner)
    task = start_warmup(inner, "JutulAgentJutulDarcy")
    assert task is not None
    julia.set_warmup(task)
    await task
    assert not task.cancelled()
    assert len(inner.calls) == 3  # bootstrap, HYPRE, GL


async def test_dropping_is_idempotent_and_safe_without_a_warmup() -> None:
    inner = _SlowJulia()
    inner.gate.set()
    julia = YieldsToWork(inner)
    julia.drop_warmup()  # no warm-up registered at all
    await julia.eval("a")

    task = start_warmup(inner, "")
    assert task is not None
    julia.set_warmup(task)
    await task  # already finished
    julia.drop_warmup()
    julia.drop_warmup()  # a second drop must not raise
    await julia.eval("b")
    assert "b" in inner.calls


async def test_wrapper_forwards_everything_else_to_the_kernel() -> None:
    inner = _SlowJulia()
    julia = YieldsToWork(inner)
    assert julia.attr == "kernel-attr"


async def test_a_cancelled_warmup_leaves_the_session_usable() -> None:
    # The point of cancelling rather than waiting: the next eval still runs. The
    # kernel's own eval turns cancellation into an interrupt-and-recover, so the
    # session survives; here we assert the wrapper does not get in the way of that.
    inner = _SlowJulia()
    julia = YieldsToWork(inner)
    task = start_warmup(inner, "JutulAgentJutulDarcy")
    assert task is not None
    julia.set_warmup(task)
    await asyncio.sleep(0)

    inner.gate.set()
    first = await julia.eval("after cancel")
    assert first.output == ""
    assert inner.cancelled >= 0  # the warm-up eval was cancelled or had not started
    second = await julia.eval("still working")
    assert second.output == ""
    assert inner.calls[-1] == "still working"


async def test_warmup_is_left_alone_while_it_is_still_loading() -> None:
    # Loading packages and building the GL context are what cancelling cannot touch.
    # Measured: a session that cancelled 15s in lost the kernel 13.5s into the tool
    # call that did it. Real work waits instead -- it needed those packages anyway.
    inner = _SlowJulia()
    julia = YieldsToWork(inner)
    abandonable = asyncio.Event()
    task = start_warmup(inner, "JutulAgentJutulDarcy", abandonable=abandonable)
    assert task is not None
    julia.set_warmup(task, abandonable)
    await asyncio.sleep(0)

    julia.drop_warmup()
    await asyncio.sleep(0)
    assert not task.cancelled(), "the load must not be cancelled"
    assert not task.done()

    inner.gate.set()  # let the warm-up run to completion
    await task
    assert abandonable.is_set()


async def test_warmup_is_dropped_once_the_loading_is_done() -> None:
    # Past the load, the capability's warm_code is genuinely optional, so real work
    # takes the kernel rather than queueing behind it.
    inner = _SlowJulia()
    inner.gate.set()
    julia = YieldsToWork(inner)
    abandonable = asyncio.Event()
    task = start_warmup(
        inner, "JutulAgentJutulDarcy", warm_code=["heavy()"], abandonable=abandonable
    )
    assert task is not None
    julia.set_warmup(task, abandonable)

    while not abandonable.is_set():
        await asyncio.sleep(0)
    inner.gate.clear()  # the warm_code step now blocks
    await asyncio.sleep(0)

    julia.drop_warmup()
    await asyncio.sleep(0)
    assert task.cancelling() or task.cancelled() or task.done()
