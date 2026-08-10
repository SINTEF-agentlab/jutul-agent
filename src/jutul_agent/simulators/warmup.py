"""The session-start GL-context warm-up.

Solve and plot paths are precompiled and cached at ``init`` (by the JutulAgent and
per-simulator JutulAgent<Sim> packages), so loading them at session start pays only
load latency. The one thing precompilation cannot bake is GLMakie's runtime GL
context, which is process-local. This module is the one simulator-agnostic snippet
that warms it, in the background, with a tiny offscreen save. ``run._start_warmup``
runs it after loading the warm packages.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Sequence
from typing import Any


class YieldsToWork:
    """A Julia session that drops the background warm-up as soon as real work arrives.

    The warm-up is only ever a bet that the session will be idle for a while. The
    kernel evaluates serially, so when the bet is wrong the warm-up is no longer free:
    the user's first call queues behind however much of it is left, and they wait
    longer than if nothing had been warmed at all.

    Cancelling the warm-up task is enough to settle that. ``JuliaKernel.eval`` already
    treats cancellation as "interrupt this eval and keep the session", so the in-flight
    warm-up eval is interrupted, its result frame is drained, and the lock is released
    before the caller's own eval takes it. Nothing here has to know how that works.

    The distinction between warm-up and real work is structural rather than inspected:
    the warm-up is handed the kernel itself, and everything else goes through this
    wrapper, so an eval reaching here is by construction not the warm-up's own.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self._warmup: asyncio.Task[Any] | None = None
        self._abandonable: asyncio.Event | None = None

    def set_warmup(
        self, task: asyncio.Task[Any] | None, abandonable: asyncio.Event | None = None
    ) -> None:
        self._warmup = task
        self._abandonable = abandonable

    async def eval(self, code: str, on_chunk: Any = None) -> Any:
        self.drop_warmup()
        return await self._inner.eval(code, on_chunk)

    def drop_warmup(self) -> None:
        """Cancel the warm-up if it has reached a point where that is safe. Idempotent.

        Before that point it is still loading packages and building the GL context,
        and interrupting either takes the kernel down (see ``start_warmup``). So this
        leaves it alone and the caller's eval simply queues behind it on the kernel
        lock -- waiting for work it needed anyway.
        """
        task = self._warmup
        if task is None or task.done():
            return
        if self._abandonable is not None and not self._abandonable.is_set():
            return
        task.cancel()

    # Everything else is the kernel's own: reset, restart, interrupt, the async
    # context manager, and whatever a backend adds.
    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def load_statement(warm_package: str, capability_packages: Sequence[str] = ()) -> str:
    """The ``using`` that brings up the session's Julia world, most-derived first.

    The warm package depends on the shared one, so naming it first loads both in the
    order they were baked in. Order is not cosmetic here: a pkgimage is only valid
    for the world it was built in, and pulling the shared package in ahead of the
    simulator's drops the parts of the simulator's bake that were inferred through a
    method the shared one brings, which are then rebuilt at first use. Naming the
    shared package second costs nothing, since by then it is loaded and this only
    binds it into ``Main``, which is what the generated plot code calls it through.

    ``capability_packages`` follow the simulator's, not the other way round. A
    capability package is a sibling of the warm package rather than a dependant. It
    is built on the simulator, but knows nothing of ``JutulAgent<Sim>``, so "most
    derived" does not order the pair, and the question is only which bake survives.
    Measured on the geoteric capability: naming it first costs 6.08s of recompilation
    at load, naming it after the warm package 1.56s. The warm package brings the
    backends and the shared runtime, so going second means arriving into a world that
    is already complete, with nothing left to invalidate it.

    A simulator with no warm package loads the shared one alone: it is where the
    figure-capture helpers live, and is needed either way.
    """

    names = [warm_package] if warm_package else []
    names.extend(n for n in capability_packages if n)
    names.append("JutulAgent")
    return "using " + ", ".join(names)


# Initialise this session's GLMakie GL context (the per-session cost precompilation
# cannot bake) with a tiny offscreen save. Best-effort: if GLMakie isn't usable here
# (no GL, no xvfb) the try/catch swallows it and plotting errors at first use.
GL_CONTEXT_WARMUP = """try
    using GLMakie  # already loaded by the warm packages; binds it into Main
    GLMakie.activate!(visible = false)
    let
        fig = Figure(size = (96, 96))
        ax = Axis3(fig[1, 1])
        surface!(ax, 1:4, 1:4, [Float64(i + j) for i in 1:4, j in 1:4])
        save(joinpath(tempdir(), "jutul_agent_gl_warmup.png"), fig)
    end
catch
end
"""


def start_warmup(
    julia: Any,
    warm_package: str,
    capability_packages: Sequence[str] = (),
    warm_code: Sequence[str] = (),
    abandonable: asyncio.Event | None = None,
) -> asyncio.Task[Any] | None:
    """Background warm-up shared by every front end: load the simulator's Julia world
    (plus any capability packages), pin HYPRE's threads, then initialise the GL context.

    The GL step also runs ``GLMakie.activate!(visible = false)``, which is what keeps
    a native plot window from popping up on a machine with a real display; every
    front end wants its plots offscreen (the TUI shows a PNG, the web serves WGLMakie).
    Best-effort: each step is wrapped so a missing piece never breaks startup, and the
    returned task is cancelled on session teardown.

    ``abandonable`` is set once the loading is done, and marks the only point from
    which this task can be cancelled safely. Interrupting a ``using`` leaves Julia's
    module system half-initialised, and the damage surfaces later as a dead process
    rather than an error: measured, a session whose first tool call arrived 15s in
    (mid-load) cancelled the warm-up and then lost the kernel 13.5s into that call,
    with an empty stderr and a torn control socket. The GL context is treated the same
    way, since a half-built context takes the process down on the next plot.

    Nothing is lost by waiting: the load is work the first tool call needs regardless,
    so cancelling it only moves the cost, and the caller queues on the kernel lock
    either way. The capability's ``warm_code`` is the genuinely optional part -- extra
    specialisation the session would be fine without -- and that is what stays
    abandonable.
    """
    bootstrap = f"try; @eval {load_statement(warm_package, capability_packages)}; catch; end"

    async def _run() -> None:
        # Shielded: cancelling here is what kills the kernel, so a cancel that lands
        # mid-load has to leave the eval running and take effect at the next step.
        for essential in (bootstrap, HYPRE_THREADS_SETUP, GL_CONTEXT_WARMUP):
            with contextlib.suppress(Exception):
                await asyncio.shield(_eval(julia, essential))
        if abandonable is not None:
            abandonable.set()
        # Last, and only once the backends are up: a capability's snippets exist to
        # warm plot and solve paths, which need the GL context this step above built.
        for code in warm_code:
            with contextlib.suppress(Exception):
                await julia.eval(code)

    return asyncio.create_task(_run(), name="julia-warmup")


async def _eval(julia: Any, code: str) -> Any:
    return await julia.eval(code)


# Pin HYPRE's OpenMP thread count for this session. JutulDarcy loads HYPRE (its
# default CPR preconditioner) and lazily calls `HYPRE.Init()` with one thread at the
# first solve. We look up the loaded HYPRE module by UUID, so this is a no-op for a
# simulator that doesn't pull HYPRE in, and needs no env-level dependency, then
# `Init()` (idempotent: the later lazy Init is a no-op) and `SetNumThreads` to the
# count Python computed (`JUTUL_AGENT_HYPRE_THREADS`). HYPRE clamps it to
# [1, Sys.CPU_THREADS] internally. Best-effort: any failure leaves HYPRE's
# single-threaded default. ``run._start_warmup`` runs this after the warm packages.
HYPRE_THREADS_SETUP = """try
    let n = tryparse(Int, get(ENV, "JUTUL_AGENT_HYPRE_THREADS", ""))
        hypre = get(Base.loaded_modules,
            Base.PkgId(Base.UUID("b5ffcf37-a2bd-41ab-a3da-4bd9bc8ad771"), "HYPRE"), nothing)
        if n !== nothing && n >= 1 && hypre !== nothing
            hypre.Init()
            hypre.SetNumThreads(n)
        end
    end
catch
end
"""
