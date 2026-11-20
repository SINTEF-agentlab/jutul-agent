"""The session-start GL-context warm-up.

Solve and plot paths are precompiled and cached at ``init`` (by the Makker and
per-simulator Makker<Sim> packages), so loading them at session start pays only
load latency. The one thing precompilation cannot bake is GLMakie's runtime GL
context, which is process-local. This module is the one simulator-agnostic snippet
that warms it, in the background, with a tiny offscreen save. ``run._start_warmup``
runs it after loading the warm packages.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
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
    is built on the simulator, but knows nothing of ``Makker<Sim>``, so "most
    derived" does not order the pair, and the question is only which bake survives.
    Measured on one capability package: naming it first costs 6.08s of recompilation
    at load, naming it after the warm package 1.56s. The warm package brings the
    backends and the shared runtime, so going second means arriving into a world that
    is already complete, with nothing left to invalidate it.

    A simulator with no warm package loads the shared one alone: it is where the
    figure-capture helpers live, and is needed either way.
    """

    names = [warm_package] if warm_package else []
    names.extend(n for n in capability_packages if n)
    names.append("Makker")
    return "using " + ", ".join(names)


def fallback_load_statement(simulator_package: str, capability_packages: Sequence[str] = ()) -> str:
    """What to load when the warm package cannot be: the same world, in the order
    the warm package brings it up itself.

    The simulator first (best-effort, the capture helpers do not need it), then the
    shared package, which brings every Makie backend with GLMakie last, then the
    capability packages. Nothing was baked against this, so the first solve and
    plot compile what the warm package would have had ready.
    """

    using = "using " + ", ".join(["Makker", *(n for n in capability_packages if n)])
    if not simulator_package:
        return using
    return f"try; @eval import {simulator_package}; catch; end\n{using}"


@dataclass
class WarmPackage:
    """One session's use of its simulator's warm package, which is an optimisation only.

    ``available`` is false from the start when the env has it on record that the
    package does not build, and turns false the first time loading it fails: a
    failed precompile is retried by every ``using``, at about a minute a time.
    ``on_unavailable`` then hears Julia's error, once.
    """

    name: str
    simulator_package: str = ""
    available: bool = True
    on_unavailable: Callable[[str], None] | None = None
    # The warm-up and the first plot both load; the second waits for the first's verdict.
    _loading: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    def load_statement(self, capability_packages: Sequence[str] = ()) -> str:
        if self.name and not self.available:
            return fallback_load_statement(self.simulator_package, capability_packages)
        return load_statement(self.name, capability_packages)

    async def load(self, julia: Any, capability_packages: Sequence[str] = ()) -> Any:
        """Load the session's Julia world, without the warm package if it will not load."""

        async with self._loading:
            result = await julia.eval(self.load_statement(capability_packages))
            if result.error and self.name and self.available:
                self.available = False
                if self.on_unavailable is not None:
                    self.on_unavailable(result.error)
                result = await julia.eval(self.load_statement(capability_packages))
            return result


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
        save(joinpath(tempdir(), "makker_gl_warmup.png"), fig)
    end
catch
end
"""

WEB_BACKEND_SETUP = """try
    using WGLMakie
    WGLMakie.activate!()
catch
end
"""


def start_warmup(
    julia: Any,
    warm_package: str | WarmPackage,
    capability_packages: Sequence[str] = (),
    warm_code: Sequence[str] = (),
    abandonable: asyncio.Event | None = None,
    *,
    native_plotting: bool = True,
) -> asyncio.Task[Any] | None:
    """Background warm-up shared by every front end: load the simulator's Julia world
    (plus any capability packages), pin HYPRE's threads, then prepare its plot backend.

    Native capture warms an offscreen GL context. Web sessions activate WGLMakie
    without creating a native window: their figures and Cairo posters do not need
    one, so startup avoids failures in native window creation.
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

    ``warm_package`` is the session's :class:`WarmPackage`, or just its name.
    """
    warm = warm_package if isinstance(warm_package, WarmPackage) else WarmPackage(warm_package)

    async def _run() -> None:
        # Shielded: cancelling here is what kills the kernel, so a cancel that lands
        # mid-load has to leave the eval running and take effect at the next step.
        await _shielded(warm.load(julia, capability_packages))
        backend_setup = GL_CONTEXT_WARMUP if native_plotting else WEB_BACKEND_SETUP
        for essential in (HYPRE_THREADS_SETUP, backend_setup):
            await _shielded(julia.eval(essential))
        if abandonable is not None:
            abandonable.set()
        # Run capability snippets after loading and selecting the session's backend.
        for code in warm_code:
            with contextlib.suppress(Exception):
                await julia.eval(code)

    return asyncio.create_task(_run(), name="julia-warmup")


async def _shielded(step: Any) -> None:
    """Await ``step`` so a cancel of the warm-up leaves it running, and quietly."""

    with contextlib.suppress(Exception):
        await asyncio.shield(_quietly(step))


async def _quietly(step: Any) -> None:
    # The failure is swallowed inside the shielded coroutine, not around the
    # shield. A shield whose inner future ends in an exception hands it to the
    # loop's exception handler even when the waiter is long gone, so a session
    # closing mid-warm-up printed a KernelDied traceback that meant nothing.
    with contextlib.suppress(Exception):
        await step


# Pin HYPRE's OpenMP thread count for this session. JutulDarcy loads HYPRE (its
# default CPR preconditioner) and lazily calls `HYPRE.Init()` with one thread at the
# first solve. We look up the loaded HYPRE module by UUID, so this is a no-op for a
# simulator that doesn't pull HYPRE in, and needs no env-level dependency, then
# `Init()` (idempotent: the later lazy Init is a no-op) and `SetNumThreads` to the
# count Python computed (`MAKKER_HYPRE_THREADS`). HYPRE clamps it to
# [1, Sys.CPU_THREADS] internally. Best-effort: any failure leaves HYPRE's
# single-threaded default. ``run._start_warmup`` runs this after the warm packages.
HYPRE_THREADS_SETUP = """try
    let n = tryparse(Int, get(ENV, "MAKKER_HYPRE_THREADS", ""))
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
