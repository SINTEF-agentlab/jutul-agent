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
from typing import Any


def load_statement(warm_package: str) -> str:
    """The ``using`` that brings up the simulator's Julia world, warm package first.

    The warm package depends on the shared one, so naming it first loads both in the
    order they were baked in. Order is not cosmetic here: a pkgimage is only valid
    for the world it was built in, and pulling the shared package in ahead of the
    simulator's drops the parts of the simulator's bake that were inferred through a
    method the shared one brings, which are then rebuilt at first use. Naming the
    shared package second costs nothing, since by then it is loaded and this only
    binds it into ``Main``, which is what the generated plot code calls it through.

    A simulator with no warm package loads the shared one alone: it is where the
    figure-capture helpers live, and is needed either way.
    """

    return f"using {warm_package}, JutulAgent" if warm_package else "using JutulAgent"


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


def start_warmup(julia: Any, warm_package: str) -> asyncio.Task[Any] | None:
    """Background warm-up shared by every front end: load the simulator's Julia world,
    pin HYPRE's threads, then initialise the GL context.

    The GL step also runs ``GLMakie.activate!(visible = false)``, which is what keeps
    a native plot window from popping up on a machine with a real display — every
    front end wants its plots offscreen (the TUI shows a PNG, the web serves WGLMakie).
    Best-effort: each step is wrapped so a missing piece never breaks startup, and the
    returned task is cancelled on session teardown.
    """
    bootstrap = f"try; @eval {load_statement(warm_package)}; catch; end"

    async def _run() -> None:
        with contextlib.suppress(Exception):
            await julia.eval(bootstrap)
        with contextlib.suppress(Exception):
            await julia.eval(HYPRE_THREADS_SETUP)
        with contextlib.suppress(Exception):
            await julia.eval(GL_CONTEXT_WARMUP)

    return asyncio.create_task(_run(), name="julia-warmup")


# Pin HYPRE's OpenMP thread count for this session. JutulDarcy loads HYPRE (its
# default CPR preconditioner) and lazily calls `HYPRE.Init()` with one thread at the
# first solve. We look up the loaded HYPRE module by UUID — so this is a no-op for a
# simulator that doesn't pull HYPRE in, and needs no env-level dependency — then
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
