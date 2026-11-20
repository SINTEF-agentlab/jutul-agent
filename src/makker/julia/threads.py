"""Compute-thread policy for the Julia kernel.

One place that decides how many compute threads the kernel's Julia process gets,
and the BLAS coordination that keeps that from oversubscribing the machine. The
launch path (``run.py``) uses :func:`resolve_compute_threads` for the kernel's
``--threads`` and :func:`blas_thread_env` for the environment.

Julia's compute-thread count is fixed at process launch, so the knob is the
``--threads`` flag / environment variable read here, not something the session can
change at runtime.
"""

from __future__ import annotations

import os

# Operator/user override (the ``--threads`` CLI flag takes precedence over it).
# A positive integer pins the count; ``auto`` uses every logical core. Unset
# falls back to the conservative default (physical cores minus one). Julia always
# also gets one interactive thread on top (the kernel's eval/interrupt thread);
# see juliakernel._thread_flag.
THREADS_ENV_VAR = "MAKKER_JULIA_THREADS"

# HYPRE's BoomerAMG (JutulDarcy's CPR pressure preconditioner) is OpenMP-threaded
# via ``HYPRE.SetNumThreads``. Both HYPRE.jl and users report that many threads
# hurt its solver performance, so we cap it well below the Julia compute-thread count.
# A positive integer in the env var overrides.
HYPRE_THREADS_ENV_VAR = "MAKKER_HYPRE_THREADS"
HYPRE_MAX_THREADS = 8


def resolve_compute_threads(cli_value: str | None = None) -> int:
    """How many Julia compute threads the kernel should launch with.

    Precedence: the ``--threads`` CLI flag (``cli_value``), then
    ``MAKKER_JULIA_THREADS``, then the default. Each may be a positive
    integer (used as-is) or ``auto`` (every logical core). The default, when
    nothing is set, is physical cores minus one. Jutul's assembly and the
    preconditioner (through HYPRE) are threaded, so this is a real speed-up
    over one thread while leaving a core for the OS/UI and skipping hyperthreads
    that do little for sparse solves. Override upward with ``--threads auto`` or a
    number.
    """

    for source in (cli_value, os.environ.get(THREADS_ENV_VAR)):
        n = _parse_threads(source)
        if n is not None:
            return n
    return _default_threads()


def _parse_threads(value: str | None) -> int | None:
    """A thread count from a CLI/env value, or ``None`` to defer to the next source."""

    if value is None:
        return None
    v = value.strip().lower()
    if not v:
        return None
    if v == "auto":
        return os.cpu_count() or 1
    try:
        return max(1, int(v))
    except ValueError:
        return None


def _default_threads() -> int:
    """Physical cores minus one (at least one)."""

    return max(1, _physical_cores() - 1)


def _physical_cores() -> int:
    """Best-effort physical core count.

    ``psutil`` reports it directly across platforms; without it we can only see
    logical CPUs, so we assume 2-way SMT and halve — a safe under-estimate that
    still leaves headroom. ``--threads``/env override either way.
    """

    try:
        import psutil

        physical = psutil.cpu_count(logical=False)
        if physical:
            return int(physical)
    except Exception:
        pass
    logical = os.cpu_count() or 1
    return max(1, logical // 2)


def resolve_hypre_threads() -> int:
    """OpenMP threads for HYPRE's BoomerAMG preconditioner.

    Defaults to physical cores minus one, capped at 8. This is enough to speed up the
    pressure solve without the slowdown HYPRE shows at high thread counts. A positive
    integer in ``MAKKER_HYPRE_THREADS`` overrides. Applied in Julia by
    :data:`makker.simulators.warmup.HYPRE_THREADS_SETUP`.
    """

    raw = os.environ.get(HYPRE_THREADS_ENV_VAR, "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return max(1, min(_physical_cores() - 1, HYPRE_MAX_THREADS))


def blas_thread_env(threads: int) -> dict[str, str]:
    """Environment that keeps BLAS from oversubscribing under Julia threading.

    With ``threads`` Julia compute threads, an OpenBLAS that itself defaults to
    every core means up to ``threads * cores`` OS threads inside any dense-BLAS
    region, contention that hurts more than it helps for Jutul's sparse solves.
    Pinning OpenBLAS to one thread is the standard Julia-threading default and
    lets the Julia threads own the parallelism. Only set when we actually thread
    and only when the user hasn't already chosen a value.
    """

    env: dict[str, str] = {}
    if threads > 1 and "OPENBLAS_NUM_THREADS" not in os.environ:
        env["OPENBLAS_NUM_THREADS"] = "1"
    return env
