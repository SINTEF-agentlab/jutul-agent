"""How many packages Julia may precompile at once here.

Julia sizes this from the core count alone (``base/precompilation.jl``):
``CPU_THREADS/2 + 1`` on Windows, ``CPU_THREADS + 1`` elsewhere, capped at 16. A
worker precompiling a package of the Makie/Jutul class peaks at gigabytes, so on
a many-core machine with ordinary memory that default oversubscribes RAM several
times over, and a machine that runs out while precompiling does not report an
error: it dies with a native fault (``0xC0000005`` on Windows, ``SIGKILL`` under
the Linux OOM killer) that says nothing about the cause.

One module because two places precompile a whole environment and both can be the
one that dies: ``simulators.env_setup`` when a workspace is prepared, and
``sysimage_build`` before an image is baked. The second is the worse of the two
(see :func:`jutul_agent.sysimage_build._precompile_for_the_build`), but nothing
about the policy differs between them.
"""

from __future__ import annotations

import os

# Julia's own knob. Set for every Julia we run that might precompile, and
# inherited by any Julia those in turn spawn -- which is the point, since
# PackageCompiler runs its own precompile from inside ``create_sysimage``.
PRECOMPILE_TASKS_ENV_VAR = "JULIA_NUM_PRECOMPILE_TASKS"

# What one worker peaks at for a package of the Makie/Jutul class. Deliberately
# generous: guessing high costs a slower precompile, guessing low costs the
# machine, with no error to explain it.
GIB_PER_PRECOMPILE_TASK = 4

# Julia caps its own default here, so neither do we go above it.
JULIA_MAX_PRECOMPILE_TASKS = 16


def precompile_task_limit() -> int | None:
    """The ceiling for this machine, judged by memory rather than by cores.

    A ceiling, never a floor: Julia's own default wins where it is already lower,
    so memory to spare never adds parallelism. ``None`` where the memory cannot be
    read, which leaves Julia's default alone rather than guessing at it blind.
    """

    total = _total_memory_bytes()
    if total is None:
        return None
    by_memory = int(total // (GIB_PER_PRECOMPILE_TASK * 2**30))
    return max(1, min(by_memory, julia_default_precompile_tasks()))


def julia_environment() -> dict[str, str]:
    """This process's environment with the ceiling applied, for a Julia subprocess.

    An operator who has set the variable keeps their value: this is a default for
    machines nobody has tuned, not a policy that overrides one who has.
    """

    env = dict(os.environ)
    if PRECOMPILE_TASKS_ENV_VAR in env:
        return env
    limit = precompile_task_limit()
    if limit is not None:
        env[PRECOMPILE_TASKS_ENV_VAR] = str(limit)
    return env


def julia_default_precompile_tasks() -> int:
    """Julia's own default, mirrored from ``base/precompilation.jl``."""

    cpus = os.cpu_count() or 1
    default = cpus // 2 + 1 if os.name == "nt" else cpus + 1
    return min(default, JULIA_MAX_PRECOMPILE_TASKS)


def _total_memory_bytes() -> int | None:
    """Physical RAM, or ``None`` where it cannot be read.

    Read from the OS rather than through ``psutil``, which only arrives with the
    ``eval`` extra and so is absent from an ordinary install -- exactly the
    installs this has to work on. ``sysconf`` covers Linux and macOS;
    ``GlobalMemoryStatusEx`` covers Windows, which is where the ceiling matters
    most, since it is the platform whose out-of-memory death carries no message.
    """

    if os.name == "nt":
        return _windows_total_memory_bytes()
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (OSError, ValueError, AttributeError):
        return None


def _windows_total_memory_bytes() -> int | None:
    """``ullTotalPhys`` from ``GlobalMemoryStatusEx``, or ``None`` if the call fails."""

    import ctypes

    class _MemoryStatusEx(ctypes.Structure):
        # The layout the API expects; only ullTotalPhys is read, but the struct
        # has to be declared whole or dwLength is wrong and the call refuses.
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    try:
        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
            return None
        return int(status.ullTotalPhys) or None
    except Exception:
        return None
