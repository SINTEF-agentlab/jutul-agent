"""Launch configuration for a :class:`.kernel.JuliaKernel`."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class KernelConfig:
    """How to launch the Julia kernel process.

    Everything the kernel needs to spawn Julia and bring up the control channel:
    the executable, the standard launch options (``julia_project``, ``sysimage``,
    ``threads``, ``extra_args``), the working directory, and any extra
    environment.
    """

    julia_executable: str = "julia"
    julia_project: Path | None = None
    sysimage: Path | None = None
    threads: str | None = None  # value for `--threads` (e.g. "auto", "4")
    extra_args: tuple[str, ...] = field(default_factory=lambda: ("--startup-file=no",))
    cwd: Path | None = None
    # Extra environment for the Julia process, merged over the inherited env.
    # Prefer setting variables here (e.g. ``DISPLAY``) over wrapping the launch in
    # a helper process, which would merge the stdout/stderr the protocol keeps on
    # separate pipes.
    env: Mapping[str, str] | None = None
    # Capture the Julia process's own stderr to a file so a startup crash (a
    # package-load failure, a bad project) can be replayed in the error message.
    stderr_file: Path | None = None
    # Seconds to wait for the control handshake before declaring a startup failure.
    startup_timeout: float = 180.0
