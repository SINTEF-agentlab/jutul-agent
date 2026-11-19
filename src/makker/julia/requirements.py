"""Julia toolchain requirement checks, shared across the CLI.

One place that knows "is Julia usable?" so the runtime launch, ``init``,
and ``doctor`` all agree on the same answer and the same remediation text.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass

# Floor set by the simulators, not the kernel (server.jl is stdlib-only): Mocca
# and JutulDarcy need 1.10. 1.10 is also the current Julia LTS.
MIN_JULIA_VERSION: tuple[int, int] = (1, 10)

_VERSION_RE = re.compile(r"julia version (\d+)\.(\d+)\.(\d+)")


@dataclass(frozen=True)
class JuliaCheck:
    """Result of probing the ``julia`` executable on PATH."""

    found: bool
    path: str | None = None
    version: tuple[int, int, int] | None = None
    error: str | None = None

    @property
    def version_str(self) -> str | None:
        if self.version is None:
            return None
        return ".".join(str(n) for n in self.version)

    @property
    def version_ok(self) -> bool:
        return self.version is not None and self.version[:2] >= MIN_JULIA_VERSION

    @property
    def ok(self) -> bool:
        return self.found and self.version_ok


def _min_version_str() -> str:
    return ".".join(str(n) for n in MIN_JULIA_VERSION)


def check_julia(executable: str = "julia") -> JuliaCheck:
    """Probe for ``julia`` on PATH and parse its version.

    Never raises — failures are reported in the returned ``JuliaCheck``.
    """

    path = shutil.which(executable)
    if path is None:
        return JuliaCheck(found=False, error=f"`{executable}` is not on PATH")

    try:
        proc = subprocess.run(
            [executable, "--version"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except OSError as exc:
        return JuliaCheck(found=True, path=path, error=f"could not run `{executable}`: {exc}")

    match = _VERSION_RE.search(proc.stdout or "")
    if match is None:
        return JuliaCheck(
            found=True,
            path=path,
            error=f"could not parse version from `{executable} --version`: {proc.stdout!r}",
        )

    version = (int(match[1]), int(match[2]), int(match[3]))
    return JuliaCheck(found=True, path=path, version=version)


def require_julia(executable: str = "julia") -> JuliaCheck:
    """Like :func:`check_julia` but raises ``JuliaRequirementError`` if unusable."""

    check = check_julia(executable)
    if not check.found:
        raise JuliaRequirementError(
            f"`{executable}` is not on PATH. Install Julia {_min_version_str()}+ via "
            "juliaup (https://github.com/JuliaLang/juliaup), then open a new terminal."
        )
    if not check.version_ok:
        raise JuliaRequirementError(
            f"Julia {_min_version_str()}+ is required, but `{executable}` is "
            f"{check.version_str or 'an unknown version'}. "
            f"Run `juliaup add {_min_version_str()} && juliaup default {_min_version_str()}`."
        )
    return check


class JuliaRequirementError(RuntimeError):
    pass
