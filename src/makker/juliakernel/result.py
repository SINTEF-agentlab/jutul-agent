"""Value types crossing the kernel boundary."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class EvalResult:
    """Outcome of a single Julia evaluation.

    ``output`` is the REPL-style text a caller renders (cleaned stdout, then the
    value's repr; stderr appended under a ``[stderr]`` heading when present).
    ``error`` is set iff the eval threw, and is authoritative (taken from the
    control channel, never sniffed from stdout). The remaining fields expose the
    structured pieces for callers that want them.
    """

    output: str
    error: str | None = None
    value_repr: str | None = None
    stdout: str = ""
    stderr: str = ""
    interrupted: bool = False


@dataclass(frozen=True)
class OutputChunk:
    """A live fragment of an eval's output, delivered as it is produced."""

    text: str
    stream: str  # "stdout" | "stderr"


# A live-output sink: called with each fragment as the eval produces it.
OnChunk = Callable[[OutputChunk], None]
