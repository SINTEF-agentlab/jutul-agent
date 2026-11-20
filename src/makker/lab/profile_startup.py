"""Profile what makes Makker slow to start, so an agent can speed it up.

Cold start is dominated by Python imports (textual, langchain, langgraph,
deepagents, pydantic) and then by building the agent graph. This measures both:

- ``measure_import`` runs a fresh interpreter under ``-X importtime`` and reports
  the heaviest imports and the roll-up per top-level package.
- ``measure_phases`` times, in process, building a session and the agent graph
  (with the lab fakes, so no Julia or network).

Run it before and after a change to see the effect:

    python -m makker.lab.profile_startup
    python -m makker.lab.profile_startup --module makker.interfaces.tui
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from statistics import median

_IMPORTTIME_RE = re.compile(r"import time:\s+(\d+) \|\s+(\d+) \|\s+(.*)")


@dataclass
class ImportProfile:
    module: str
    total_seconds: float
    slowest: list[tuple[str, float]] = field(default_factory=list)  # (name, self_seconds)
    by_package: list[tuple[str, float]] = field(default_factory=list)


def parse_importtime(stderr: str, module: str = "") -> ImportProfile:
    """Tally ``-X importtime`` self-costs by module and by top-level package."""
    self_by_name: dict[str, float] = {}
    by_package: dict[str, float] = {}
    for line in stderr.splitlines():
        match = _IMPORTTIME_RE.match(line.strip())
        if not match:
            continue
        seconds = int(match.group(1)) / 1e6
        name = match.group(3).strip()
        self_by_name[name] = self_by_name.get(name, 0.0) + seconds
        top = name.split(".")[0]
        by_package[top] = by_package.get(top, 0.0) + seconds
    return ImportProfile(
        module=module,
        total_seconds=sum(self_by_name.values()),
        slowest=sorted(self_by_name.items(), key=lambda kv: kv[1], reverse=True)[:15],
        by_package=sorted(by_package.items(), key=lambda kv: kv[1], reverse=True)[:12],
    )


def measure_import(module: str) -> ImportProfile:
    """Import ``module`` in a fresh interpreter and tally importtime self-costs."""
    proc = subprocess.run(
        [sys.executable, "-X", "importtime", "-c", f"import {module}"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return parse_importtime(proc.stderr, module)


def measure_phases(repeat: int = 3) -> dict[str, float]:
    """Median seconds for the post-import runtime phases of bringing up a session."""
    import logging
    import tempfile
    from pathlib import Path

    from langchain_core.messages import AIMessage

    from makker.agent.builder import build_agent
    from makker.lab.fakes import (
        FakeJulia,
        ScriptedChatModel,
        make_fake_adapter,
    )
    from makker.paths import set_workspace_root
    from makker.session import Session

    # The scripted model isn't a known provider, so the profile lookup warns; that
    # noise is irrelevant to timing.
    logging.disable(logging.WARNING)
    create_times: list[float] = []
    build_times: list[float] = []
    for _ in range(repeat):
        tmp = Path(tempfile.mkdtemp(prefix="makker-profile-"))
        set_workspace_root(tmp)
        t0 = time.perf_counter()
        session = Session.create(
            julia=FakeJulia(), state_root=tmp, simulator=make_fake_adapter(tmp)
        )
        t1 = time.perf_counter()
        build_agent(
            session,
            model=ScriptedChatModel(responses=[AIMessage(content="")]),
            approval_mode="auto",
        )
        t2 = time.perf_counter()
        create_times.append(t1 - t0)
        build_times.append(t2 - t1)
    logging.disable(logging.NOTSET)
    return {"session_create": median(create_times), "build_agent": median(build_times)}


def _report(module: str) -> str:
    imp = measure_import(module)
    lines = [
        f"Cold-start profile for `import {module}`",
        "",
        f"Total import self-time: {imp.total_seconds * 1000:.0f} ms",
        "",
        "Heaviest packages (summed self-time):",
    ]
    lines += [f"  {seconds * 1000:7.1f} ms  {name}" for name, seconds in imp.by_package]
    lines += ["", "Heaviest individual modules:"]
    lines += [f"  {seconds * 1000:7.1f} ms  {name}" for name, seconds in imp.slowest]

    phases = measure_phases()
    lines += ["", "Runtime phases (median, after imports):"]
    lines += [f"  {seconds * 1000:7.1f} ms  {name}" for name, seconds in phases.items()]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m makker.lab.profile_startup")
    parser.add_argument(
        "--module",
        default="makker.interfaces.cli.run",
        help="Module to import for the cold-start measurement.",
    )
    args = parser.parse_args(argv)
    print(_report(args.module))
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
