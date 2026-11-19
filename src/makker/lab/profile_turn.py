"""Profile a representative agent turn to surface slow makker code.

The import profiler (:mod:`makker.lab.profile_startup`) covers cold start.
This covers the hot path: it runs a scenario through the full agent runtime and TUI
with the fakes (no model or Julia latency), so the time that shows up is our own
Python, the code worth optimising: rendering, middleware, the trace, the tools.

    python -m makker.lab.profile_turn
    python -m makker.lab.profile_turn --scenario long_output --repeat 5
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Hotspot:
    function: str
    calls: int
    tottime: float  # seconds in the function itself
    cumtime: float  # seconds including callees


def profile_turn(scenario_name: str = "tool_call", repeat: int = 3) -> list[Hotspot]:
    """Profile rendering a scenario and return the makker functions by self-time."""
    from makker.lab.scenarios import get
    from makker.lab.tui import capture

    scenario = get(scenario_name)
    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(repeat):
        capture(scenario, Path(tempfile.mkdtemp(prefix="jutul-profturn-")))
    profiler.disable()

    stats = pstats.Stats(profiler)
    rows: list[Hotspot] = []
    for (filename, _line, func), (calls, _nc, tottime, cumtime, _callers) in stats.stats.items():
        norm = filename.replace("\\", "/")
        if "makker" not in norm or "/lab/" in norm:
            continue
        name = norm.split("makker")[-1].lstrip("/")
        rows.append(Hotspot(f"{name}:{func}", calls, tottime, cumtime))
    rows.sort(key=lambda r: r.tottime, reverse=True)
    return rows[:20]


def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m makker.lab.profile_turn")
    parser.add_argument("--scenario", default="tool_call")
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args(argv)

    rows = profile_turn(args.scenario, args.repeat)
    print(f"Hottest makker functions rendering `{args.scenario}` x{args.repeat}:\n")
    print(f"  {'self ms':>8} {'cum ms':>8} {'calls':>7}  function")
    for r in rows:
        print(f"  {r.tottime * 1000:8.1f} {r.cumtime * 1000:8.1f} {r.calls:7d}  {r.function}")
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
