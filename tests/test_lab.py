"""The headless TUI lab: every scenario renders, and capture writes artifacts."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from jutul_agent.lab.scenarios import all_scenarios, get
from jutul_agent.lab.tui import capture, normalize_screen, svg_to_text

# Plain-text TUI captures (regenerated via the lab); distinct from the
# syrupy .ambr snapshots under __snapshots__, which `--snapshot-update` owns.
_SNAPSHOTS = Path(__file__).parent / "tui_captures" / "tui"


def test_scenarios_are_registered():
    names = [s.name for s in all_scenarios()]
    assert {"welcome", "tool_call", "approval", "tool_error"} <= set(names)
    assert get("tool_call").build_agent is not None


def test_svg_to_text_recovers_rows():
    svg = (
        '<svg viewBox="0 0 100 100">'
        '<text x="0" y="20">hello &amp; </text><text x="40" y="20">world</text>'
        '<text x="0" y="40">second &lt;row&gt;</text></svg>'
    )
    assert svg_to_text(svg) == "hello & world\nsecond <row>\n"


def test_capture_writes_svg_text_and_transcript(tmp_path):
    paths = capture(get("tool_call"), tmp_path)
    svg = paths["svg"].read_text(encoding="utf-8")
    assert "<svg" in svg
    text = paths["txt"].read_text(encoding="utf-8")
    assert "jutul-agent" in text  # the header rendered
    assert "Vector" in text  # the tool output reached the screen
    assert paths["transcript"].read_text(encoding="utf-8").strip()


@pytest.mark.parametrize("scenario", all_scenarios(), ids=lambda s: s.name)
def test_scenario_renders_and_matches_snapshot(scenario, tmp_path):
    """Every scenario renders without crashing and matches its committed snapshot.

    A UI change shows up here as a diff. Intentional? Regenerate the snapshots with
    ``JUTUL_LAB_UPDATE=1 pytest tests/test_lab.py``.
    """
    paths = capture(scenario, tmp_path)
    assert paths["svg"].stat().st_size > 0
    screen = normalize_screen(paths["txt"].read_text(encoding="utf-8"))
    assert "jutul-agent" in screen

    golden = _SNAPSHOTS / f"{scenario.name}.txt"
    if os.environ.get("JUTUL_LAB_UPDATE"):
        golden.parent.mkdir(parents=True, exist_ok=True)
        golden.write_text(screen, encoding="utf-8")
        return
    assert golden.exists(), (
        f"no snapshot for {scenario.name}; create it with JUTUL_LAB_UPDATE=1 pytest"
    )
    assert screen == golden.read_text(encoding="utf-8"), (
        f"{scenario.name} render changed; review the diff or regenerate with "
        "JUTUL_LAB_UPDATE=1 pytest"
    )


def test_parse_importtime_rolls_up_by_package():
    from jutul_agent.lab.profile_startup import parse_importtime

    stderr = (
        "import time: self [us] | cumulative | imported package\n"
        "import time:       500 |        500 |   langsmith.schemas\n"
        "import time:       200 |        700 |   langsmith\n"
        "import time:       100 |        100 |   textual.app\n"
    )
    prof = parse_importtime(stderr, module="x")
    by_pkg = dict(prof.by_package)
    assert by_pkg["langsmith"] == pytest.approx(0.0007)  # 500us + 200us
    assert by_pkg["textual"] == pytest.approx(0.0001)
    assert prof.slowest[0] == ("langsmith.schemas", pytest.approx(0.0005))


def test_measure_phases_returns_timings():
    from jutul_agent.lab.profile_startup import measure_phases

    phases = measure_phases(repeat=1)
    assert set(phases) == {"session_create", "build_agent"}
    assert all(v >= 0 for v in phases.values())


def test_profile_turn_surfaces_our_functions():
    from jutul_agent.lab.profile_turn import profile_turn

    rows = profile_turn("answer", repeat=1)
    assert rows, "the turn profiler should find some jutul-agent functions"
    assert all(r.tottime >= 0 and r.calls > 0 for r in rows)
    # It reports our code, not the lab itself.
    assert all(not r.function.startswith("lab/") for r in rows)
