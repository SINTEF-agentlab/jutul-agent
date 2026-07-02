"""Tests for session-prompt assembly."""

from __future__ import annotations

from pathlib import Path

from jutul_agent.agent.prompts import assemble_session_prompt
from jutul_agent.simulators.base import SimulatorAdapter


def _adapter(tmp_path: Path) -> SimulatorAdapter:
    return SimulatorAdapter(
        name="test",
        display_name="Test",
        module_dir=tmp_path,
        package_imports=("Foo",),
        primary_package="Foo",
        domain_hints="",
    )


def test_prompt_is_native_first(tmp_path: Path) -> None:
    p = assemble_session_prompt(_adapter(tmp_path))
    assert "native plotters" in p
    assert "view=true" in p
    # The old return-a-Figure contract is gone.
    assert "must return a Makie" not in p
    assert "Code must return" not in p
    # The general prompt stays simulator-agnostic: no hardcoded plotter names.
    assert "plot_reservoir" not in p
    assert "plot_well_results" not in p


def test_prompt_describes_glmakie_window_behavior(tmp_path: Path) -> None:
    p = assemble_session_prompt(_adapter(tmp_path))
    assert "GLMakie" in p
    assert "live window" in p


def test_prompt_states_window_availability_per_session(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    windowed = assemble_session_prompt(adapter, open_windows=True)
    headless = assemble_session_prompt(adapter, open_windows=False)
    # Windowed: promises an interactive window, no caveat.
    assert "live plot windows are available" in windowed
    assert "HEADLESS" not in windowed
    # Headless: explicit caveat so the agent won't claim a window opened.
    assert "HEADLESS" in headless
    assert "Never tell the user a window opened" in headless


def test_prompt_states_working_directory_when_given(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    with_ws = assemble_session_prompt(adapter, workspace=tmp_path)
    without = assemble_session_prompt(adapter)
    # Given a workspace, the absolute working directory is stated up front so
    # the agent builds correct paths without a pwd/ls round-trip.
    assert f"Working directory: {tmp_path}" in with_ws
    # Omitted when not given (the RunConfig hash path) so a per-session path
    # does not destabilize the attribution hash.
    assert "Working directory" not in without


def test_session_prompt_resume_note(tmp_path) -> None:
    from jutul_agent.agent.prompts import assemble_session_prompt
    from jutul_agent.lab.fakes import make_fake_adapter

    adapter = make_fake_adapter(tmp_path)
    fresh = assemble_session_prompt(adapter)
    resumed = assemble_session_prompt(adapter, resumed=True)
    assert "resumed" not in fresh
    assert "Session continuity" in resumed
    assert "Julia REPL restarted" in resumed
