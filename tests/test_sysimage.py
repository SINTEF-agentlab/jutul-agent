"""Tests for the system-image stamp and the launch guard.

Everything here runs against files on disk, with no Julia and no real image,
because that is exactly what the guard itself does: it decides from the manifest
and the sources of path-tracked packages, so the whole decision is testable
without paying for a build.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jutul_agent import display as sysimage_display
from jutul_agent import sysimage
from jutul_agent.sysimage import (
    CURRENT,
    DIVERGENT,
    INCOMPLETE,
    MISSING,
    OFF,
    UNUSABLE,
    Decision,
    decide,
    read_env_state,
    refusal,
    resolve_enabled,
    write_stamp,
)

JULIA = "1.12.4"


@pytest.fixture(autouse=True)
def _fixed_julia(monkeypatch):
    """Pin the Julia version so the guard never depends on the host's install."""
    monkeypatch.setattr(sysimage, "julia_version", lambda: JULIA)


def write_manifest(env: Path, *, versions: dict[str, str], paths: dict[str, str]) -> None:
    lines = ['julia_version = "1.12.4"', 'manifest_format = "2.0"', ""]
    for name, version in versions.items():
        lines += [f"[[deps.{name}]]", f'version = "{version}"', ""]
    for name, path in paths.items():
        # As Julia writes it: a TOML basic string, where a raw Windows backslash
        # would be an escape sequence. Forward slashes are valid on every OS.
        posix = path.replace("\\", "/")
        lines += [f"[[deps.{name}]]", f'path = "{posix}"', 'version = "0.1.0"', ""]
    env.mkdir(parents=True, exist_ok=True)
    (env / "Manifest.toml").write_text("\n".join(lines), encoding="utf-8")


def make_package(root: Path, body: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "Project.toml").write_text('name = "Demo"\n', encoding="utf-8")
    (root / "Demo.jl").write_text(body, encoding="utf-8")
    return root


def build_workspace(tmp_path: Path, *, source: str = "original") -> tuple[Path, Path]:
    """A workspace whose env resolves one registry package and one path package."""
    ws = tmp_path / "ws"
    env = ws / ".jutul-agent" / "julia-env"
    make_package(env / "JutulAgent", source)
    write_manifest(env, versions={"JutulDarcy": "0.2.44"}, paths={"JutulAgent": "JutulAgent"})
    return ws, env


def install_image(ws: Path) -> Path:
    path = sysimage.sysimage_path(ws)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not a real system image")
    return path


# ---------------------------------------------------------------------------
# Reading the environment.


def test_env_state_splits_registry_versions_from_path_sources(tmp_path: Path) -> None:
    _ws, env = build_workspace(tmp_path)
    state = read_env_state(env)
    assert state.versions == {"JutulDarcy": "0.2.44"}
    # A path package is hashed, not versioned: its source changes without a release.
    assert set(state.path_packages) == {"JutulAgent"}
    assert len(state.path_packages["JutulAgent"]) == 64


def test_env_state_resolves_a_relative_path_against_the_manifest(tmp_path: Path) -> None:
    _ws, env = build_workspace(tmp_path)
    relative = read_env_state(env).path_packages["JutulAgent"]
    # The same package named absolutely has to hash identically, or the guard
    # would fire on the two ways Julia writes the same dependency.
    write_manifest(env, versions={}, paths={"JutulAgent": str(env / "JutulAgent")})
    assert read_env_state(env).path_packages["JutulAgent"] == relative


def test_env_state_of_an_unresolved_env_is_empty(tmp_path: Path) -> None:
    assert read_env_state(tmp_path).versions == {}
    assert read_env_state(tmp_path).path_packages == {}


def test_a_broken_manifest_does_not_turn_into_a_stale_image(tmp_path: Path) -> None:
    """A broken env has louder problems than a stale image, so it must not block here.

    Preferences live in their own file and are still read, so the digest has to
    survive the manifest being unreadable rather than reading as absent.
    """
    ws, env = build_workspace(tmp_path)
    (env / "LocalPreferences.toml").write_text('[CondaPkg]\nbackend = "Null"\n', encoding="utf-8")
    install_image(ws)
    write_stamp(ws, env, cpu_target="native", build_seconds=1.0, julia=JULIA)

    (env / "Manifest.toml").write_text("{ not toml", encoding="utf-8")
    assert decide(ws, env, enabled=True).status == CURRENT


# ---------------------------------------------------------------------------
# The decision.


def test_off_ignores_an_image_that_is_there(tmp_path: Path) -> None:
    ws, env = build_workspace(tmp_path)
    install_image(ws)
    decision = decide(ws, env, enabled=False)
    assert decision.status == OFF
    assert decision.path is None
    assert not decision.blocks


def test_a_stamped_image_matching_the_env_is_used(tmp_path: Path) -> None:
    ws, env = build_workspace(tmp_path)
    image = install_image(ws)
    write_stamp(ws, env, cpu_target="native", build_seconds=1.0, julia=JULIA)
    decision = decide(ws, env, enabled=True)
    assert decision.status == CURRENT
    assert decision.path == image
    assert decision.notes == ()


def test_a_missing_image_blocks_rather_than_starting_slowly(tmp_path: Path) -> None:
    ws, env = build_workspace(tmp_path)
    decision = decide(ws, env, enabled=True)
    assert decision.status == MISSING
    assert decision.blocks


def test_an_image_without_a_readable_stamp_is_not_trusted(tmp_path: Path) -> None:
    ws, env = build_workspace(tmp_path)
    install_image(ws)
    sysimage.stamp_path(ws).write_text("{ not json", encoding="utf-8")
    decision = decide(ws, env, enabled=True)
    assert decision.status == UNUSABLE
    assert decision.blocks


def test_a_changed_package_version_diverges(tmp_path: Path) -> None:
    ws, env = build_workspace(tmp_path)
    install_image(ws)
    write_stamp(ws, env, cpu_target="native", build_seconds=1.0, julia=JULIA)
    write_manifest(env, versions={"JutulDarcy": "0.2.45"}, paths={"JutulAgent": "JutulAgent"})

    decision = decide(ws, env, enabled=True)
    assert decision.status == DIVERGENT
    assert decision.blocks
    assert decision.path is None
    assert "JutulDarcy 0.2.44 -> 0.2.45" in decision.reason


def test_an_edited_path_package_diverges(tmp_path: Path) -> None:
    """The case a version number cannot catch, and the one that fires daily."""
    ws, env = build_workspace(tmp_path)
    install_image(ws)
    write_stamp(ws, env, cpu_target="native", build_seconds=1.0, julia=JULIA)
    (env / "JutulAgent" / "Demo.jl").write_text("edited", encoding="utf-8")

    decision = decide(ws, env, enabled=True)
    assert decision.status == DIVERGENT
    assert "edited since the image was built" in decision.reason
    assert "JutulAgent" in decision.reason


def test_a_changed_preference_diverges(tmp_path: Path) -> None:
    """A preference read while precompiling is baked in as surely as the code is.

    Nothing in the manifest moves when one is edited, so without this the image
    would keep answering with the setting it was built with.
    """
    ws, env = build_workspace(tmp_path)
    (env / "LocalPreferences.toml").write_text('[CondaPkg]\nbackend = "Null"\n', encoding="utf-8")
    install_image(ws)
    write_stamp(ws, env, cpu_target="native", build_seconds=1.0, julia=JULIA)
    assert decide(ws, env, enabled=True).status == CURRENT

    (env / "LocalPreferences.toml").write_text(
        '[CondaPkg]\nbackend = "MicroMamba"\n', encoding="utf-8"
    )
    decision = decide(ws, env, enabled=True)
    assert decision.status == DIVERGENT
    assert "LocalPreferences.toml" in decision.reason


def test_rewriting_the_preferences_unchanged_does_not_diverge(tmp_path: Path) -> None:
    """Capability composition rewrites this file on every launch."""
    ws, env = build_workspace(tmp_path)
    (env / "LocalPreferences.toml").write_text(
        '# a comment\n[CondaPkg]\nbackend = "Null"\n', encoding="utf-8"
    )
    install_image(ws)
    write_stamp(ws, env, cpu_target="native", build_seconds=1.0, julia=JULIA)

    # Same settings, reserialised: no comment, different key order.
    (env / "LocalPreferences.toml").write_text('[CondaPkg]\nbackend = "Null"\n', encoding="utf-8")
    assert decide(ws, env, enabled=True).status == CURRENT


def test_an_env_with_no_preferences_file_is_not_a_change(tmp_path: Path) -> None:
    """Most environments never have one; absent on both sides has to agree."""
    ws, env = build_workspace(tmp_path)
    install_image(ws)
    write_stamp(ws, env, cpu_target="native", build_seconds=1.0, julia=JULIA)
    assert decide(ws, env, enabled=True).status == CURRENT

    # Gaining one is a change, though: it was not what the packages were built against.
    (env / "LocalPreferences.toml").write_text('[CondaPkg]\nbackend = "Null"\n', encoding="utf-8")
    assert decide(ws, env, enabled=True).status == DIVERGENT


def test_a_package_added_after_the_build_is_a_note_not_a_block(tmp_path: Path) -> None:
    ws, env = build_workspace(tmp_path)
    install_image(ws)
    write_stamp(ws, env, cpu_target="native", build_seconds=1.0, julia=JULIA)
    write_manifest(
        env,
        versions={"JutulDarcy": "0.2.44", "CSV": "0.10.15"},
        paths={"JutulAgent": "JutulAgent"},
    )

    decision = decide(ws, env, enabled=True)
    assert decision.status == INCOMPLETE
    assert decision.usable and not decision.blocks
    assert "CSV" in decision.notes[0]


def test_a_different_julia_diverges(tmp_path: Path, monkeypatch) -> None:
    ws, env = build_workspace(tmp_path)
    install_image(ws)
    write_stamp(ws, env, cpu_target="native", build_seconds=1.0, julia=JULIA)
    monkeypatch.setattr(sysimage, "julia_version", lambda: "1.13.0")

    decision = decide(ws, env, enabled=True)
    assert decision.status == DIVERGENT
    assert "1.13.0" in decision.reason


def test_a_different_platform_diverges(tmp_path: Path) -> None:
    ws, env = build_workspace(tmp_path)
    install_image(ws)
    write_stamp(ws, env, cpu_target="native", build_seconds=1.0, julia=JULIA)
    stamp = json.loads(sysimage.stamp_path(ws).read_text(encoding="utf-8"))
    # A platform no host answers to, so the test diverges on Linux and Windows alike.
    stamp["platform"] = "plan9-mips"
    sysimage.stamp_path(ws).write_text(json.dumps(stamp), encoding="utf-8")

    decision = decide(ws, env, enabled=True)
    assert decision.status == DIVERGENT
    assert "plan9-mips" in decision.reason


def test_an_older_build_recipe_diverges(tmp_path: Path) -> None:
    """An upgrade that changes how images are built retires the old ones."""
    ws, env = build_workspace(tmp_path)
    install_image(ws)
    write_stamp(ws, env, cpu_target="native", build_seconds=1.0, julia=JULIA)
    stamp = json.loads(sysimage.stamp_path(ws).read_text(encoding="utf-8"))
    stamp["recipe"] = sysimage.RECIPE_VERSION - 1
    sysimage.stamp_path(ws).write_text(json.dumps(stamp), encoding="utf-8")

    assert decide(ws, env, enabled=True).status == DIVERGENT


def test_jutul_agent_version_alone_does_not_diverge(tmp_path: Path) -> None:
    """A commit that leaves the Julia side alone must not invalidate the image.

    What has to agree with the image is the Julia jutul-agent ships, and that is
    already covered as a path package once it is copied into the env.
    """
    ws, env = build_workspace(tmp_path)
    install_image(ws)
    write_stamp(ws, env, cpu_target="native", build_seconds=1.0, julia=JULIA)
    stamp = json.loads(sysimage.stamp_path(ws).read_text(encoding="utf-8"))
    stamp["jutul_agent_version"] = "0.0.1.dev1+gdeadbee"
    sysimage.stamp_path(ws).write_text(json.dumps(stamp), encoding="utf-8")

    assert decide(ws, env, enabled=True).status == CURRENT


# ---------------------------------------------------------------------------
# What the user is told, and how they turned it on.


def test_the_refusal_names_the_cause_and_both_ways_out() -> None:
    decision = Decision(status=DIVERGENT, reason="  edited: CapabilityPackage")
    message = refusal(decision, command="jutul-agent tui")
    assert "CapabilityPackage" in message
    assert "jutul-agent sysimage build" in message
    assert "jutul-agent tui --no-sysimage" in message


def test_a_refusal_rebuilding_cannot_fix_says_so_instead() -> None:
    decision = Decision(status=UNUSABLE, reason="no display", fix=("Install Xvfb:", "    xvfb-run"))
    message = refusal(decision)
    assert "Install Xvfb" in message
    # Telling someone to spend 20 minutes rebuilding would not help them here.
    assert "jutul-agent sysimage build" not in message
    assert "--no-sysimage" in message


# ---------------------------------------------------------------------------
# A baked GLMakie needs a display before the process starts, not before it plots.


def _headless(monkeypatch, *, opted_out: bool = False) -> None:
    monkeypatch.setattr(sysimage_display, "has_display", lambda: False)
    monkeypatch.setattr(sysimage_display, "xvfb_run_available", lambda: not opted_out)
    if opted_out:
        monkeypatch.setenv(sysimage.XVFB_OPT_OUT, "1")
    else:
        monkeypatch.delenv(sysimage.XVFB_OPT_OUT, raising=False)
    monkeypatch.setattr(sysimage_display.platform, "system", lambda: "Linux")


def test_a_baked_glmakie_without_a_display_refuses_rather_than_aborting(
    tmp_path: Path, monkeypatch
) -> None:
    """Julia dies with 'no exception handler available' otherwise, which explains nothing."""
    ws, env = build_workspace(tmp_path)
    write_manifest(env, versions={"GLFW": "3.4.3"}, paths={"JutulAgent": "JutulAgent"})
    install_image(ws)
    write_stamp(ws, env, cpu_target="native", build_seconds=1.0, julia=JULIA)

    _headless(monkeypatch, opted_out=True)
    decision = decide(ws, env, enabled=True)
    assert decision.status == UNUSABLE
    assert decision.blocks
    assert "no display" in decision.reason
    # The fix is the display, never a rebuild: the image is perfectly good.
    assert any(sysimage.XVFB_OPT_OUT in line for line in decision.fix)


def test_an_image_without_opengl_in_it_is_fine_headless(tmp_path: Path, monkeypatch) -> None:
    """Nothing initialises a window system, so there is nothing to refuse."""
    ws, env = build_workspace(tmp_path)
    install_image(ws)
    write_stamp(ws, env, cpu_target="native", build_seconds=1.0, julia=JULIA)

    _headless(monkeypatch, opted_out=True)
    assert decide(ws, env, enabled=True).status == CURRENT


def test_a_virtual_display_is_enough(tmp_path: Path, monkeypatch) -> None:
    ws, env = build_workspace(tmp_path)
    write_manifest(env, versions={"GLFW": "3.4.3"}, paths={"JutulAgent": "JutulAgent"})
    install_image(ws)
    write_stamp(ws, env, cpu_target="native", build_seconds=1.0, julia=JULIA)

    _headless(monkeypatch, opted_out=False)  # headless, but xvfb-run is there
    assert decide(ws, env, enabled=True).status == CURRENT


def test_enabled_resolution_prefers_the_flag_then_the_workspace(monkeypatch) -> None:
    monkeypatch.delenv(sysimage.SYSIMAGE_ENV_VAR, raising=False)
    assert resolve_enabled(None, workspace_enabled=None) is False
    assert resolve_enabled(None, workspace_enabled=True) is True
    assert resolve_enabled(False, workspace_enabled=True) is False
    assert resolve_enabled(True, workspace_enabled=False) is True


def test_the_environment_is_the_last_word_before_the_default(monkeypatch) -> None:
    monkeypatch.setenv(sysimage.SYSIMAGE_ENV_VAR, "1")
    assert resolve_enabled(None, workspace_enabled=None) is True
    assert resolve_enabled(None, workspace_enabled=False) is False
    monkeypatch.setenv(sysimage.SYSIMAGE_ENV_VAR, "off")
    assert resolve_enabled(None, workspace_enabled=None) is False


def test_clear_removes_the_image_and_its_stamp(tmp_path: Path) -> None:
    ws, env = build_workspace(tmp_path)
    install_image(ws)
    write_stamp(ws, env, cpu_target="native", build_seconds=1.0, julia=JULIA)
    assert sysimage.clear(ws) is True
    assert not sysimage.sysimage_path(ws).exists()
    assert sysimage.clear(ws) is False
