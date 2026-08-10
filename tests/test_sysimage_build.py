"""Tests for the system-image builder.

Julia is the one thing stubbed out: a real build costs tens of minutes, and none
of what can go wrong here is inside PackageCompiler. What matters is everything
around it, above all that an image is only ever installed after it has been shown
to work, and that a build no one verified leaves the workspace exactly as it was.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from jutul_agent import sysimage, sysimage_build
from jutul_agent.sysimage_build import (
    SysimageBuildError,
    _verify_script,
    baked_packages,
    build,
    describe,
)

VERIFIED = "sysimage-verify: ok\n"


def write_project(env: Path, deps: dict[str, str]) -> Path:
    env.mkdir(parents=True, exist_ok=True)
    lines = ["[deps]"] + [f'{name} = "{uuid}"' for name, uuid in deps.items()]
    (env / "Project.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (env / "Manifest.toml").write_text(
        'julia_version = "1.12.4"\nmanifest_format = "2.0"\n\n'
        + "\n".join(f'[[deps.{name}]]\nversion = "1.0.0"\n' for name in deps),
        encoding="utf-8",
    )
    return env


class _FakeJulia:
    """Stands in for every Julia subprocess the builder runs.

    Records the argv it was handed, creates the image file when asked to build,
    and answers verification however the test wants it answered.
    """

    def __init__(self, *, verify_ok: bool = True, build_ok: bool = True) -> None:
        self.verify_ok = verify_ok
        self.build_ok = build_ok
        self.calls: list[list[str]] = []

    def __call__(self, argv, *, capture: bool = False):
        self.calls.append(list(argv))
        script = argv[-1]
        if "create_sysimage" in script:
            if self.build_ok:
                Path(_quoted_path(script, "sysimage_path")).write_bytes(b"image")
            return subprocess.CompletedProcess(argv, 0 if self.build_ok else 1, "", "")
        if "sysimage-verify" in script:
            return subprocess.CompletedProcess(
                argv, 0 if self.verify_ok else 1, VERIFIED if self.verify_ok else "", "boom"
            )
        return subprocess.CompletedProcess(argv, 0, "", "")

    @property
    def verified(self) -> bool:
        return any("sysimage-verify" in call[-1] for call in self.calls)


def _quoted_path(script: str, key: str) -> str:
    after = script.split(f"{key} = raw", 1)[1]
    return after.split('"')[1]


@pytest.fixture
def julia(monkeypatch: pytest.MonkeyPatch) -> _FakeJulia:
    fake = _FakeJulia()
    monkeypatch.setattr(sysimage_build, "_run_julia", fake)
    monkeypatch.setattr(sysimage, "julia_version", lambda: "1.12.4")
    return fake


# ---------------------------------------------------------------------------
# What goes into the image.


def test_every_direct_dependency_is_baked(tmp_path: Path) -> None:
    env = write_project(tmp_path / "env", {"JutulDarcy": "a", "GLMakie": "b"})
    assert baked_packages(env) == ("GLMakie", "JutulDarcy")


def test_a_project_with_nothing_in_it_is_refused(tmp_path: Path, julia: _FakeJulia) -> None:
    env = write_project(tmp_path / "env", {})
    with pytest.raises(SysimageBuildError, match="no dependencies"):
        build(workspace=tmp_path / "ws", julia_project=env)
    assert julia.calls == []


# ---------------------------------------------------------------------------
# Verification, and what it protects.


def test_verification_looks_for_the_packages_before_loading_them() -> None:
    script = _verify_script(("JutulDarcy", "JutulAgent"), deps=set())
    # Bound into Main at startup is the only evidence that the bake took; a
    # `using` would succeed either way and prove nothing.
    assert "isdefined(Main, Symbol(p))" in script
    assert '"JutulDarcy"' in script and '"JutulAgent"' in script


def test_verification_covers_plotting_only_where_the_env_can(tmp_path: Path) -> None:
    bare = _verify_script(("JutulDarcy",), deps={"JutulDarcy"})
    assert "GLMakie" not in bare and "Bonito" not in bare

    full = _verify_script(("JutulDarcy",), deps={"GLMakie", "WGLMakie", "Bonito"})
    assert "GLMakie.activate!" in full
    assert "Bonito.export_static" in full


def test_a_verified_image_is_installed_and_stamped(tmp_path: Path, julia: _FakeJulia) -> None:
    ws = tmp_path / "ws"
    env = write_project(tmp_path / "env", {"JutulDarcy": "a"})

    result = build(workspace=ws, julia_project=env)

    assert julia.verified
    assert result.path == sysimage.sysimage_path(ws)
    assert result.path.exists()
    # The end the whole feature is for: the guard now starts sessions from it.
    assert sysimage.decide(ws, env, enabled=True).status == sysimage.CURRENT


def test_an_image_that_fails_verification_is_never_installed(
    tmp_path: Path, julia: _FakeJulia
) -> None:
    ws = tmp_path / "ws"
    env = write_project(tmp_path / "env", {"JutulDarcy": "a"})
    julia.verify_ok = False

    with pytest.raises(SysimageBuildError, match="failed verification"):
        build(workspace=ws, julia_project=env)

    assert not sysimage.sysimage_path(ws).exists()
    assert sysimage.read_stamp(ws) is None
    # And nothing half-built is left behind for the next build to trip over.
    assert list(sysimage.sysimage_dir(ws).glob("candidate-*")) == []


def test_a_failed_build_leaves_the_previous_image_alone(tmp_path: Path, julia: _FakeJulia) -> None:
    """The reason a rebuild is safe to attempt on a machine that is about to demo."""
    ws = tmp_path / "ws"
    env = write_project(tmp_path / "env", {"JutulDarcy": "a"})
    build(workspace=ws, julia_project=env)
    good = sysimage.sysimage_path(ws).read_bytes()

    julia.verify_ok = False
    with pytest.raises(SysimageBuildError):
        build(workspace=ws, julia_project=env)

    assert sysimage.sysimage_path(ws).read_bytes() == good
    assert sysimage.decide(ws, env, enabled=True).status == sysimage.CURRENT


def test_an_image_held_open_fails_cleanly_rather_than_with_a_traceback(
    tmp_path: Path, julia: _FakeJulia, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Windows locks a loaded image against replacement, so a rebuild under a
    running session has to come back as advice, not a PermissionError."""
    ws = tmp_path / "ws"
    env = write_project(tmp_path / "env", {"JutulDarcy": "a"})

    def locked(self: Path, target: Path) -> Path:
        raise PermissionError(13, "the file is in use by another process")

    monkeypatch.setattr(Path, "replace", locked)
    with pytest.raises(SysimageBuildError, match="running session"):
        build(workspace=ws, julia_project=env)

    assert sysimage.read_stamp(ws) is None
    assert list(sysimage.sysimage_dir(ws).glob("candidate-*")) == []


def test_the_cpu_target_reaches_both_the_build_and_the_stamp(
    tmp_path: Path, julia: _FakeJulia
) -> None:
    ws = tmp_path / "ws"
    env = write_project(tmp_path / "env", {"JutulDarcy": "a"})

    build(workspace=ws, julia_project=env, cpu_target="generic")

    create = next(call for call in julia.calls if "create_sysimage" in call[-1])
    assert 'cpu_target = "generic"' in create[-1]
    assert (sysimage.read_stamp(ws) or {})["cpu_target"] == "generic"


def test_packagecompiler_never_enters_the_workspace_environment(
    tmp_path: Path, julia: _FakeJulia
) -> None:
    """It would otherwise show up in the manifest the image is checked against."""
    ws = tmp_path / "ws"
    env = write_project(tmp_path / "env", {"JutulDarcy": "a"})

    build(workspace=ws, julia_project=env)

    create = next(call for call in julia.calls if "create_sysimage" in call[-1])
    project_flag = next(arg for arg in create if arg.startswith("--project="))
    assert Path(project_flag.removeprefix("--project=")) == sysimage_build.builder_env()
    assert "PackageCompiler" not in (env / "Project.toml").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Reporting.


def test_the_command_turns_the_folder_on_after_a_build(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, julia: _FakeJulia
) -> None:
    """Building one and then not using it is the failure the design exists to stop."""
    from jutul_agent.interfaces.cli import sysimage as cmd
    from jutul_agent.workspace import WorkspaceConfig, load_workspace_config, write_workspace_config

    write_project(workspace / ".jutul-agent" / "julia-env", {"JutulDarcy": "a"})
    write_workspace_config(WorkspaceConfig(simulator="jutuldarcy"), workspace=workspace)
    monkeypatch.setattr(cmd, "prepare_environment", lambda *a, **k: None)
    monkeypatch.setattr("jutul_agent.julia.requirements.require_julia", lambda *a, **k: None)

    args = cmd.build_parser().parse_args(["build"])
    assert cmd.run(args) == 0
    assert load_workspace_config(workspace).sysimage is True


def test_the_command_turns_the_folder_off_when_the_image_goes(
    workspace: Path, julia: _FakeJulia
) -> None:
    """Or the next launch refuses over an image the user just deleted on purpose."""
    from jutul_agent.interfaces.cli import sysimage as cmd
    from jutul_agent.workspace import WorkspaceConfig, load_workspace_config, write_workspace_config

    write_project(workspace / ".jutul-agent" / "julia-env", {"JutulDarcy": "a"})
    write_workspace_config(
        WorkspaceConfig(simulator="jutuldarcy", sysimage=True), workspace=workspace
    )

    assert cmd.run(cmd.build_parser().parse_args(["clear"])) == 0
    assert load_workspace_config(workspace).sysimage is False


def test_the_build_and_the_status_agree_on_how_big_the_image_is(
    tmp_path: Path, julia: _FakeJulia
) -> None:
    """One image, one number. They come from different places and used to differ."""

    ws = tmp_path / "ws"
    env = write_project(tmp_path / "env", {"JutulDarcy": "a", "GLMakie": "b"})

    result = build(workspace=ws, julia_project=env)

    assert f"Contains:   {result.contained} packages" in describe(ws, env)
    # And it is the whole closure, not just what was asked for by name.
    assert result.contained >= len(result.packages)


def test_status_says_what_to_do_when_there_is_no_image(tmp_path: Path) -> None:
    env = write_project(tmp_path / "env", {"JutulDarcy": "a"})
    assert "jutul-agent sysimage build" in describe(tmp_path / "ws", env)


def test_status_names_what_moved_once_the_env_changes(tmp_path: Path, julia: _FakeJulia) -> None:
    ws = tmp_path / "ws"
    env = write_project(tmp_path / "env", {"JutulDarcy": "a"})
    build(workspace=ws, julia_project=env)
    assert "Up to date" in describe(ws, env)

    (env / "Manifest.toml").write_text(
        'julia_version = "1.12.4"\nmanifest_format = "2.0"\n\n'
        '[[deps.JutulDarcy]]\nversion = "2.0.0"\n',
        encoding="utf-8",
    )
    out = describe(ws, env)
    assert "Out of date" in out
    assert "JutulDarcy 1.0.0 -> 2.0.0" in out
