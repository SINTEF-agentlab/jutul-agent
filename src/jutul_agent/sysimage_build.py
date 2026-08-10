"""Building a workspace's Julia system image.

Separate from :mod:`jutul_agent.sysimage`, which every launch imports to decide
whether the image on disk can be used. This half runs a handful of times in a
workspace's life and costs tens of minutes, so it stays out of the launch path.

The shape of a build:

1. ``PackageCompiler`` runs from an environment of its own under the state root,
   pointed at the workspace environment through ``project=``. It never becomes a
   dependency of the workspace, whose manifest is what the stamp describes.
2. Every direct dependency of the project is baked, which is ``create_sysimage``'s
   own default, so there is no package selection to get wrong.
3. The image is built to a temporary name, then **verified before it is
   installed**: a system image that fails to start would otherwise break every
   surface at once, with no way in to fix it.
4. Only then is it moved into place, and only then stamped. An image is never
   trusted before it has been shown to work, and the stamp is what confers trust.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path

from jutul_agent.sysimage import (
    image_suffix,
    sysimage_dir,
    sysimage_path,
    write_stamp,
)

BUILDER_ENV_DIRNAME = "sysimage-build"

# Rough, and only ever used in a sentence telling the user what they are in for.
TYPICAL_BUILD_MINUTES = 20


class SysimageBuildError(RuntimeError):
    """A build (or the verification that follows it) did not produce a usable image."""


@dataclass(frozen=True)
class BuildResult:
    path: Path
    seconds: float
    # The direct dependencies the build was asked for, which is also what
    # verification looks for in ``Main``. The image holds far more than these:
    # ``create_sysimage`` bakes everything they pull in. ``contained`` is that
    # larger number, so a report can say which it means.
    packages: tuple[str, ...]
    verified: bool
    contained: int = 0


def builder_env() -> Path:
    """Where ``PackageCompiler`` itself lives: once per install, not per workspace."""

    from jutul_agent.paths import state_home

    return state_home() / BUILDER_ENV_DIRNAME


def ensure_builder_env() -> Path:
    """Create (or top up) the environment the build runs from.

    Kept apart from the workspace environment on purpose. Adding
    ``PackageCompiler`` to the workspace would change the manifest the stamp
    describes, so every image would be born describing an environment slightly
    different from the one the session uses.
    """

    env = builder_env()
    env.mkdir(parents=True, exist_ok=True)
    project = env / "Project.toml"
    if not project.exists():
        project.write_text("[deps]\n", encoding="utf-8")
    if "PackageCompiler" in _project_deps(env):
        _julia(["using Pkg", "Pkg.instantiate()"], project=env, what="preparing the build tools")
    else:
        _julia(
            ["using Pkg", 'Pkg.add("PackageCompiler")'],
            project=env,
            what="installing PackageCompiler",
        )
    return env


def baked_packages(julia_project: Path) -> tuple[str, ...]:
    """The packages an image built from this project will contain.

    The same set ``create_sysimage`` picks by default (every direct dependency),
    read here so the build can say what it is about to do and how long a list it
    is working through.
    """

    return tuple(sorted(_project_deps(julia_project)))


def _project_deps(julia_project: Path) -> set[str]:
    try:
        data = tomllib.loads((julia_project / "Project.toml").read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return set()
    deps = data.get("deps")
    return set(deps) if isinstance(deps, dict) else set()


def build(
    *,
    workspace: Path,
    julia_project: Path,
    cpu_target: str = "native",
    verify: bool = True,
) -> BuildResult:
    """Build, verify and install this workspace's system image.

    ``cpu_target`` defaults to ``"native"``: the fastest code for the machine
    doing the build, and unusable anywhere else. Passing a portable target trades
    some speed for an image that runs on other hardware.
    """

    packages = baked_packages(julia_project)
    if not packages:
        raise SysimageBuildError(
            f"{julia_project} has no dependencies to build an image from. "
            "Run `jutul-agent init --sim <name>` first."
        )

    env = ensure_builder_env()
    destination = sysimage_path(workspace)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Built beside its destination so the final move is a rename within one
    # filesystem, and so a half-built image is never at the path the guard reads.
    candidate = destination.with_name(f"candidate-{os.getpid()}{image_suffix()}")

    started = time.monotonic()
    try:
        _julia(
            [
                "using PackageCompiler",
                "create_sysimage(;"
                f' sysimage_path = raw"{candidate.as_posix()}",'
                f' project = raw"{julia_project.as_posix()}",'
                f' cpu_target = "{cpu_target}",'
                " incremental = true, filter_stdlibs = false)",
            ],
            project=env,
            what="building the system image",
        )
        if not candidate.exists():
            raise SysimageBuildError("the build reported success but produced no image")
        if verify:
            verify_image(candidate, julia_project, packages)
        # Replace is atomic on both POSIX and Windows, so a session starting
        # during a rebuild sees either the old image or the new one. Windows
        # locks a loaded image against replacement (POSIX unlinks it and lets
        # running sessions keep their mapping), hence the translation.
        try:
            candidate.replace(destination)
        except OSError as exc:
            raise SysimageBuildError(
                f"the new image was built and verified but could not replace {destination}: "
                f"{exc}. A running session may be holding the old one open; close "
                "jutul-agent sessions in this folder and rebuild."
            ) from exc
    finally:
        candidate.unlink(missing_ok=True)

    seconds = time.monotonic() - started
    # Stamped last: the stamp is what promotes a file on disk to an image the
    # guard will start from, so an interrupted build leaves nothing to trust.
    write_stamp(workspace, julia_project, cpu_target=cpu_target, build_seconds=seconds)
    _precompile_against(destination, julia_project)

    from jutul_agent.sysimage import read_env_state

    state = read_env_state(julia_project)
    return BuildResult(
        path=destination,
        seconds=seconds,
        packages=packages,
        verified=verify,
        contained=len(state.versions) + len(state.path_packages),
    )


def _precompile_against(image: Path, julia_project: Path) -> None:
    """Refresh the cache for anything left outside the image. Best-effort.

    A pkgimage records which system image it was built against and is rebuilt
    when that changes, so without this the first session after a build pays a
    compile it looks like the image should have removed. With everything baked
    there is usually nothing to do, which is why a failure here is only worth a
    note: the image is already installed and already verified.
    """

    result = _run_julia(
        [
            "julia",
            f"--sysimage={image}",
            f"--project={julia_project}",
            "--startup-file=no",
            "-e",
            "using Pkg; Pkg.precompile()",
        ],
        capture=True,
    )
    if result.returncode != 0:
        print("note: could not precompile against the new image; the first session may be slower.")


def verify_image(image: Path, julia_project: Path, packages: tuple[str, ...]) -> None:
    """Start Julia on ``image`` and walk the paths a session actually uses.

    Three things, in the order they would bite:

    - the packages really are baked in, checked by looking for them in ``Main``
      before anything is loaded, since that is where ``PackageCompiler`` binds
      them. An image that built cleanly but contains nothing is otherwise
      indistinguishable from a fast one.
    - GLMakie renders, which needs the GL context an image cannot bake.
    - Bonito writes its static assets, because those resolve through
      ``RelocatableFolders``, which is exactly the kind of build-time path a
      system image can freeze at the wrong value.

    A failure here means the image is discarded, not installed and warned about.
    """

    script = _verify_script(packages, deps=_project_deps(julia_project))
    result = _run_julia(
        [
            "julia",
            f"--sysimage={image}",
            f"--project={julia_project}",
            "--startup-file=no",
            "-e",
            script,
        ],
        capture=True,
    )
    if result.returncode != 0 or "sysimage-verify: ok" not in (result.stdout or ""):
        tail = "\n".join(((result.stdout or "") + (result.stderr or "")).strip().splitlines()[-15:])
        raise SysimageBuildError(
            "the system image was built but failed verification, so it was "
            f"discarded and the previous one (if any) is untouched.\nJulia said:\n{tail}"
        )


def _verify_script(packages: tuple[str, ...], *, deps: set[str]) -> str:
    """Julia for :func:`verify_image`. Steps the environment cannot support are skipped."""

    # Checked before any `using`, which is the whole point: PackageCompiler binds
    # every baked package into Main at startup, so from an image these are
    # already there and from a plain start none of them would be.
    lines = [
        "let absent = [p for p in "
        + _julia_string_vector(list(packages))
        + " if !isdefined(Main, Symbol(p))]",
        '    isempty(absent) || error("not baked into the image: " * join(absent, ", "))',
        "end",
    ]
    if "GLMakie" in deps:
        lines += [
            "using GLMakie",
            "GLMakie.activate!(visible = false)",
            "let fig = Figure(size = (96, 96))",
            "    ax = Axis3(fig[1, 1])",
            "    surface!(ax, 1:4, 1:4, [Float64(i + j) for i in 1:4, j in 1:4])",
            '    png = joinpath(mktempdir(), "verify.png")',
            "    save(png, fig)",
            '    filesize(png) > 0 || error("GLMakie saved an empty figure")',
            "end",
        ]
    if {"WGLMakie", "Bonito"} <= deps:
        # A figure of its own rather than the GLMakie one above: a figure carries
        # the screens of every backend that has drawn it, and reusing one across
        # backends is its own source of trouble. Names are qualified so the block
        # stands on its own in an environment that has the web backend and no GLMakie.
        lines += [
            "import WGLMakie, Bonito",
            "WGLMakie.activate!()",
            "let fig = WGLMakie.Figure(size = (96, 96))",
            "    WGLMakie.lines!(WGLMakie.Axis(fig[1, 1]), 1:4, [1.0, 2.0, 3.0, 4.0])",
            '    html = joinpath(mktempdir(), "verify.html")',
            "    Bonito.export_static(html, Bonito.App(() -> Bonito.DOM.div(fig)))",
            '    filesize(html) > 0 || error("Bonito exported an empty page")',
            "end",
        ]
    lines.append('println("sysimage-verify: ok")')
    return "\n".join(lines)


def _julia_string_vector(names: list[str]) -> str:
    return "[" + ", ".join(f'"{name}"' for name in names) + "]"


def _julia(cmds: list[str], *, project: Path, what: str) -> None:
    """Run Julia in ``project``, streaming its output; raise with context on failure.

    Streamed rather than captured because these are the long steps: a build with
    no output for twenty minutes reads as a hang.
    """

    argv = ["julia", f"--project={project}", "--startup-file=no", "-e", "\n".join(cmds)]
    result = _run_julia(argv)
    if result.returncode != 0:
        raise SysimageBuildError(f"{what} failed (Julia exited with {result.returncode})")


def _run_julia(argv: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    if shutil.which("julia") is None:
        raise SysimageBuildError("`julia` is not on PATH")

    # The same wrap the runtime and the env bootstrap use: the Makie workloads
    # that run while building and verifying want a GL context, and headless Linux
    # has none without it.
    from jutul_agent.display import should_wrap_xvfb

    if should_wrap_xvfb():
        argv = ["xvfb-run", "-a", "-s", "-screen 0 1280x1024x24", *argv]

    try:
        return subprocess.run(argv, capture_output=capture, text=True, check=False)
    except OSError as exc:
        raise SysimageBuildError(f"could not run julia: {exc}") from exc


def describe(workspace: Path, julia_project: Path) -> str:
    """A human-readable account of the workspace's image, for ``sysimage status``."""

    from jutul_agent import sysimage as sysimage_mod

    image = sysimage_path(workspace)
    if not image.exists():
        return (
            f"No system image in {sysimage_dir(workspace)}.\n"
            "Build one with `jutul-agent sysimage build`."
        )

    stamp = sysimage_mod.read_stamp(workspace) or {}
    size_mb = image.stat().st_size / (1024 * 1024)
    contained = len(stamp.get("versions", {})) + len(stamp.get("path_packages", {}))
    lines = [
        f"Image:      {image} ({size_mb:.0f} MB)",
        f"Built:      {stamp.get('built_at', 'unknown')}"
        + (f" in {stamp['build_seconds'] / 60:.0f} min" if stamp.get("build_seconds") else ""),
        f"Julia:      {stamp.get('julia', 'unknown')} on {stamp.get('platform', 'unknown')}",
        f"CPU target: {stamp.get('cpu_target', 'unknown')}",
        f"Contains:   {contained} packages",
    ]

    decision = sysimage_mod.decide(workspace, julia_project, enabled=True)
    if decision.blocks:
        # A refusal carrying its own remedy is one a rebuild would not fix, which
        # also means the image itself is fine and "out of date" would misdescribe it.
        lines += ["", "Cannot be used here:" if decision.fix else "Out of date:", decision.reason]
        lines += ["", *(decision.fix or ("Rebuild: jutul-agent sysimage build",))]
    else:
        lines += ["", "Up to date with this workspace's environment."]
        lines += [f"note: {note}" for note in decision.notes]
    return "\n".join(lines)
