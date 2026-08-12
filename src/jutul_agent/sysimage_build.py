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
import platform
import shutil
import subprocess
import time
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from jutul_agent.sysimage import (
    image_suffix,
    on_windows,
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


# Left out of a Windows image so it fits under the DLL limit *with its
# docstrings intact*: sessions look Julia documentation up all the time, and an
# image stripped of metadata answers every ``@doc`` with nothing. Each name here
# is a leaf of the environment -- nothing else that is baked depends on it, or
# excluding it would remove nothing (``create_sysimage`` always bakes the
# closure of what it is given). These stay in the environment and load from
# their own pkgimages at first use: the graph/table export paths, not the
# solve/plot core.
WINDOWS_UNBAKED = frozenset({"CSV", "DataFrames", "GraphMakie", "LayeredLayouts", "NetworkLayout"})


def baked_packages(julia_project: Path) -> tuple[str, ...]:
    """The packages the image is built from (and verified to really contain).

    Every direct dependency of the project -- ``create_sysimage``'s own default
    -- except, on Windows, the :data:`WINDOWS_UNBAKED` leaves the image sheds to
    stay under the OS's image size limit (:data:`WINDOWS_IMAGE_LIMIT`).
    """

    deps = _project_deps(julia_project)
    if on_windows():
        deps -= WINDOWS_UNBAKED
    return tuple(sorted(deps))


def _project_deps(julia_project: Path) -> set[str]:
    try:
        data = tomllib.loads((julia_project / "Project.toml").read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return set()
    deps = data.get("deps")
    return set(deps) if isinstance(deps, dict) else set()


# The exact text HostCPUFeatures 0.1.x ships (src/cpu_info_aarch64.jl). Matched
# verbatim so a release that rewrites the file is left alone rather than mangled.
_VSCALE_UNGUARDED = """\
if Int === Int64
    @noinline vscale() = ccall("llvm.vscale.i64", llvmcall, Int64, ())
else
    @noinline vscale() = ccall("llvm.vscale.i32", llvmcall, Int32, ())
end"""

_VSCALE_GUARDED = """\
# Patched by jutul-agent before a system-image build: llvm.vscale is an SVE
# instruction, and LLVM aborts trying to emit it for CPUs without SVE. The
# function is never called on those CPUs, but a system-image build compiles
# even never-called methods (JuliaLang/PackageCompiler.jl#1070).
if _has_aarch64_sve()
    if Int === Int64
        @noinline vscale() = ccall("llvm.vscale.i64", llvmcall, Int64, ())
    else
        @noinline vscale() = ccall("llvm.vscale.i32", llvmcall, Int32, ())
    end
else
    vscale() = 1
end"""


def _depot_paths() -> list[Path]:
    """The depots Julia loads packages from, in `JULIA_DEPOT_PATH` order."""

    default = Path.home() / ".julia"
    raw = os.environ.get("JULIA_DEPOT_PATH", "")
    if not raw:
        return [default]
    # An empty entry in JULIA_DEPOT_PATH stands for the default depot list.
    return [Path(entry).expanduser() if entry else default for entry in raw.split(os.pathsep)]


def _guard_vscale_llvmcall(
    *,
    machine: str | None = None,
    depots: Sequence[Path] | None = None,
) -> list[Path]:
    """Keep HostCPUFeatures' `vscale()` out of codegen on CPUs without SVE.

    `vscale()` is an unconditional `llvm.vscale` llvmcall — an SVE instruction
    Apple Silicon cannot select. It is never *called* there, but an image build
    compiles even never-called methods, so LLVM aborts the whole build
    (JuliaLang/PackageCompiler.jl#1070, unfixed as of HostCPUFeatures 0.1.18).
    Guarded behind the package's own ``_has_aarch64_sve()``, SVE hardware keeps
    the original definition, so the patch applies to every aarch64 machine. It
    lands in the depot — the manifest pins a registry version, so there is
    nowhere else short of a fork — idempotently, and a copy whose text has
    moved on from the known shape is reported and left alone.

    Returns the files patched by this call.
    """

    machine = (machine or platform.machine()).lower()
    if machine not in ("arm64", "aarch64"):
        return []
    patched: list[Path] = []
    for depot in depots if depots is not None else _depot_paths():
        for source in sorted(depot.glob("packages/HostCPUFeatures/*/src/cpu_info_aarch64.jl")):
            try:
                text = source.read_text(encoding="utf-8")
            except OSError:
                continue
            if _VSCALE_UNGUARDED in text:
                source.chmod(source.stat().st_mode | 0o200)
                source.write_text(
                    text.replace(_VSCALE_UNGUARDED, _VSCALE_GUARDED, 1), encoding="utf-8"
                )
                patched.append(source)
            elif "llvm.vscale" in text and "vscale() = 1" not in text:
                # Neither the shape this guard knows nor an already-guarded copy:
                # a build on this machine will likely abort in LLVM, and saying
                # where beats a silent pass followed by that abort.
                print(
                    f"note: {source} defines an llvm.vscale llvmcall this build "
                    "cannot guard (the package's text is not the shape it knows); "
                    "if the build aborts with 'LLVM ERROR: Cannot select: "
                    "vscale', this file is why."
                )
    return patched


def _warmup_script(snippets: Sequence[str]) -> str:
    """One Julia file running every warm-up snippet, each on its own fuse.

    The same ``warm_code`` a session runs in the background after start-up;
    run during the build instead, everything it compiles is baked into the
    image. A snippet that throws costs only its own coverage, never the build
    or the other snippets' work: by this point the workload is an optimisation,
    worth a warning in the build log, not a discarded image.
    """

    parts = []
    for index, snippet in enumerate(snippets, 1):
        parts.append(
            f"# --- warm-up snippet {index} of {len(snippets)}\n"
            "try\n"
            f"{snippet.strip()}\n"
            "catch err\n"
            f'    @warn "warm-up snippet {index} failed; the image loses its coverage" err\n'
            "end\n"
        )
    return "\n".join(parts)


def build(
    *,
    workspace: Path,
    julia_project: Path,
    cpu_target: str = "native",
    verify: bool = True,
    warmup_code: Sequence[str] = (),
) -> BuildResult:
    """Build, verify and install this workspace's system image.

    ``cpu_target`` defaults to ``"native"``: the fastest code for the machine
    doing the build, and unusable anywhere else. Passing a portable target trades
    some speed for an image that runs on other hardware.

    ``warmup_code`` are Julia snippets (a capability's ``warm_code``, typically)
    run once during the build so their compilation lands in the image. The stamp
    does not describe them: a snippet that changes later affects how much the
    next image has pre-compiled, never whether the current one is safe to use.
    """

    packages = baked_packages(julia_project)
    if not packages:
        raise SysimageBuildError(
            f"{julia_project} has no dependencies to build an image from. "
            "Run `jutul-agent init --sim <name>` first."
        )

    for source in _guard_vscale_llvmcall():
        print(
            f"note: patched {source} so the build can run on this CPU: its "
            "vscale() llvmcall is an SVE instruction a system-image build "
            "cannot compile here (JuliaLang/PackageCompiler.jl#1070)."
        )

    env = ensure_builder_env()
    destination = sysimage_path(workspace)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Built beside its destination so the final move is a rename within one
    # filesystem, and so a half-built image is never at the path the guard reads.
    candidate = destination.with_name(f"candidate-{os.getpid()}{image_suffix()}")

    execution_file = destination.with_name(f"warmup-{os.getpid()}.jl")
    execution = ""
    if warmup_code:
        execution_file.write_text(_warmup_script(warmup_code), encoding="utf-8")
        execution = f' precompile_execution_file = raw"{execution_file.as_posix()}",'
        print(
            f"Baking the warm-up workload into the image ({len(warmup_code)} "
            "snippet(s)); what it compiles, no session pays for again..."
        )

    started = time.monotonic()
    try:
        _julia(
            [
                "using PackageCompiler",
                # The bake list is passed explicitly (rather than defaulted to
                # every direct dependency) so the Windows exclusions above are
                # honoured; elsewhere the list *is* every direct dependency.
                "create_sysimage(" + _julia_string_vector(list(packages)) + ";"
                f' sysimage_path = raw"{candidate.as_posix()}",'
                f' project = raw"{julia_project.as_posix()}",'
                f' cpu_target = "{cpu_target}",'
                + execution
                + " incremental = true, filter_stdlibs = false)",
            ],
            project=env,
            what="building the system image",
        )
        if not candidate.exists():
            raise SysimageBuildError("the build reported success but produced no image")
        _check_loadable_size(candidate)
        if verify:
            # Narrated because it is captured: starting Julia on a multi-GB image
            # and rendering a figure takes long enough to read as a stall otherwise.
            print("Verifying the new image before installing it...")
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
        execution_file.unlink(missing_ok=True)

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


# Windows refuses to map a DLL whose in-memory span -- the PE header's
# SizeOfImage, not the file's size on disk -- exceeds this. LoadLibrary fails
# with the unhelpful "%1 is not a valid Win32 application" (error 193), a hard
# OS limit PackageCompiler cannot lift. The value was bisected empirically on
# Windows 11 with synthetic DLLs: 0x77000000 itself loads, one page more is
# refused, and the boundary is identical whether the bytes are file-backed or
# virtual-only. A full simulator environment builds to right around this line.
WINDOWS_IMAGE_LIMIT = 0x77000000


def _check_loadable_size(candidate: Path) -> None:
    """Refuse an image Windows will not load, with the reason spelled out.

    Refuse, not repair: stripping the DWARF sections would fit easily (~170 MiB
    of a full build) and the image still runs, but every stack frame of
    image-compiled code then resolves to nothing -- errors lose their traces,
    and real init code that indexes ``stacktrace()`` (FreeTypeAbstraction, via
    the leaf-package precompiles) dies on the empty array. An image that
    cannot exist beats one that fails like that at a distance.

    Without this the failure arrives from verification as "%1 is not a valid
    Win32 application", which explains nothing and looks like a broken build
    rather than a hard OS limit.
    """

    if not on_windows():
        return
    size = _loader_size(candidate)
    if size <= WINDOWS_IMAGE_LIMIT:
        return
    raise SysimageBuildError(
        f"the built image spans {size / 2**30:.2f} GiB in memory, and Windows "
        f"refuses to load a DLL over {WINDOWS_IMAGE_LIMIT / 2**30:.2f} GiB, so "
        f"it can never be used (it is {(size - WINDOWS_IMAGE_LIMIT) / 2**20:.0f} "
        "MiB over). The environment has too much to bake on this platform: trim "
        "its dependencies or precompile workloads (capability packages "
        "included), or run without a system image until Julia ships "
        "compressed images (planned for 1.13)."
    )


def _loader_size(candidate: Path) -> int:
    """The size the Windows loader judges: SizeOfImage, file size as fallback.

    SizeOfImage is the image's mapped span, and it is what the loader checks:
    the file on disk is tens of MiB bigger (debug sections and symbol table
    counted differently), which is how a build once passed a file-size check
    and was refused anyway. When the header cannot be read, the file size errs
    toward refusing, never toward shipping an unloadable image.
    """

    import struct

    try:
        with candidate.open("rb") as f:
            head = f.read(1024)
        if len(head) >= 0x40 and head[:2] == b"MZ":
            (pe_off,) = struct.unpack_from("<I", head, 0x3C)
            if head[pe_off : pe_off + 4] == b"PE\0\0":
                (size_of_image,) = struct.unpack_from("<I", head, pe_off + 24 + 56)
                return size_of_image
    except (OSError, struct.error):
        pass
    return candidate.stat().st_size


def _precompile_against(image: Path, julia_project: Path) -> None:
    """Refresh the cache for anything left outside the image. Best-effort.

    A pkgimage records which system image it was built against and is rebuilt
    when that changes, so without this the first session after a build pays a
    compile it looks like the image should have removed. With everything baked
    there is usually nothing to do, which is why a failure here is only worth a
    note: the image is already installed and already verified.
    """

    print("Refreshing precompile caches against the new image (usually instant)...")
    result = _run_julia(
        [
            "julia",
            f"--sysimage={image}",
            f"--project={julia_project}",
            "--startup-file=no",
            "-e",
            "using Pkg; Pkg.precompile()",
        ],
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
        # The loader-size check runs before verification, so this only fires if
        # the OS limit drifts from the measured constant; better a translated
        # error than the loader's, an hour of build time in.
        hint = (
            "\nThat error is the Windows loader refusing the DLL for its size, "
            "not a broken build; WINDOWS_IMAGE_LIMIT is calibrated too high for "
            "this machine."
            if "not a valid Win32 application" in tail
            else ""
        )
        raise SysimageBuildError(
            "the system image was built but failed verification, so it was "
            f"discarded and the previous one (if any) is untouched.\nJulia said:\n{tail}{hint}"
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
