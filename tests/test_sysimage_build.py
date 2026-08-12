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
    _VSCALE_GUARDED,
    _VSCALE_UNGUARDED,
    SysimageBuildError,
    _guard_vscale_llvmcall,
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
        # What the "build" writes; a test can swap in real PE bytes.
        self.image = b"image"

    def __call__(self, argv, *, capture: bool = False):
        self.calls.append(list(argv))
        script = argv[-1]
        if "create_sysimage" in script:
            if self.build_ok:
                Path(_quoted_path(script, "sysimage_path")).write_bytes(self.image)
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


@pytest.fixture(autouse=True)
def _no_real_depot(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep every test away from the machine's actual ~/.julia.

    ``build`` guards the depot's HostCPUFeatures copy as a side effect; run
    unstubbed on an aarch64 developer machine, the suite would edit real
    packages. Tests of the guard itself call the imported function, which this
    stub of the module attribute does not touch.
    """

    monkeypatch.setattr(sysimage_build, "_guard_vscale_llvmcall", lambda **kw: [])


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


def test_windows_bakes_fewer_leaves_to_duck_the_dll_limit(
    tmp_path: Path, julia: _FakeJulia, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Windows cannot load a DLL of 2 GiB or more, and a full env builds to right
    over that line. The image sheds leaf packages rather than docstrings:
    stripping metadata would fit too, but then every `@doc` in a session
    answers with nothing, which an agent that reads documentation cannot afford."""
    ws = tmp_path / "ws"
    env = write_project(tmp_path / "env", {"JutulDarcy": "a", "DataFrames": "b", "CSV": "c"})

    monkeypatch.setattr(sysimage_build, "on_windows", lambda: True)
    monkeypatch.setattr(sysimage, "on_windows", lambda: True)
    assert baked_packages(env) == ("JutulDarcy",)
    build(workspace=ws, julia_project=env)
    create = next(call for call in julia.calls if "create_sysimage" in call[-1])
    assert '"JutulDarcy"' in create[-1]
    assert "DataFrames" not in create[-1] and "CSV" not in create[-1]

    julia.calls.clear()
    monkeypatch.setattr(sysimage_build, "on_windows", lambda: False)
    monkeypatch.setattr(sysimage, "on_windows", lambda: False)
    # Elsewhere there is no limit to duck under, and everything is baked.
    assert baked_packages(env) == ("CSV", "DataFrames", "JutulDarcy")
    build(workspace=ws, julia_project=env)
    create = next(call for call in julia.calls if "create_sysimage" in call[-1])
    assert '"DataFrames"' in create[-1] and '"CSV"' in create[-1]


def test_an_image_windows_cannot_load_is_refused_with_the_reason(
    tmp_path: Path, julia: _FakeJulia, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Over the OS limit the loader says '%1 is not a valid Win32 application',
    which explains nothing; the build has to say what actually happened."""
    ws = tmp_path / "ws"
    env = write_project(tmp_path / "env", {"JutulDarcy": "a"})
    monkeypatch.setattr(sysimage_build, "on_windows", lambda: True)
    monkeypatch.setattr(sysimage, "on_windows", lambda: True)
    # The fake build writes a 5-byte image; a 3-byte "limit" puts it over,
    # and 5 bytes of not-a-PE also exercises the file-size fallback.
    monkeypatch.setattr(sysimage_build, "WINDOWS_IMAGE_LIMIT", 3)

    with pytest.raises(SysimageBuildError, match="refuses to load"):
        build(workspace=ws, julia_project=env)

    assert not sysimage.sysimage_path(ws).exists()
    assert sysimage.read_stamp(ws) is None
    assert list(sysimage.sysimage_dir(ws).glob("candidate-*")) == []


# ---------------------------------------------------------------------------
# The Windows loader limit, which is on the mapped span, not the file.


def write_pe(path: Path, *, data_size: int = 0x1000, debug_sizes: tuple[int, ...] = ()) -> int:
    """A minimal PE32+ DLL shaped the way the linker shapes a sysimage:
    load-bearing sections first, DWARF debug sections as a contiguous tail,
    COFF symbol table after everything. Returns the header's SizeOfImage."""
    import struct

    file_align, sect_align = 512, 4096

    def up(x: int, a: int) -> int:
        return (x + a - 1) // a * a

    names = [b".text", b".data"] + [f"/{4 + 15 * i}".encode() for i in range(len(debug_sizes))]
    vsizes = [16, data_size, *debug_sizes]
    headers = up(64 + 4 + 20 + 240 + len(names) * 40, file_align)
    va, raw, rows = sect_align, headers, []
    for name, vsize in zip(names, vsizes, strict=True):
        rawsize = up(vsize, file_align)
        rows.append((name, vsize, va, rawsize, raw))
        va, raw = up(va + vsize, sect_align), raw + rawsize
    size_of_image = va

    dos = bytearray(64)
    dos[:2] = b"MZ"
    struct.pack_into("<I", dos, 0x3C, 64)
    coff = struct.pack("<HHIIIHH", 0x8664, len(names), 0, raw, 1, 240, 0x2022)
    opt = bytearray(240)
    struct.pack_into("<H", opt, 0, 0x20B)
    struct.pack_into("<Q", opt, 24, 0x180000000)
    struct.pack_into("<II", opt, 32, sect_align, file_align)
    struct.pack_into("<I", opt, 56, size_of_image)
    struct.pack_into("<I", opt, 60, headers)
    struct.pack_into("<I", opt, 108, 16)
    table = b"".join(
        struct.pack("<8sIIIIIIHHI", name, vsize, va, rawsize, rawptr, 0, 0, 0, 0, 0x42000040)
        for name, vsize, va, rawsize, rawptr in rows
    )
    body = bytes(dos) + b"PE\0\0" + coff + bytes(opt) + table
    with path.open("wb") as f:
        f.write(body)
        f.write(b"\0" * (headers - len(body)))
        for _, _, _, rawsize, _ in rows:
            f.write(b"\0" * rawsize)
        f.write(b"symtab")  # what PointerToSymbolTable points at
    return size_of_image


def test_loader_size_reads_the_pe_header_not_the_file(tmp_path: Path) -> None:
    """Windows judges SizeOfImage (the mapped span), not bytes on disk; the
    failing geoteric build was 95 MiB under the old file-size check and was
    refused anyway."""
    image = tmp_path / "sys.dll"
    size_of_image = write_pe(image, debug_sizes=(0x1000,))
    assert sysimage_build._loader_size(image) == size_of_image
    assert sysimage_build._loader_size(image) != image.stat().st_size

    not_pe = tmp_path / "not-pe.dll"
    not_pe.write_bytes(b"image")
    assert sysimage_build._loader_size(not_pe) == 5


def test_a_real_pe_over_the_limit_is_refused_by_its_mapped_size(
    tmp_path: Path, julia: _FakeJulia, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The judgement is the header's SizeOfImage, never the file: an image
    whose file squeaks under the limit is still refused when its mapped span
    does not, which is exactly the build that once died in verification."""
    ws = tmp_path / "ws"
    env = write_project(tmp_path / "env", {"JutulDarcy": "a"})
    template = tmp_path / "template.dll"
    size_of_image = write_pe(template)
    julia.image = template.read_bytes()

    monkeypatch.setattr(sysimage_build, "on_windows", lambda: True)
    monkeypatch.setattr(sysimage, "on_windows", lambda: True)
    monkeypatch.setattr(sysimage_build, "WINDOWS_IMAGE_LIMIT", size_of_image - 1)
    with pytest.raises(SysimageBuildError, match="refuses to load"):
        build(workspace=ws, julia_project=env)

    monkeypatch.setattr(sysimage_build, "WINDOWS_IMAGE_LIMIT", size_of_image)
    assert build(workspace=ws, julia_project=env).path.exists()


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


def test_a_current_image_is_not_rebuilt_when_the_caller_allows_skipping(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, julia: _FakeJulia
) -> None:
    """``init`` re-runs the build step on every run of an opted-in folder, so an
    image that still matches the environment must short-circuit there. The
    explicit ``sysimage build`` never skips: asking for a build gets one."""
    from jutul_agent.interfaces.cli import sysimage as cmd
    from jutul_agent.workspace import WorkspaceConfig

    env = write_project(workspace / ".jutul-agent" / "julia-env", {"JutulDarcy": "a"})
    monkeypatch.setattr(cmd, "prepare_environment", lambda *a, **k: None)
    config = WorkspaceConfig(simulator="jutuldarcy", sysimage=True)

    def builds() -> int:
        return sum("create_sysimage" in call[-1] for call in julia.calls)

    assert cmd.build_for_workspace(object(), workspace, env, config, skip_current=True)
    assert builds() == 1  # nothing to skip yet: the first init really builds

    assert cmd.build_for_workspace(object(), workspace, env, config, skip_current=True)
    assert builds() == 1  # unchanged env: the re-init skips

    assert cmd.build_for_workspace(object(), workspace, env, config)
    assert builds() == 2  # the explicit command rebuilds regardless


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


# ---------------------------------------------------------------------------
# The HostCPUFeatures vscale guard (aarch64 without SVE cannot build otherwise).


def depot_with(tmp_path: Path, text: str) -> tuple[Path, Path]:
    """A depot holding one installed HostCPUFeatures copy, read-only like Pkg's."""

    source = tmp_path / "packages" / "HostCPUFeatures" / "ZTXz4" / "src" / "cpu_info_aarch64.jl"
    source.parent.mkdir(parents=True)
    source.write_text(text, encoding="utf-8")
    source.chmod(0o444)
    return tmp_path, source


PRISTINE = "_has_aarch64_sve() = false\n\n" + _VSCALE_UNGUARDED + "\n\nfma_fast() = True()\n"


def test_the_vscale_llvmcall_is_guarded_on_aarch64(tmp_path: Path) -> None:
    depot, source = depot_with(tmp_path, PRISTINE)

    patched = _guard_vscale_llvmcall(machine="arm64", depots=[depot])

    assert patched == [source]
    text = source.read_text(encoding="utf-8")
    assert _VSCALE_GUARDED in text
    assert _VSCALE_UNGUARDED not in text
    # The rest of the file is not the guard's to touch.
    assert text.startswith("_has_aarch64_sve() = false\n")
    assert text.endswith("fma_fast() = True()\n")


def test_the_guard_is_idempotent(tmp_path: Path) -> None:
    depot, source = depot_with(tmp_path, PRISTINE)
    _guard_vscale_llvmcall(machine="aarch64", depots=[depot])
    before = source.read_text(encoding="utf-8")

    assert _guard_vscale_llvmcall(machine="aarch64", depots=[depot]) == []
    assert source.read_text(encoding="utf-8") == before


def test_a_hand_patched_copy_is_left_alone(tmp_path: Path) -> None:
    # The shape someone applies by hand differs in its comments; what marks it
    # as already guarded is the stub definition.
    hand = PRISTINE.replace(
        _VSCALE_UNGUARDED,
        "# guarded by hand\nif _has_aarch64_sve()\n"
        '    @noinline vscale() = ccall("llvm.vscale.i64", llvmcall, Int64, ())\n'
        "else\n    vscale() = 1\nend",
    )
    depot, source = depot_with(tmp_path, hand)

    assert _guard_vscale_llvmcall(machine="arm64", depots=[depot]) == []
    assert source.read_text(encoding="utf-8") == hand


def test_an_unfamiliar_vscale_is_reported_not_mangled(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    unfamiliar = 'sve_width() = ccall("llvm.vscale.i64", llvmcall, Int64, ())\n'
    depot, source = depot_with(tmp_path, unfamiliar)

    assert _guard_vscale_llvmcall(machine="arm64", depots=[depot]) == []
    assert source.read_text(encoding="utf-8") == unfamiliar
    out = capsys.readouterr().out
    assert str(source) in out
    assert "Cannot select" in out


def test_the_guard_does_nothing_off_aarch64(tmp_path: Path) -> None:
    depot, source = depot_with(tmp_path, PRISTINE)

    assert _guard_vscale_llvmcall(machine="x86_64", depots=[depot]) == []
    assert _VSCALE_UNGUARDED in source.read_text(encoding="utf-8")


def test_the_build_says_what_it_patched(
    tmp_path: Path,
    julia: _FakeJulia,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ws = tmp_path / "ws"
    env = write_project(tmp_path / "env", {"JutulDarcy": "a"})
    guarded = tmp_path / "depot" / "cpu_info_aarch64.jl"
    monkeypatch.setattr(sysimage_build, "_guard_vscale_llvmcall", lambda **kw: [guarded])

    build(workspace=ws, julia_project=env)

    out = capsys.readouterr().out
    assert f"patched {guarded}" in out
    assert "PackageCompiler.jl#1070" in out


# ---------------------------------------------------------------------------
# The warm-up workload baked into the image.


def test_the_warmup_workload_reaches_the_build(
    tmp_path: Path, julia: _FakeJulia, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = tmp_path / "ws"
    env = write_project(tmp_path / "env", {"JutulDarcy": "a"})
    seen: dict[str, object] = {}

    def capturing(argv, *, capture: bool = False):
        script = argv[-1]
        if "precompile_execution_file" in script:
            path = Path(_quoted_path(script, "precompile_execution_file"))
            seen["path"] = path
            seen["content"] = path.read_text(encoding="utf-8")
        return julia(argv, capture=capture)

    monkeypatch.setattr(sysimage_build, "_run_julia", capturing)

    build(workspace=ws, julia_project=env, warmup_code=("import Foo\nFoo.warm()", "Bar.warm()"))

    content = seen["content"]
    assert isinstance(content, str)
    assert "import Foo\nFoo.warm()" in content
    assert "Bar.warm()" in content
    # Each snippet on its own fuse: a drifted workload costs coverage, not the build.
    assert content.count("try\n") == 2
    assert content.count("catch err") == 2
    # The workload script is scaffolding, not part of the installed image.
    path = seen["path"]
    assert isinstance(path, Path)
    assert not path.exists()


def test_without_a_workload_the_build_asks_for_none(tmp_path: Path, julia: _FakeJulia) -> None:
    ws = tmp_path / "ws"
    env = write_project(tmp_path / "env", {"JutulDarcy": "a"})

    build(workspace=ws, julia_project=env)

    script = next(call[-1] for call in julia.calls if "create_sysimage" in call[-1])
    assert "precompile_execution_file" not in script


def test_the_command_bakes_capability_warm_code(
    tmp_path: Path, julia: _FakeJulia, monkeypatch: pytest.MonkeyPatch
) -> None:
    from jutul_agent.agent.capabilities import Capability
    from jutul_agent.interfaces.cli import sysimage as cmd
    from jutul_agent.workspace import WorkspaceConfig

    ws = tmp_path / "ws"
    ws.mkdir()
    env = write_project(ws / ".jutul-agent" / "julia-env", {"JutulDarcy": "a"})
    monkeypatch.setattr(cmd, "prepare_environment", lambda *a, **k: None)
    monkeypatch.setattr(
        "jutul_agent.agent.capabilities.discover_extensions",
        lambda: [Capability(name="geo", warm_code=("Geo.warm()",))],
    )
    received: dict[str, object] = {}
    real_build = sysimage_build.build

    def recording(**kwargs):
        received.update(kwargs)
        return real_build(**kwargs)

    monkeypatch.setattr(sysimage_build, "build", recording)
    config = WorkspaceConfig(simulator="jutuldarcy", sysimage=True)

    assert cmd.build_for_workspace(object(), ws, env, config)
    assert received["warmup_code"] == ["Geo.warm()"]
