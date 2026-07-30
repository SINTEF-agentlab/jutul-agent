"""Tests for the workspace backend: real paths, the depot write-guard, and grep.

The agent's filesystem is real paths; installed package source in the shared
Julia depot is read-only (writes and deletes refused), a ``Pkg.develop``
checkout is writable, and a type-filtered grep recurses into subdirectories.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from jutul_agent.agent.backend import WorkspaceShellBackend
from jutul_agent.agent.builder import PackageSource, build_backend


@pytest.fixture
def source_dir(tmp_path: Path) -> Path:
    pkg = tmp_path / "BattMo"
    (pkg / "examples").mkdir(parents=True)
    (pkg / "examples" / "demo.jl").write_text("println(1)\n", encoding="utf-8")
    return pkg


def test_workspace_backend_relative_and_absolute_resolve_to_the_same_file(tmp_path: Path) -> None:
    # Real-path mode (as production runs it): a relative path resolves against
    # the workspace and the file's absolute path resolves to itself — the same
    # file the REPL and execute see, with no phantom <ws>/<abs> re-rooting.
    ws = tmp_path.resolve()
    backend = WorkspaceShellBackend(root_dir=ws, virtual_mode=False, inherit_env=True)

    backend.write("model.jl", "x = 1\n")
    assert (ws / "model.jl").read_text() == "x = 1\n"
    # The relative path and the file's real absolute path read the same file.
    for key in ("model.jl", str(ws / "model.jl")):
        result = backend.read(key)
        assert result.error is None
        assert "x = 1" in result.file_data["content"]


def test_workspace_backend_nests_relative_subdirs(tmp_path: Path) -> None:
    ws = tmp_path.resolve()
    backend = WorkspaceShellBackend(root_dir=ws, virtual_mode=False, inherit_env=True)
    backend.write("sub/dir/a.jl", "y = 2\n")
    assert (ws / "sub" / "dir" / "a.jl").read_text() == "y = 2\n"


def test_workspace_backend_absolute_path_writes_the_real_file(tmp_path: Path) -> None:
    # In real-path mode an absolute path is honored as itself (no workspace-only
    # restriction, no phantom): the file tools behave like the real filesystem,
    # matching execute and run_julia.
    ws = tmp_path.resolve()
    backend = WorkspaceShellBackend(root_dir=ws, virtual_mode=False, inherit_env=True)

    target = ws / "nested" / "model.jl"
    res = backend.write(str(target), "x = 1\n")
    assert res.error is None
    assert target.read_text() == "x = 1\n"
    # Exactly one file was created, at the real absolute path (no re-rooted copy).
    assert list(ws.rglob("model.jl")) == [target]


def test_registry_package_source_is_read_only_at_its_real_path(
    tmp_path: Path, source_dir: Path
) -> None:
    # Registry source lives in the shared depot: the agent reads/greps it at its
    # real path, but writes are refused so it can't corrupt the depot.
    backend = build_backend(
        workspace=tmp_path,
        package_sources=[PackageSource(name="BattMo", path=source_dir)],
    )
    assert backend.read(str(source_dir / "examples" / "demo.jl")).error is None
    write = backend.write(str(source_dir / "examples" / "x.jl"), "y")
    assert write.error is not None
    assert "read-only" in write.error
    assert not (source_dir / "examples" / "x.jl").exists()


def test_developed_source_outside_the_depot_is_writable(tmp_path: Path, source_dir: Path) -> None:
    # A Pkg.develop checkout (writable=True) is not a read-only root, so the
    # agent can edit it at its real path.
    backend = build_backend(
        workspace=tmp_path,
        package_sources=[PackageSource(name="BattMo", path=source_dir, writable=True)],
    )
    result = backend.write(str(source_dir / "examples" / "x.jl"), "y = 1")
    assert result.error is None
    assert (source_dir / "examples" / "x.jl").read_text(encoding="utf-8") == "y = 1"


def test_depot_readonly_roots_guard_the_packages_dir(tmp_path: Path) -> None:
    from jutul_agent.agent.builder import _depot_readonly_roots

    depot = tmp_path / ".julia" / "packages"
    registry = depot / "JutulDarcy" / "abc123"
    dev = tmp_path / "dev" / "MyPkg"
    # A registry source guards the whole depot `packages` dir (so a mid-session
    # Pkg.add is covered); a dev checkout is excluded; no sources -> nothing.
    roots = _depot_readonly_roots(
        [
            PackageSource(name="JutulDarcy", path=registry, writable=False),
            PackageSource(name="MyPkg", path=dev, writable=True),
        ]
    )
    assert roots == (depot,)
    assert _depot_readonly_roots(None) == ()


def test_recursive_glob_normalizes_bare_extension() -> None:
    from jutul_agent.agent.backend import _recursive_glob

    assert _recursive_glob("*.jl") == "**/*.jl"
    assert _recursive_glob("foo.jl") == "**/foo.jl"
    assert _recursive_glob("**/*.jl") == "**/*.jl"  # already recursive
    assert _recursive_glob("src/*.jl") == "src/*.jl"  # carries a path, left alone
    assert _recursive_glob(None) is None


def test_grep_with_extension_filter_recurses_into_subdirs(tmp_path: Path, source_dir: Path) -> None:
    # A type-filtered grep (glob="*.jl") must find matches in subdirectories. The
    # deepagents default treats a bare glob as non-recursive, which silently hid
    # package-extension code (e.g. plot_variable_graph in Jutul's src/ext/).
    deep = source_dir / "src" / "ext" / "graphmakie_ext.jl"
    deep.parent.mkdir(parents=True)
    deep.write_text("function plot_variable_graph(x) end\n", encoding="utf-8")

    backend = build_backend(workspace=tmp_path)
    hit = backend.grep("plot_variable_graph", path=str(source_dir), glob="*.jl")
    matches = getattr(hit, "matches", hit)
    assert matches, "grep with glob='*.jl' must recurse into src/ext/"


def test_grep_forwards_search_options_to_the_composite(tmp_path: Path) -> None:
    # Our grep override exists only to normalize `glob`, so it must pass the
    # framework's other search arguments straight through. deepagents inspects
    # the signature and silently degrades to trimming a fully materialized
    # result when one is missing, so a narrowed override is invisible until a
    # search on a big tree is unexpectedly slow.
    for index in range(3):
        (tmp_path / f"a{index}.jl").write_text("using Jutul\n", encoding="utf-8")

    backend = build_backend(workspace=tmp_path)
    capped = backend.grep("using Jutul", path=str(tmp_path), glob="*.jl", max_count=1)
    assert capped.error is None
    assert len(capped.matches) == 1
    assert capped.truncated


async def test_agrep_forwards_search_options_to_the_composite(tmp_path: Path) -> None:
    for index in range(3):
        (tmp_path / f"a{index}.jl").write_text("using Jutul\n", encoding="utf-8")

    backend = build_backend(workspace=tmp_path)
    capped = await backend.agrep("using Jutul", path=str(tmp_path), glob="*.jl", max_count=1)
    assert capped.error is None
    assert len(capped.matches) == 1
    assert capped.truncated


def test_delete_is_refused_inside_the_read_only_depot(tmp_path: Path, source_dir: Path) -> None:
    # `delete` removes a directory recursively, so the depot guard has to cover
    # it too: one call would otherwise take out an installed package's source.
    backend = build_backend(
        workspace=tmp_path,
        package_sources=[PackageSource(name="BattMo", path=source_dir)],
    )
    result = backend.delete(str(source_dir / "examples"))
    assert result.error is not None
    assert "read-only" in result.error
    assert (source_dir / "examples" / "demo.jl").exists()


def test_delete_removes_a_workspace_directory(tmp_path: Path) -> None:
    backend = build_backend(workspace=tmp_path)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (scratch / "run.jl").write_text("x = 1\n", encoding="utf-8")

    assert backend.delete(str(scratch)).error is None
    assert not scratch.exists()


def test_execute_refuses_shell_julia_only(tmp_path) -> None:
    from jutul_agent.agent.backend import WorkspaceShellBackend

    backend = WorkspaceShellBackend(root_dir=tmp_path, virtual_mode=False)
    for command in (
        "julia -e 'include(\"analysis.jl\")'",
        "pwd && julia script.jl",
        "echo hi | julia",
    ):
        result = backend.execute(command)
        assert result.exit_code == 2, command
        assert "run_julia" in result.output

    # Only the head token of a shell segment counts as an invocation.
    for command in ("python3 -c 'print(1)'", "grep -r julia . || true"):
        assert backend.execute(command).exit_code != 2, command
