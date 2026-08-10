"""End-to-end checks against a system image that was really built.

Everything else about the feature is tested with Julia stubbed out, which is the
right trade for tens of minutes of build time. What no stub can answer is
whether the image jutul-agent produces actually does the thing it exists to do:
start a kernel with the packages already in memory. That is what these check.

They need a workspace that has been built, so they are gated on
``JUTUL_AGENT_SYSIMAGE_WORKSPACE`` pointing at one:

    jutul-agent init --sim jutuldarcy --workspace /tmp/ws
    jutul-agent sysimage build --workspace /tmp/ws
    JUTUL_AGENT_SYSIMAGE_WORKSPACE=/tmp/ws pytest tests/test_sysimage_integration.py

The weekly simulator workflow builds one and sets the variable; locally these
skip until you point them at a workspace of your own.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from jutul_agent import sysimage
from jutul_agent.juliakernel import JuliaKernel, KernelConfig
from jutul_agent.workspace import resolve_julia_project

pytestmark = pytest.mark.integration

WORKSPACE_VAR = "JUTUL_AGENT_SYSIMAGE_WORKSPACE"


@pytest.fixture
def built() -> tuple[Path, Path, Path]:
    """The workspace, its Julia project, and its image, or a skip."""

    raw = os.environ.get(WORKSPACE_VAR)
    if not raw:
        pytest.skip(f"set {WORKSPACE_VAR} to a workspace with a built system image")
    ws = Path(raw).expanduser().resolve()
    project = resolve_julia_project(ws)
    decision = sysimage.decide(ws, project, enabled=True)
    if not decision.usable:
        pytest.skip(f"{ws} has no usable system image: {decision.reason or decision.status}")
    assert decision.path is not None
    return ws, project, decision.path


@pytest.fixture
def display():
    """A DISPLAY for the kernel, as production gives it."""

    from jutul_agent.display import has_display, managed_display, xvfb_available

    if has_display():
        yield None
        return
    if not xvfb_available():
        pytest.skip("no display and Xvfb is not available")
    with managed_display() as value:
        yield value


def _kernel(project: Path, display: str | None, image: Path | None) -> JuliaKernel:
    return JuliaKernel(
        KernelConfig(
            julia_project=project,
            sysimage=image,
            env={"DISPLAY": display} if display else None,
        )
    )


async def test_the_packages_are_already_loaded_at_startup(built, display) -> None:
    """The whole point: no ``using``, and the simulator is already there.

    Contrasted against the same environment without the image, because
    ``isdefined`` would also be satisfied by a package something else loaded.
    """

    _, project, image = built

    async with _kernel(project, display, image) as julia:
        result = await julia.eval("isdefined(Main, :JutulDarcy)")
    assert "true" in result.output, f"not loaded from the image: {result.output!r}"

    async with _kernel(project, display, None) as julia:
        without = await julia.eval("isdefined(Main, :JutulDarcy)")
    assert "false" in without.output, (
        "the plain kernel already has JutulDarcy, so the check above proves nothing: "
        f"{without.output!r}"
    )


async def test_the_baked_environment_still_solves(built, display) -> None:
    """An image that starts but can't run the simulator would be worse than none."""

    _, project, image = built

    async with _kernel(project, display, image) as julia:
        result = await julia.eval(
            "using Jutul, JutulDarcy; "
            "mesh = CartesianMesh((3, 3, 1), (30.0, 30.0, 10.0)); "
            "domain = reservoir_domain(mesh, permeability = 1.0e-13, porosity = 0.2); "
            "size(domain[:permeability])"
        )
    assert result.error is None, result.error


def test_the_guard_notices_an_edited_package(built) -> None:
    """The refusal, against a real image and a real edit.

    Path-tracked packages are the ones that change without a version bump, so
    this is the case the whole staleness check is built around.
    """

    ws, project, _ = built
    stamp = sysimage.read_stamp(ws) or {}
    tracked = stamp.get("path_packages") or {}
    if not tracked:
        pytest.skip("this image contains no path-tracked packages")

    state = sysimage.read_env_state(project)
    name = next(iter(tracked))
    source = _a_source_file(project, name)
    if source is None:
        pytest.skip(f"cannot locate the sources of {name}")

    original = source.read_bytes()
    try:
        source.write_bytes(original + b"\n# edited by the system-image guard test\n")
        decision = sysimage.decide(ws, project, enabled=True)
        assert decision.status == sysimage.DIVERGENT
        assert decision.blocks and not decision.usable
        assert name in decision.reason
        assert "sysimage build" in sysimage.refusal(decision)
    finally:
        source.write_bytes(original)

    # And the restored tree is trusted again: the check is content-addressed,
    # not a one-way "this workspace has been touched" latch.
    assert sysimage.decide(ws, project, enabled=True).status == sysimage.CURRENT
    assert sysimage.read_env_state(project).path_packages == state.path_packages


def _a_source_file(project: Path, name: str) -> Path | None:
    """A file inside a path-tracked package, found the way Julia finds the package."""

    import tomllib

    manifest = tomllib.loads((project / "Manifest.toml").read_text(encoding="utf-8"))
    entry = manifest.get("deps", {}).get(name, [{}])[0]
    source = entry.get("path")
    if not source:
        return None
    root = Path(source)
    if not root.is_absolute():
        root = project / root
    named = root / "src" / f"{name}.jl"
    if named.exists():
        return named
    return next(iter(sorted(root.rglob("*.jl"))), None)
