"""The split JutulAgent Julia runtime must stay consistent across envs.

The runtime is two kinds of package:

* One shared, simulator-agnostic ``JutulAgent`` package with a *single* source in
  the repo (``julia_runtime/``); env_setup copies it into each workspace env at
  bootstrap. There is no per-env copy to drift.
* One per-simulator ``JutulAgent<Sim>`` warm package per env (named by the
  adapter's ``warm_package``), carrying that simulator's backend-aware solve/plot
  bake.

These guard the structure: the shared package exists once, every env declares both
packages, no env ships a stale copy of the shared one, and each warm package does
the load-bearing ``@recompile_invalidations`` trick.
"""

from __future__ import annotations

import inspect
import re
import tomllib

from jutul_agent.agent import plot_julia_src
from jutul_agent.simulators import registry
from jutul_agent.workspace import SHARED_JULIA_PACKAGE_DIRNAME, shared_julia_package_path

_SHARED = "JutulAgent"
_SHARED_CORE_FILES = ("src/JutulAgent.jl", "src/plots.jl", "src/ensemble.jl")


def _plot_path_backends() -> set[str]:
    """The Makie backends the plot tool loads into a session, read from its source.

    Read rather than listed, so the invariant below tracks the plot path instead of
    a copy of it that can go stale. Only the Julia ``import`` statements the module
    emits count, so that merely naming a backend in prose does not oblige every warm
    package to depend on it. ``Makie`` itself is excluded: the backends carry it, and
    no package here depends on it directly.
    """

    src = inspect.getsource(plot_julia_src)
    imported = {
        name.strip()
        for statement in re.findall(r"import\s+([\w,\s]+)", src)
        for name in statement.split(",")
    }
    return {name for name in imported if name.endswith("Makie") and name != "Makie"}


def test_shared_package_has_single_source_with_core_files() -> None:
    pkg = shared_julia_package_path()
    assert pkg.is_dir(), f"shared {_SHARED} package missing at {pkg}"
    assert (pkg / "Project.toml").exists()
    for rel in _SHARED_CORE_FILES:
        assert (pkg / rel).exists(), f"shared {_SHARED} missing {rel}"


def test_no_env_ships_a_copy_of_the_shared_package() -> None:
    # The shared package is copied in at bootstrap, not committed per env — so a
    # stale per-env copy can never drift from the single source.
    for name in registry.names():
        env = registry.get(name).julia_env_template_path
        assert not (env / SHARED_JULIA_PACKAGE_DIRNAME).exists(), (
            f"{name}: env template still ships a copy of {_SHARED}; it must be "
            "synced from julia_runtime/ at bootstrap instead"
        )


def test_every_env_has_its_warm_package() -> None:
    for name in registry.names():
        adapter = registry.get(name)
        pkg_name = adapter.warm_package
        assert pkg_name, f"{name}: adapter has no warm_package"
        pkg = adapter.julia_env_template_path / pkg_name
        assert (pkg / "Project.toml").exists(), f"{name}: missing {pkg_name}/Project.toml"
        assert (pkg / "src" / f"{pkg_name}.jl").exists(), f"{name}: missing {pkg_name}/src module"


def test_warm_packages_recompile_glmakie_invalidations() -> None:
    # The load-bearing trick: each warm package loads GLMakie under
    # @recompile_invalidations so the simulator's solver is recompiled GLMakie-aware.
    for name in registry.names():
        adapter = registry.get(name)
        pkg = adapter.julia_env_template_path / adapter.warm_package
        src = pkg / "src" / f"{adapter.warm_package}.jl"
        text = src.read_text(encoding="utf-8")
        assert "@recompile_invalidations" in text, f"{name}: no @recompile_invalidations"
        assert "using GLMakie" in text, f"{name}: warm package does not load GLMakie"
        assert "_warm_solve" in text, f"{name}: warm package defines no _warm_solve"


def test_warm_packages_load_every_backend_the_plot_path_does() -> None:
    # Loading a Makie backend invalidates code specialised against the backends
    # already loaded, and that reaches past plotting into the solver. A backend the
    # session loads *after* the warm package therefore discards part of the bake,
    # solve included, and the only symptom is that everything is slow again. So the
    # warm packages have to load them all, and the shared package too, since it
    # loads first and would otherwise be invalidated by the per-simulator one.
    backends = _plot_path_backends()
    assert backends, "no Makie backends found in the plot path; the regex has gone stale"

    projects = [("shared " + _SHARED, shared_julia_package_path() / "Project.toml")]
    for name in registry.names():
        adapter = registry.get(name)
        projects.append(
            (name, adapter.julia_env_template_path / adapter.warm_package / "Project.toml")
        )

    for label, proj in projects:
        deps = tomllib.loads(proj.read_text(encoding="utf-8")).get("deps") or {}
        missing = sorted(backends - set(deps))
        assert not missing, (
            f"{label}: warm package does not load {', '.join(missing)}, which the plot "
            "path does; whatever it bakes will be invalidated when the session loads them"
        )


def test_warm_packages_expose_the_workload_as_a_function() -> None:
    # @compile_workload swallows exceptions by design, so a workload that has drifted
    # out of step with its simulator bakes nothing and says nothing. `_warm()` is the
    # same workload without the swallow, and tests/test_simulators_smoke.py runs it
    # against a live env.
    for name in registry.names():
        adapter = registry.get(name)
        pkg = adapter.julia_env_template_path / adapter.warm_package
        text = (pkg / "src" / f"{adapter.warm_package}.jl").read_text(encoding="utf-8")
        assert "function _warm()" in text, f"{name}: warm package defines no _warm()"
        assert "_warm()\n        catch" in text, (
            f"{name}: @compile_workload should call _warm() so the baked workload and "
            "the tested one cannot drift apart"
        )


def test_every_env_declares_both_runtime_packages() -> None:
    for name in registry.names():
        adapter = registry.get(name)
        proj = adapter.julia_env_template_path / "Project.toml"
        data = tomllib.loads(proj.read_text(encoding="utf-8"))
        deps = data.get("deps") or {}
        sources = data.get("sources") or {}
        for pkg in (_SHARED, adapter.warm_package):
            assert pkg in deps, f"{name}: {pkg} not in [deps]"
            assert pkg in sources, f"{name}: {pkg} not in [sources]"
