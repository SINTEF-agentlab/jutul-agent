"""Set up and reconcile a workspace's Julia environment.

Each simulator ships a template env at ``simulators/<name>/julia_env/``
(already declaring the simulator's package dependencies). Bootstrap = copy
the template into the workspace, optionally ``Pkg.develop`` a user-provided
source path, optionally ``Pkg.instantiate`` to precompile.
:func:`prepare_workspace_env` is the launch-time entry point that makes a
missing or drifted env ready, best-effort, before the kernel starts.

A workspace that already has its own root ``Project.toml`` is left alone:
the user owns that env, and we only run dev and instantiate on request.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tomllib
from collections.abc import Sequence
from pathlib import Path

from jutul_agent.simulators.base import SimulatorAdapter
from jutul_agent.workspace import (
    WorkspaceBootstrapError,
    bootstrap_julia_env,
    ensure_env_template_stamp,
    env_declares_warm_packages,
    env_precompile_is_current,
    env_template_drifted,
    mark_env_precompiled,
    mark_warm_source,
    recopy_warm_sources,
    resolve_julia_project,
    sync_julia_env_with_template,
    sync_julia_project_with_dependencies,
    warm_source_is_current,
    workspace_is_simulator_source,
    write_env_template_stamp,
)


class EnvSetupError(RuntimeError):
    pass


def is_workspace_env_ready(workspace: Path | None = None) -> bool:
    """A workspace env is ready when its resolved project has a Project.toml."""

    project = resolve_julia_project(workspace)
    return (project / "Project.toml").exists()


def manifest_has_package(julia_project: Path, package: str) -> bool:
    """True if the env's ``Manifest.toml`` actually resolves ``package``.

    A ``Project.toml`` can list a dependency that the manifest never resolved
    (deps edited, or a template merged, without a follow-up ``Pkg.resolve`` /
    ``Pkg.instantiate``). Such a package is *declared* but not installed, and
    ``using <package>`` then fails at runtime with "is required but does not
    seem to be installed", even though ``jutul-agent doctor``'s Julia-runs check
    passes. Inspecting the manifest catches that before launch.
    """

    manifest = julia_project / "Manifest.toml"
    if not manifest.exists():
        return False
    try:
        data = tomllib.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return False
    deps = data.get("deps")
    if isinstance(deps, dict):  # manifest format 2.0: [[deps.Pkg]]
        return package in deps
    # format 1.0: package sections are top-level tables
    return package in data


def project_has_package(julia_project: Path, package: str) -> bool:
    """True if ``package`` is a direct dependency in the env's Project.toml."""

    proj = julia_project / "Project.toml"
    if not proj.exists():
        return False
    try:
        data = tomllib.loads(proj.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return False
    return package in (data.get("deps") or {})


# One tab-separated "JPKG" line per resolved package: name, on-disk source dir,
# and whether it is a `Pkg.develop` checkout (writable) rather than a registry
# install. Run through a Julia subprocess to resolve installed sources without
# loading them; the JPKG tag lets the parser skip any unrelated output
# interleaved on stdout.
ENUMERATE_PACKAGES_CODE = (
    "import Pkg\n"
    "for (_u, _i) in Pkg.dependencies()\n"
    "    _i.source === nothing && continue\n"
    '    println("JPKG\\t", _i.name, "\\t", _i.source, "\\t", _i.is_tracking_path ? 1 : 0)\n'
    "end\n"
)


def parse_enumerated_packages(output: str) -> dict[str, tuple[Path, bool]]:
    """Parse ``ENUMERATE_PACKAGES_CODE`` output to ``{name: (source_dir, is_dev)}``.

    Lines without the JPKG tag and packages whose source dir no longer exists
    are skipped.
    """

    sources: dict[str, tuple[Path, bool]] = {}
    for line in output.splitlines():
        parts = line.split("\t")
        if len(parts) != 4 or parts[0] != "JPKG":
            continue
        _, name, path, is_dev = (p.strip() for p in parts)
        candidate = Path(path)
        if name and candidate.is_dir():
            sources[name] = (candidate, is_dev == "1")
    return sources


def resolve_env_package_sources(julia_project: Path) -> dict[str, tuple[Path, bool]]:
    """Source dir + dev flag of every package the env resolves.

    Enumerates ``Pkg.dependencies()`` (reads the manifest, no loading/compiling)
    so the agent can browse the simulator, its dependencies, and anything it
    later installs at their real ``pkgdir`` paths. Returns ``{name: (source_dir,
    is_dev)}`` where ``is_dev`` marks a ``Pkg.develop`` checkout (writable).
    Best-effort: an unresolved manifest, missing Julia, or a vanished path
    yields ``{}``.
    """

    if shutil.which("julia") is None:
        return {}
    try:
        result = subprocess.run(
            [
                "julia",
                f"--project={julia_project}",
                "--startup-file=no",
                "-e",
                ENUMERATE_PACKAGES_CODE,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}
    if result.returncode != 0:
        return {}
    return parse_enumerated_packages(result.stdout)


# Install deps without precompiling: auto-precompile is off so one package's
# precompile error can't abort the download/resolve step.
_INSTANTIATE = 'withenv("JULIA_PKG_PRECOMPILE_AUTO" => "0") do; Pkg.instantiate(); end'
_UPDATE = 'withenv("JULIA_PKG_PRECOMPILE_AUTO" => "0") do; Pkg.update(); end'
# A registry refresh needs the network; failing it must not block offline use,
# where the local registry copy is the freshest knowledge available anyway.
_REGISTRY_UPDATE = (
    "try; Pkg.Registry.update(); catch e; "
    '@warn "Registry update failed (offline?); resolving against the local copy" '
    "exception=e; end"
)
# Precompile everything that can be, best-effort. GLMakie is a default dep but can
# fail to precompile on a headless box with no GL/display (Makie issue #2791); that
# must not break the env, so we swallow the rest (plotting then errors at use).
_PRECOMPILE = (
    "try; Pkg.precompile(); catch e; "
    '@warn "Some packages failed to precompile (continuing; GL plotting may be '
    'unavailable here)" exception=e; end'
)
_RESILIENT_INSTANTIATE = [_INSTANTIATE, _PRECOMPILE]


def bootstrap_workspace(
    adapter: SimulatorAdapter,
    *,
    workspace: Path | None = None,
    source_path: Path | None = None,
    precompile: bool = False,
    force: bool = False,
) -> Path:
    """Prepare the workspace's Julia env for ``adapter``.

    Steps:
      1. If the workspace lacks both a root ``Project.toml`` and a
         workspace-local env, copy the template from the install.
         With ``force``, replace an existing workspace-local env first.
      2. If ``source_path`` is set, ``Pkg.develop(path=source_path)`` in
         the workspace env (idempotent; safe to re-run).
      3. If ``precompile``, ``Pkg.instantiate()`` then ``Pkg.precompile()``. The
         plotting stack is warmed by the env's ``JutulAgent`` package,
         whose ``@compile_workload`` bakes the Makie save path into the
         precompile cache (run under xvfb here so the GL bake has a context).

    Returns the path to the resolved Julia project.
    """

    try:
        bootstrap_julia_env(adapter.julia_env_template_path, workspace=workspace, force=force)
    except WorkspaceBootstrapError as exc:
        raise EnvSetupError(str(exc)) from exc

    project = resolve_julia_project(workspace)

    # If the workspace itself is the simulator source, skip Pkg.develop:
    # the user is editing the package in place.
    is_source = workspace_is_simulator_source(adapter.primary_package, workspace)

    cmds: list[str] = ["using Pkg"]
    if source_path is not None and not is_source:
        cmds.append(f'Pkg.develop(path=raw"{source_path}")')
    if precompile:
        cmds.extend(_RESILIENT_INSTANTIATE)

    if len(cmds) > 1:
        _run_pkg(project, cmds)
    if precompile:
        # Catch a broken/half-resolved manifest here (Julia fails to boot in the
        # env) instead of at launch. The kernel server is stdlib-only, so a
        # trivial eval is the right probe.
        verify_julia_runs(project)
        mark_env_precompiled(project)  # so launch skips a redundant bake

    return project


def precompile_env(project: Path) -> None:
    """Instantiate and precompile the env in place; bakes the warm-up packages.

    Idempotent and quick when the env is already precompiled. Streams Julia's
    progress (not captured) so a real bake reads as work in progress, not a hang.
    Precompile is best-effort (a headless GLMakie failure is warned, not fatal);
    a hard instantiate failure raises ``EnvSetupError``.
    """
    _run_pkg(project, ["using Pkg", _INSTANTIATE, _PRECOMPILE], echo=False)


def verify_julia_runs(project: Path) -> None:
    """Confirm Julia boots cleanly in ``project`` (a trivial eval, no packages).

    The kernel's server is stdlib-only, so the failure mode is a broken or
    half-resolved manifest. Surfacing it here beats a baffling crash at launch.
    Raises ``EnvSetupError`` on failure.
    """

    print("Verifying Julia runs in the env...", flush=True)
    _run_pkg(project, ["print(1 + 1)"])


def resolve_and_instantiate(
    project: Path, *, precompile: bool = True, capture: bool = False
) -> None:
    """Re-resolve the manifest and install deps.

    Plain ``Pkg.instantiate`` errors if a dep appears in Project.toml but
    not the Manifest.toml (e.g. after auto-syncing new deps in). Running
    ``Pkg.resolve`` first updates the manifest from Project.toml before
    install.

    ``Pkg.resolve`` assumes the General registry already exists and fails with
    "no registries have been installed" on a fresh Julia depot (unlike
    ``Pkg.instantiate``, ``resolve`` does not bootstrap it). So install General
    first when none is reachable.

    ``precompile=False`` installs without the (potentially minutes-long) precompile
    bake, leaving it for ``init``; used at launch so a self-healing
    sync doesn't block startup. ``capture=True`` hides Julia's output and folds a
    failure into a short ``EnvSetupError`` instead of dumping a backtrace.
    """

    cmds = [
        "using Pkg",
        'isempty(Pkg.Registry.reachable_registries()) && Pkg.Registry.add("General")',
        "Pkg.resolve()",
        _INSTANTIATE,
    ]
    if precompile:
        cmds.append(_PRECOMPILE)
    _run_pkg(project, cmds, capture=capture)


def update_env(project: Path) -> None:
    """Pull every dependency to the latest registry-compatible version.

    Envs carry no version pins, so a fresh instantiate tracks upstream while
    an existing manifest freezes whatever the registry served at build time.
    ``Pkg.update`` re-resolves the manifest the way a fresh build would,
    which is what keeps a cached env aligned with current releases. A failed
    resolve raises ``EnvSetupError``: a stale env should be a loud stop, not
    a silent result. Precompile is tolerant the same way the bootstrap is (a
    headless GL failure warns), and the marker is refreshed so later
    reconciles stay cheap.
    """

    _run_pkg(project, ["using Pkg", _REGISTRY_UPDATE, _UPDATE, _PRECOMPILE])
    mark_env_precompiled(project)


def _run_pkg(project: Path, cmds: list[str], *, capture: bool = False, echo: bool = True) -> None:
    if shutil.which("julia") is None:
        raise EnvSetupError("`julia` is not on PATH")

    argv = [
        "julia",
        f"--project={project}",
        "--startup-file=no",
        "-e",
        "; ".join(cmds),
    ]
    # On headless Linux, wrap in xvfb so the GL-dependent precompile has an
    # OpenGL context: the @compile_workload bakes save a real GLMakie figure
    # (the shared JutulAgent package, and each JutulAgent<Sim> package's
    # _warm_plot). This is the same wrap the runtime uses. A real display, or
    # any non-Linux OS, skips it.
    from jutul_agent.display import should_wrap_xvfb

    if should_wrap_xvfb():
        argv = ["xvfb-run", "-a", "-s", "-screen 0 1280x1024x24", *argv]

    if capture:
        # Best-effort callers (the launch-time self-heal) want a quiet run and a
        # short error, not a page of Julia backtrace on the terminal.
        result = subprocess.run(argv, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            tail = "\n".join((result.stdout + result.stderr).strip().splitlines()[-12:])
            raise EnvSetupError(f"Julia exited with code {result.returncode}:\n{tail}")
        return

    if echo:
        print(f"$ {' '.join(argv)}")
    result = subprocess.run(argv, check=False)
    if result.returncode != 0:
        raise EnvSetupError(f"Julia exited with code {result.returncode}; see output above")


# ---------------------------------------------------------------------------
# Launch-time preparation.


def prepare_workspace_env(
    adapter: SimulatorAdapter,
    *,
    workspace: Path,
    julia_project: Path,
    sim_name: str | None = None,
    dependencies: Sequence[Path] | None = None,
) -> None:
    """Make the workspace's Julia env ready for ``adapter`` before launch.

    A missing env is bootstrapped from the template (raises ``EnvSetupError``
    on failure). An existing one is reconciled best-effort, printing progress:

    - A managed env built for a *different* simulator can't be merged; e.g.
      BattMo and JutulDarcy pin incompatible shared deps; so it is rebuilt
      from the active template. A user-owned root ``Project.toml`` is never
      touched.
    - New template deps are synced in (self-healing after a ``git pull``),
      rolling back on an install conflict.
    - A simulator that is declared but never resolved is installed (the
      "doctor is happy but ``using <Sim>`` fails" trap).
    - The warm-up precompile is baked when something changed, so the first
      solve is fast without paying the bake on every launch.
    """

    if not is_workspace_env_ready(workspace):
        # Implicit auto-bootstrap (without dev or precompile; those are init's job).
        bootstrap_workspace(adapter, workspace=workspace)
    elif not (workspace / "Project.toml").exists():
        foreign = _foreign_simulator(julia_project, adapter)
        if foreign is not None:
            _rebuild_managed_env(adapter, workspace, sim_name, reason=f"was built for {foreign}")
            return

    _sync_workspace_env(adapter, workspace, julia_project, sim_name)
    _sync_project_dependencies(julia_project, dependencies)
    _refresh_warm_sources(adapter, workspace, julia_project, sim_name)

    _ensure_simulator_installed(adapter, workspace, julia_project, sim_name)
    _ensure_env_warmed(workspace, julia_project, sim_name)
    _reconcile_env_template(adapter, workspace, julia_project, sim_name)


def _foreign_simulator(julia_project: Path, adapter: SimulatorAdapter) -> str | None:
    """Display name of another simulator whose package this env declares.

    Shared Jutul-stack packages (e.g. JutulDarcy for Fimbul) are in the active
    adapter's ``package_imports`` and don't count; only a different
    simulator's primary package marks the env as built for something else.
    """

    from jutul_agent.simulators import registry

    for name in registry.names():
        other = registry.get(name)
        if other.name == adapter.name or other.primary_package in adapter.package_imports:
            continue
        if project_has_package(julia_project, other.primary_package):
            return other.display_name
    return None


def _rebuild_managed_env(
    adapter: SimulatorAdapter, ws: Path, sim_name: str | None, *, reason: str
) -> None:
    """Replace the managed workspace env with the active simulator's template."""

    print(
        f"Workspace Julia env {reason}; rebuilding it for {adapter.display_name} "
        "from the template (one-time, can take a few minutes)...",
        flush=True,
    )
    try:
        bootstrap_workspace(adapter, workspace=ws, force=True, precompile=True)
    except EnvSetupError as exc:
        _warn_rebuild(adapter.primary_package, sim_name, exc)


def _warn_rebuild(pkg: str, sim_name: str | None, exc: Exception) -> None:
    rebuild = "jutul-agent init --force"
    if sim_name:
        rebuild += f" --sim {sim_name}"
    print(
        f"warning: could not prepare {pkg} ({exc}).\n"
        f"         The agent may fail to load it. Rebuild the env with:\n"
        f"             {rebuild}",
        file=sys.stderr,
    )


def _sync_workspace_env(
    adapter: SimulatorAdapter,
    ws: Path,
    julia_project: Path,
    sim_name: str | None,
) -> None:
    """Bring the workspace env up to date with its simulator template, then install.

    Self-healing: when an upstream change adds packages to the template (e.g. the
    JutulAgent warm-up packages), ``sync_julia_env_with_template`` brings the deps,
    plus the ``[sources]`` paths and package directories they need, into the env,
    so a plain ``git pull`` + launch keeps working without a manual rebuild. We only
    resolve and instantiate here so the install is quick; the warm-up bake runs
    afterwards in :func:`_ensure_env_warmed` (with visible progress).

    Best-effort: if the install fails (e.g. the new deps conflict with what is
    pinned), we roll the Project.toml back so the env is no worse than before and
    point at the clean-rebuild command. Either way we proceed to launch.
    """

    project_toml = julia_project / "Project.toml"
    before = project_toml.read_text(encoding="utf-8") if project_toml.exists() else None

    try:
        added = sync_julia_env_with_template(
            adapter.julia_env_template_path,
            workspace=ws,
        )
    except Exception as exc:
        print(f"warning: env sync failed: {exc}", file=sys.stderr)
        return

    if not added:
        return

    print(f"Updating workspace env with {', '.join(added)} (added upstream)...", flush=True)
    try:
        resolve_and_instantiate(julia_project, precompile=False, capture=True)
        # The env now matches the current template; refresh the stamp so the
        # drift check below doesn't then flag the very change we just healed.
        write_env_template_stamp(julia_project, adapter.julia_env_template_path)
    except EnvSetupError as exc:
        if before is not None:
            project_toml.write_text(before, encoding="utf-8")
        rebuild = "jutul-agent init --force"
        if sim_name:
            rebuild += f" --sim {sim_name}"
        print(
            f"warning: could not install {', '.join(added)} — rolled the env back, so it "
            "still works as before and the agent will start normally.\n"
            f"         To rebuild it cleanly, run: {rebuild}\n"
            f"         (cause: {exc})",
            file=sys.stderr,
        )


def _sync_project_dependencies(julia_project: Path, dependencies: Sequence[Path] | None) -> None:
    """Add capability-provided deps to the active Julia project."""

    try:
        added = sync_julia_project_with_dependencies(julia_project, dependencies)
    except Exception as exc:
        print(f"warning: capability dependency sync failed: {exc}", file=sys.stderr)
        return

    if added:
        print(
            f"Updating Julia project with {', '.join(added)} (added by capability)...",
            flush=True,
            file=sys.stderr,
        )
        try:
            resolve_and_instantiate(julia_project, precompile=False, capture=True)
        except EnvSetupError as exc:
            print(
                "Warning: could not resolve and instantiate the "
                + f"capability dependencies: ({exc}). ",
                file=sys.stderr,
            )
            return


def _refresh_warm_sources(
    adapter: SimulatorAdapter,
    ws: Path,
    julia_project: Path,
    sim_name: str | None,
) -> None:
    """Re-copy the in-env JutulAgent runtime when the install's source has changed.

    The warm packages are dev-pathed copies, only laid down at bootstrap, so after
    a jutul-agent update an existing managed env keeps the old JutulAgent until a
    manual rebuild. We detect that by fingerprint, re-copy the packages, and
    re-resolve so a new in-package dependency installs; the bake re-runs in
    :func:`_ensure_env_warmed` because the re-copy dropped the precompile marker.
    The env is marked current only after a successful resolve, so a conflict (which
    we warn about and launch through) retries next launch.
    """

    template = adapter.julia_env_template_path
    if warm_source_is_current(julia_project, template):
        return
    if not recopy_warm_sources(julia_project, template):
        return  # user-owned env (no [sources]); nothing to refresh

    print("Updating the in-env JutulAgent runtime (jutul-agent was updated)…", flush=True)
    try:
        resolve_and_instantiate(julia_project, precompile=False, capture=True)
    except EnvSetupError as exc:
        rebuild = "jutul-agent init --force"
        if sim_name:
            rebuild += f" --sim {sim_name}"
        print(
            f"warning: could not refresh the JutulAgent runtime ({exc}). The agent "
            f"will start on the previous copy; rebuild cleanly with: {rebuild}",
            file=sys.stderr,
        )
        return
    mark_warm_source(julia_project, template)


def _ensure_simulator_installed(
    adapter: SimulatorAdapter, ws: Path, julia_project: Path, sim_name: str | None
) -> None:
    """Install the simulator package if the env declares but never resolved it.

    Catches the "`jutul-agent doctor` is happy but `using <Sim>` fails" trap:
    the Project lists the package but the Manifest never resolved it, so it
    loads neither at startup nor in the agent's first call. Cheap when the env
    is healthy (a manifest read); only pays the install cost when needed. If
    the resolve itself fails on a managed env (a broken or conflicted manifest),
    rebuild it from the template. Best-effort; on failure we warn and launch.
    """

    pkg = adapter.primary_package
    if manifest_has_package(julia_project, pkg):
        return

    print(
        f"Workspace Julia env has not resolved {pkg} yet — installing "
        "(one-time, can take a few minutes)...",
        flush=True,
    )
    try:
        # Resolve + install only; the warm-up bake is _ensure_env_warmed's job.
        resolve_and_instantiate(julia_project, precompile=False)
    except EnvSetupError as exc:
        # A user-owned root env is theirs to fix; only rebuild the managed env.
        if not (ws / "Project.toml").exists():
            _rebuild_managed_env(adapter, ws, sim_name, reason="could not be resolved")
            return
        _warn_rebuild(pkg, sim_name, exc)


def _reconcile_env_template(
    adapter: SimulatorAdapter, ws: Path, julia_project: Path, sim_name: str | None
) -> None:
    """Warn when the managed env was built from an older template than the install.

    The other reconcile steps self-heal *added* deps; this catches the changes
    they can't (a dep dropped, a compat tightened, a ``[sources]`` path moved)
    after an upgrade, pointing at the one-command rebuild the docs prescribe.
    Never touches a user-owned root env, and stays quiet for an env whose stamp
    already matches — baselining a stamp-less (pre-feature) env instead of nagging.
    """

    if (ws / "Project.toml").exists() or not env_declares_warm_packages(julia_project):
        return

    template = adapter.julia_env_template_path
    if not env_template_drifted(julia_project, template):
        ensure_env_template_stamp(julia_project, template)
        return

    rebuild = "jutul-agent init --force"
    if sim_name:
        rebuild += f" --sim {sim_name}"
    print(
        "warning: this workspace's Julia env was built from an older "
        f"{adapter.display_name} template than the installed jutul-agent. It will "
        "still run, but to pick up the new template rebuild it with:\n"
        f"             {rebuild}",
        file=sys.stderr,
    )


def _ensure_env_warmed(ws: Path, julia_project: Path, sim_name: str | None) -> None:
    """Precompile the managed env before launch, but only when something changed.

    The per-simulator ``JutulAgent<Sim>`` package's precompile runs the
    ``@recompile_invalidations`` solve that makes the first real solve fast
    (seconds instead of ~30 s). When the env changed (a fresh install, or new deps
    pulled in) we bake it here; blocking launch until it's ready, and streaming
    Julia's progress so it reads as work in progress, not a silent hang in the first
    eval. When nothing changed, a marker check skips it with no Julia process, so a
    plain launch stays fast. Best-effort: on failure the agent still starts.
    """

    # Only the managed env carries the warm-up packages; a user-owned env is theirs.
    if (ws / "Project.toml").exists() or not env_declares_warm_packages(julia_project):
        return
    if env_precompile_is_current(julia_project):
        return  # nothing changed since the last bake; skip without spawning Julia

    print(
        "Precompiling the Julia env (one-time after a change; can take a few minutes)...",
        flush=True,
    )
    try:
        precompile_env(julia_project)
    except EnvSetupError as exc:
        retry = "jutul-agent init"
        if sim_name:
            retry += f" --sim {sim_name}"
        print(
            f"warning: env precompile did not finish ({exc}); the agent will start, "
            f"but the first solve may be slow. Retry with: {retry}",
            file=sys.stderr,
        )
        return
    mark_env_precompiled(julia_project)
