"""``jutul-agent doctor``: diagnose a workspace's setup.

One command a user can run (and paste the output of) when launch fails.
Each check prints PASS / WARN / FAIL with a one-line remediation, so the
common setup mistakes are obvious without reading a traceback.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from jutul_agent.display import (
    has_display,
    plotting_display_available,
    xvfb_opted_out,
)
from jutul_agent.interfaces.cli._helpers import (
    add_workspace_flags,
    known_packages_map,
)
from jutul_agent.julia.requirements import MIN_JULIA_VERSION, check_julia
from jutul_agent.models import key_env_var, provider_info, provider_of
from jutul_agent.paths import workspace_root
from jutul_agent.simulators import registry
from jutul_agent.workspace import (
    auto_detect_simulator,
    env_declares_warm_packages,
    env_template_drifted,
    load_workspace_config,
    resolve_julia_project,
)

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"

_MARK = {PASS: "[ok]  ", WARN: "[warn]", FAIL: "[FAIL]"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jutul-agent doctor",
        description="Check that this workspace is ready to run jutul-agent.",
    )
    parser.add_argument(
        "--sim",
        choices=registry.names(),
        required=False,
        help="Simulator to check against (default: workspace config / auto-detect).",
    )
    add_workspace_flags(parser)
    return parser


class _Report:
    def __init__(self) -> None:
        self.worst = PASS

    def line(self, status: str, label: str, detail: str = "", fix: str = "") -> None:
        if status == FAIL or (status == WARN and self.worst != FAIL):
            self.worst = status
        msg = f"{_MARK[status]} {label}"
        if detail:
            msg += f": {detail}"
        print(msg)
        if fix and status != PASS:
            print(f"        -> {fix}")


def run(args: argparse.Namespace) -> int:
    ws = workspace_root()
    config = load_workspace_config(ws)
    sim_name = args.sim or config.simulator or auto_detect_simulator(known_packages_map(), ws)

    report = _Report()
    print(f"jutul-agent doctor - workspace: {ws}\n")

    julia = _check_julia(report)
    _check_model_and_key(report, _resolve_model_id(config))
    sim_name = _check_simulator(report, sim_name)
    project = _check_julia_project(report, ws)
    _check_simulator_installed(report, project, sim_name)
    _check_env_template_current(report, ws, project, sim_name)
    _check_plotting_display(report)

    # Confirm Julia actually boots in this env; catches a broken/half-resolved
    # manifest before the agent starts the kernel. Only when the cheap checks
    # passed, else the error would just restate what we already reported.
    if julia.ok and project is not None:
        _check_julia_runs(report, project)
    else:
        report.line(WARN, "Julia runs in env", "skipped (fix the items above first)")

    print()
    if report.worst == FAIL:
        print("Setup has problems — fix the [FAIL] items above, then re-run `jutul-agent doctor`.")
        return 1
    if report.worst == WARN:
        print("Setup looks usable, but see the [warn] items above.")
        return 0
    print("All checks passed. You're ready to run `jutul-agent`.")
    return 0


def _check_julia(report: _Report):
    julia = check_julia()
    min_str = ".".join(str(n) for n in MIN_JULIA_VERSION)
    if not julia.found:
        report.line(
            FAIL,
            "Julia on PATH",
            "`julia` not found",
            f"Install Julia {min_str}+ via juliaup, then open a new terminal.",
        )
    elif julia.version is None:
        report.line(WARN, "Julia on PATH", julia.error or "version unknown", julia.path or "")
    elif not julia.version_ok:
        report.line(
            FAIL,
            "Julia version",
            f"{julia.version_str} (need {min_str}+)",
            f"juliaup add {min_str} && juliaup default {min_str}",
        )
    else:
        report.line(PASS, "Julia version", f"{julia.version_str} ({julia.path})")
    return julia


def _resolve_model_id(config) -> str:
    """The model this workspace will use (same precedence as the agent)."""
    # Lazy: builder pulls in deepagents, too heavy to import on every CLI call.
    from jutul_agent.models import resolve_model
    from jutul_agent.user_config import load_user_config

    return resolve_model(None, workspace_model=config.model, user_model=load_user_config().model)


def _check_model_and_key(report: _Report, model_id: str) -> None:
    """Report the resolved model and whether its provider's credential is ready."""
    info = provider_info(model_id)
    label = info.label if info else (provider_of(model_id) or "unknown provider")
    report.line(PASS, "Model", f"{model_id} ({label})")

    if info is not None and info.local:
        _check_ollama(report, model_id)
        return

    env_var = key_env_var(model_id)
    if env_var is None:
        report.line(
            WARN,
            "Provider API key",
            f"unknown provider for {model_id}",
            "Make sure the right langchain-<provider> package is installed and the key is set.",
        )
        return
    if os.environ.get(env_var):
        report.line(PASS, "Provider API key", f"{env_var} set")
    else:
        report.line(
            FAIL,
            "Provider API key",
            f"{env_var} not set for {label}",
            f"Set it with `jutul-agent key {provider_of(model_id) or '<provider>'}`, add it to "
            "your shell or .env, or pick a model in the app to be prompted for it.",
        )


def _check_ollama(report: _Report, model_id: str) -> None:
    import asyncio

    from jutul_agent import ollama_client
    from jutul_agent.models import is_ollama_cloud

    name = ollama_client.model_name(model_id)
    if not asyncio.run(ollama_client.is_reachable()):
        report.line(
            WARN,
            "Ollama server",
            f"not reachable at {ollama_client.host()}",
            "Start it with `ollama serve` (or install it from https://ollama.com).",
        )
        return
    if is_ollama_cloud(model_id):
        report.line(PASS, "Ollama model", f"{name} (hosted by Ollama; needs `ollama signin`)")
        return
    if not asyncio.run(ollama_client.is_installed(name)):
        report.line(
            WARN,
            "Ollama model",
            f"{name} not pulled",
            f"Pull it with `ollama pull {name}` (or select it in the TUI to pull in-app).",
        )
        return
    # The agent is tool-driven, so a model the daemon reports as tool-less can't run.
    if asyncio.run(ollama_client.supports_tools(name)):
        report.line(PASS, "Ollama model", f"{name} pulled (tools supported)")
    else:
        report.line(
            FAIL,
            "Ollama model",
            f"{name} doesn't support tool calling, which jutul-agent requires",
            f"Pick a tool-capable model, or update Ollama and re-pull if it should "
            f"support tools (`ollama show {name}` lists capabilities).",
        )


def _check_simulator(report: _Report, sim_name: str | None) -> str | None:
    if sim_name is None:
        report.line(
            FAIL,
            "Simulator",
            "not set and not auto-detected",
            "Run `jutul-agent init --sim <name>`. Known: " + ", ".join(registry.names()) + ".",
        )
        return None
    if sim_name not in registry.names():
        report.line(
            FAIL, "Simulator", f"unknown: {sim_name}", "Known: " + ", ".join(registry.names())
        )
        return None
    report.line(PASS, "Simulator", sim_name)
    return sim_name


def _check_julia_project(report: _Report, ws: Path) -> Path | None:
    project = resolve_julia_project(ws)
    proj_toml = project / "Project.toml"
    using_root = (ws / "Project.toml").resolve() == proj_toml.resolve()
    where = (
        "workspace root Project.toml" if using_root else "workspace-local .jutul-agent/julia-env"
    )
    if not proj_toml.exists():
        report.line(
            FAIL,
            "Julia project",
            f"no Project.toml at {project}",
            "Run `jutul-agent init --sim <name>` in this directory.",
        )
        return None
    detail = f"{project} ({where})"
    if using_root:
        report.line(
            WARN,
            "Julia project",
            detail,
            "A root Project.toml takes precedence over .jutul-agent/julia-env — "
            "make sure it includes the simulator package.",
        )
    else:
        report.line(PASS, "Julia project", detail)
    return project


def _check_simulator_installed(report: _Report, project: Path | None, sim_name: str | None) -> None:
    """Verify the simulator's package is resolved in the env's manifest.

    `_check_julia_runs` only proves Julia boots; a workspace whose
    Project lists the simulator but whose Manifest never resolved it still
    passes that check, then fails at runtime on `using <Sim>`. This closes
    that gap cheaply (a manifest read, no Julia subprocess).
    """

    if project is None or sim_name is None or sim_name not in registry.names():
        return
    from jutul_agent.simulators.env_setup import manifest_has_package

    adapter = registry.get(sim_name)
    pkg = adapter.primary_package
    if manifest_has_package(project, pkg):
        report.line(PASS, f"{pkg} resolved in env")
        return
    report.line(
        FAIL,
        f"{pkg} resolved in env",
        f"{pkg} is in Project.toml but not in Manifest.toml (env not instantiated)",
        f"Run `jutul-agent init --sim {sim_name}` to install it.",
    )


def _check_env_template_current(
    report: _Report, ws: Path, project: Path | None, sim_name: str | None
) -> None:
    """Warn when the managed env was built from an older simulator template.

    The matching launch-time check only warns on stderr mid-startup; surfacing
    it here too means a user debugging a stale env sees it in the one report
    they're told to paste. Only applies to a managed ``.jutul-agent/julia-env``
    (a user-owned root Project.toml is theirs), and only when a stamp exists.
    """

    if project is None or sim_name is None or sim_name not in registry.names():
        return
    if (ws / "Project.toml").resolve() == (project / "Project.toml").resolve():
        return  # user-owned root env
    if not env_declares_warm_packages(project):
        return

    template = registry.get(sim_name).julia_env_template_path
    if env_template_drifted(project, template):
        report.line(
            WARN,
            "Env template current",
            "built from an older template than the installed jutul-agent",
            f"Rebuild it with `jutul-agent init --sim {sim_name} --force`.",
        )
    else:
        report.line(PASS, "Env template current")


def _check_plotting_display(report: _Report) -> None:
    """Warn when GLMakie has no display here, so plotting will be unavailable.

    Plotting is optional: simulation, eval, and the file tools work without it.
    So this only ever WARNs; a real display (any desktop OS, or X/Wayland on
    Linux) passes; headless Linux passes when ``xvfb-run`` is on PATH (the kernel
    wraps the Julia process in it). It warns when headless Linux has neither, with the
    one command that fixes it.
    """

    if plotting_display_available():
        detail = (
            "display detected" if has_display() else "headless; xvfb-run found (renders to file)"
        )
        report.line(PASS, "Plotting display (GLMakie)", detail)
        return
    # Only reachable on headless Linux (desktop OSes always report a display).
    if xvfb_opted_out():
        report.line(
            WARN,
            "Plotting display (GLMakie)",
            "headless and JUTUL_AGENT_NO_XVFB is set, so GLMakie plotting is disabled",
            "Unset JUTUL_AGENT_NO_XVFB (and install xvfb) to enable plotting; "
            "simulation works without it.",
        )
        return
    report.line(
        WARN,
        "Plotting display (GLMakie)",
        "headless (no display) and xvfb-run not found, so GLMakie plotting is unavailable",
        "Install xvfb (e.g. `sudo apt-get install -y xvfb`) to enable plotting; "
        "simulation works without it.",
    )


def _check_julia_runs(report: _Report, project: Path) -> None:
    """Confirm Julia boots cleanly in this env (a trivial eval, no packages).

    The kernel's server is stdlib-only, so the real failure mode is a broken or
    half-resolved manifest. A trivial ``print(1 + 1)`` in the project surfaces it
    with the same actionable rebuild hint.
    """

    from jutul_agent.simulators.env_setup import julia_probe

    ok, tail = julia_probe(project)
    if ok:
        report.line(PASS, "Julia runs in env")
        return
    report.line(
        FAIL,
        "Julia runs in env",
        "Julia exited with an error in this env",
        "Run `jutul-agent init --sim <name> --force` to rebuild the env.",
    )
    if tail:
        print("        Julia said:", file=sys.stderr)
        for line in tail.splitlines():
            print(f"          {line}", file=sys.stderr)
