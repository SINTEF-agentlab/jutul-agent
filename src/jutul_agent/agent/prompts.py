"""Build the per-session system prompt.

Everything jutul-agent adds to the model's system prompt is assembled here,
deterministically from the ``SimulatorAdapter`` and the session's display
capability; Deep Agents appends its built-in BASE prompt after it. Each
behavioral rule is stated exactly once; the per-provider ``HarnessProfile``
registered in ``agent.builder`` carries no prompt text, only the
general-purpose-subagent disable.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from jutul_agent.simulators.base import SimulatorAdapter


def assemble_session_prompt(
    adapter: SimulatorAdapter,
    *,
    open_windows: bool = True,
    resumed: bool = False,
    workspace: Path | None = None,
    primary_source: str | Path | None = None,
    surface: str = "tui",
    extra_fragments: Sequence[str] = (),
) -> str:
    sections = [
        f"Active simulator: {adapter.display_name} ({adapter.name}).",
        "Primary Julia packages: " + ", ".join(adapter.package_imports) + ".",
    ]
    if workspace is not None:
        # The agent's real working directory. Stated up front so the file tools
        # (whose descriptions demand absolute paths), `execute`, and the REPL all
        # build correct paths without a pwd/ls round-trip to discover it. Omitted
        # when assembling the prompt for the RunConfig hash (workspace=None) so a
        # per-session path does not destabilize the attribution hash.
        sections.append(
            f"Working directory: {workspace}\nThis is the user's workspace and the "
            "working directory for the file tools, `execute`, and the Julia REPL."
        )
    if primary_source is not None:
        # The simulator package's source path, already resolved at session start.
        # Stated so the agent reads it straight away instead of running
        # `using <Sim>; pkgdir(<Sim>)`, which loads the package (seconds, the first
        # time) only to discover a path we already have. Omitted for the attribution
        # hash (a per-install path would destabilize it).
        sections.append(_source_note(adapter, primary_source))
    sections += [_tool_guide(adapter), _ground_rules()]
    # The web surface renders plots/reports in the app, so its surface note is the
    # complete display guidance; the terminal display note would contradict it
    # (it forbids claiming interactivity, which the browser does provide).
    if surface == "web":
        sections.append(_surface_note(surface))
    else:
        sections.append(_display_note(open_windows))
    if resumed:
        sections.append(_resume_note())
    hints = adapter.domain_hints.strip()
    if hints:
        sections.append("Simulator hints:\n" + hints)
    sections += [fragment.strip() for fragment in extra_fragments if fragment.strip()]
    return "\n\n".join(sections) + "\n"


def _surface_note(surface: str) -> str | None:
    """Tell the agent which front end it is driving, when that changes behaviour.

    The terminal is the default and needs no note. On the web the agent talks to
    an application that can show richer output and expose interface controls, so
    say so; the specific controls arrive as extension tools.
    """

    if surface == "web":
        return (
            "Interface: you are driving a web application, not a terminal. Your "
            "replies, plots, and reports appear in that app, and the user reads them "
            "there; there is no desktop window.\n"
            "Plots: make every figure with `plot_julia` — build one with Makie (e.g. "
            "`heatmap`, `lines`, `surface`, `contourf`) or call the simulator's native "
            "interactive plotter. It renders as an interactive figure in a side panel "
            "the user can rotate, zoom, and pan, and it stays pinned there so you can "
            "refer back to it. When a native plotter would otherwise open a desktop "
            "window (e.g. a reservoir or results viewer), pass `window = false` so "
            "it returns the figure for the browser. A figure built in `run_julia` is "
            "shown to no one — always go through `plot_julia`.\n"
            "Reports: `write_report` opens a written report in that same side panel "
            "(not a desktop window); use it when the user wants a written summary.\n"
            "Some tools may update the application's interface directly; use them when "
            "the task calls for it."
        )
    return None


def _resume_note() -> str:
    """Tell the agent the conversation outlived the Julia process.

    Without it the agent assumes variables and loaded packages from earlier
    turns still exist and walks into UndefVarErrors instead of re-running
    its setup.
    """

    return (
        "Session continuity: this conversation was resumed from an earlier "
        "session. The chat history is restored, but the Julia REPL restarted "
        "with the process: no variables, loaded packages, or in-memory results "
        "from earlier turns exist now. Re-run the necessary setup before "
        "building on earlier results. Files written to the workspace and "
        "earlier artifacts are still on disk."
    )


def _display_note(open_windows: bool) -> str:
    """Tell the agent, for *this* session, whether a live plot window can appear.

    Without it the agent can't know it's headless and will wrongly tell the user a
    window opened (e.g. after a native ``plot_well_results`` call). When no window
    can show, steer it to ``plot_julia`` (which still renders a PNG) and away from
    claiming interactivity the user can't see.
    """

    if open_windows:
        return (
            "Display: live plot windows are available this session. `plot_julia` "
            "opens an interactive Makie window the user can rotate/zoom/step, and "
            "also saves a PNG."
        )
    return (
        "Display: this session is HEADLESS, so no on-screen window can appear. "
        "`plot_julia` still renders and saves a PNG (the user sees it in the "
        "transcript/report), so use it for every figure. Native interactive viewers "
        "(`plot_well_results`, `plot_reservoir`, `plot_cell_data`, …) called in "
        "`run_julia` draw to an offscreen virtual display the user cannot see, so "
        "wrap such results in `plot_julia` instead. Never tell the user a window "
        "opened or that they can rotate/zoom/interact with a plot; they can't."
    )


def _source_note(adapter: SimulatorAdapter, primary_source: str | Path) -> str:
    """State the simulator package's already-resolved source path.

    So the agent greps/reads it directly instead of paying ``using <Sim>`` just to
    call ``pkgdir`` for a path we already know; ``pkgdir`` stays the way to locate
    *other* packages' source.
    """
    primary = adapter.primary_package
    return (
        f"{adapter.display_name}'s package source is already on disk at:\n"
        f"  {primary_source}\n"
        f"Read it directly with `read_file`, `glob`, and `grep` (its `src/`, "
        f"`examples/`, `docs/`). Do not run `using {primary}` or `pkgdir({primary})` "
        f"just to find it — that loads the package (slow the first time) for a path "
        f"you already have. Use `pkgdir(<Package>)` in `run_julia` only to locate a "
        f"different package's source."
    )


def _tool_guide(adapter: SimulatorAdapter) -> str:
    return (
        "You operate in the user's *workspace* (their current working "
        "directory). Two tool families:\n"
        "  - `run_julia` and `plot_julia` run code in a persistent Julia "
        "REPL. State persists across calls. Use the REPL for probing APIs "
        "(`@doc`, `methods`, `names`, `fieldnames`, `pkgdir`), running "
        "simulations, and including workspace scripts.\n"
        "  - The stock file/shell tools (`read_file`, `write_file`, "
        "`edit_file`, `glob`, `grep`, `execute`) work on the real filesystem "
        "from the workspace. Use them to create real implementation files the "
        "user can inspect and edit, and to read installed package source: every "
        "package the environment resolves has its source on disk at the path "
        "`pkgdir(<Package>)` returns (via `run_julia`); "
        "`read_file`, `glob`, and `grep` that path to study its "
        "`examples/`, `docs/`, and `src/`. Installed source is read-only (the "
        "shared depot); `Pkg.develop` a package to edit it. See the "
        "`workspace-and-source` skill.\n"
        "Two ways to run Julia, one shared REPL:\n"
        '    1. Direct: `run_julia("<code>")` for probes, quick computations, '
        "and building/solving inline.\n"
        "    2. From a file: for a real implementation the user can keep, "
        "`write_file` a `.jl` file in the workspace, then run it in the same REPL "
        "with `run_julia('include(\"candidate.jl\")')`. Edit the file with "
        "`edit_file` and re-`include` to re-run. The REPL keeps state, so loaded "
        "packages and earlier results survive across calls.\n"
        "Decision rule: real implementations → write a `.jl` file and `include` it; "
        "quick probes → `run_julia` directly.\n"
        "Plots: build figures only with `plot_julia`, never `run_julia`, which "
        "draws a figure nobody can see. See the `plotting-basics` skill. Prefer "
        "the simulator's documented native plotters (the per-simulator skill names "
        "them); you may also build a `Figure` inline, and you don't need to return "
        "it or avoid `display`. Plotting runs on GLMakie like normal Julia; whether "
        "the user sees a live window, an in-app panel, or just the saved PNG depends "
        "on this session and is described below. "
        "Give related plots a stable `slot`: the same slot refreshes one plot "
        "in place, distinct slots are distinct plots, and "
        "`recapture_plot(slot=...)` / `close_plots(slot=...)` address that plot. "
        "Pass `view=true` only when you need to see the result yourself (verify a "
        "fit, diagnose an anomaly), not for every plot. Don't repeat artifact "
        "paths in prose."
    )


def _ground_rules() -> str:
    """House rules that hold for every simulator and every provider.

    Each rule is stated here and nowhere else; the tool guide above only
    orients (what exists and where it is).
    """

    return (
        "Ground rules:\n"
        "  - Paths are real and shared: the file tools, `execute`, and `run_julia` "
        "all use the working directory above. Name a workspace file by a relative "
        "path (`model.jl`, `experiments/foo.csv`) or its absolute path under the "
        "working directory; both mean the same file in all three, so the file you "
        '`write_file` is the file you `include("model.jl")`. A bare leading slash is '
        "the machine root, not the workspace: `/model.jl` and `/workspace/...` do not "
        "exist, so don't invent them. Everything you touch (your files, installed "
        "package source, memory, added folders) is a real path that opens the same "
        "in every tool.\n"
        "  - Julia runs only in the shared REPL: `run_julia`, or `plot_julia` for "
        "figures. Never spawn `julia` (or `julia --project`, `julia -e`) through "
        "`execute`, since a fresh process shares no state, pays a full precompile, "
        "and needs approval. Results that feed the task's conclusions "
        "(simulation outputs, quantities the user asked for) come from the "
        "session REPL, not recomputed elsewhere; `execute` covers ordinary "
        "shell work.\n"
        "  - When a tool or Julia call fails, read the full error output, diagnose "
        "the root cause (wrong path, missing package, API mismatch, stale REPL "
        "state), and retry with a concrete fix rather than repeating the same "
        "failing call.\n"
        "  - Prefer evidence over memory: read installed package source (its path "
        "is `pkgdir(<Package>)`), probe the REPL (`@doc`, `methods`, `names`), or "
        "consult the skills before guessing an API. If a package is missing, check "
        "what the workspace env already provides, use a stdlib alternative, or "
        "`Pkg.add` it when the task needs it.\n"
        "  - Folders the user adds to the session are available at their real "
        "absolute path in every tool: the file tools, `execute`, and `run_julia`.\n"
        "  - Write maths as text the reader sees, not as LaTeX: no `\\frac`, "
        "`\\times`, `\\mathrm` or `$...$`. Nothing renders it on either front end, "
        "so it reaches the user as source. Use Unicode and plain notation instead "
        "(`dFOPT/dm = 5.44e8 m^3`, `∂p/∂x`, `1.2e-13 m²`), and prefer a "
        "sentence or a table to a formula."
    )
