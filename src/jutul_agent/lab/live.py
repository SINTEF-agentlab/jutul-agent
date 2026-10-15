"""Drive the real TUI with a real model and a real Julia kernel, headlessly.

The scripted scenarios in :mod:`jutul_agent.lab.scenarios` are fast and
deterministic, but they cannot surface what a live model and a live solve actually
render: real streaming, real tool output, real errors. This runs one prompt through
the full stack (Julia kernel, ``build_agent`` with a metered model, the Textual app)
and captures the screen, so an agent can see how the UI behaves on a real turn.

It costs API and needs a workspace with an instantiated Julia env. Everything is
wrapped in an overall timeout so a hung kernel or a slow turn cannot stall a run.

    python -m jutul_agent.lab.live "compute mean([1.0, 2.0, 3.0]) in Julia" \
        --workspace testbed/jutuldarcy --model openai:gpt-5.4 --png
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import tempfile
from pathlib import Path
from typing import Any

from jutul_agent.lab.tui import _settle, svg_to_text


async def _drive_frames(
    app: Any,
    pilot: Any,
    text: str,
    *,
    settle_timeout: float,
    interval: float,
    max_frames: int,
) -> list[str]:
    """Submit a prompt and snapshot the screen over time until the turn finishes.

    Returns one SVG per frame, ending with a final frame once the turn is idle, so
    an agent can watch a real turn progress (thinking, tool, output, answer) rather
    than only see the end state.
    """
    from jutul_agent.interfaces.tui.approval_menu import ApprovalMenu
    from jutul_agent.interfaces.tui.prompt import PromptTextArea

    prompt = app.query_one("#prompt", PromptTextArea)
    prompt.text = text
    prompt.focus()
    await pilot.pause()
    await pilot.press("enter")

    frames: list[str] = []
    loop = asyncio.get_event_loop()
    deadline = loop.time() + settle_timeout
    while loop.time() < deadline and len(frames) < max_frames:
        await asyncio.sleep(interval)
        await pilot.pause()
        frames.append(app.export_screenshot())
        if not getattr(app, "_busy", True) and not app.query(ApprovalMenu):
            break
    # A final settled frame, in case the loop stopped mid-update.
    await _settle(app, timeout=5)
    frames.append(app.export_screenshot())
    return frames


async def _live_render(
    prompt: str,
    *,
    workspace: Path,
    simulator: str,
    model: str,
    size: tuple[int, int],
    settle_timeout: float,
    interval: float,
    max_frames: int,
) -> tuple[list[str], str]:
    from jutul_agent.agent.builder import build_agent, resolve_package_sources
    from jutul_agent.credentials import is_secret_env_var
    from jutul_agent.interfaces.tui import TUIApp
    from jutul_agent.juliakernel import JuliaKernel, KernelConfig
    from jutul_agent.paths import set_workspace_root
    from jutul_agent.session import Session
    from jutul_agent.simulators import registry
    from jutul_agent.transcript import render_markdown
    from jutul_agent.workspace import resolve_julia_project

    adapter = registry.get(simulator)
    set_workspace_root(workspace)
    julia_project = resolve_julia_project(workspace)
    scratch = Path(tempfile.mkdtemp(prefix="jutul-live-"))

    config = KernelConfig(julia_project=julia_project, cwd=workspace, env_filter=is_secret_env_var)
    async with JuliaKernel(config) as julia:
        session = Session.create(
            julia=julia, simulator=adapter, state_root=scratch, ephemeral_memory=True
        )
        package_sources = await asyncio.to_thread(resolve_package_sources, julia_project)
        agent, backend = build_agent(
            session, model=model, approval_mode="auto", package_sources=package_sources
        )
        app = TUIApp(
            agent=agent,
            session=session,
            backend=backend,
            model_label=model,
            approval_mode="auto",
        )
        async with app.run_test(size=size) as pilot:
            await _settle(app, timeout=30)
            frames = await _drive_frames(
                app,
                pilot,
                prompt,
                settle_timeout=settle_timeout,
                interval=interval,
                max_frames=max_frames,
            )
        transcript = render_markdown(session.trace.iter_events())
    return frames, transcript


def capture_live(
    prompt: str,
    out_dir: Path,
    *,
    workspace: Path,
    simulator: str = "jutuldarcy",
    model: str = "openai:gpt-5.4",
    size: tuple[int, int] = (110, 32),
    settle_timeout: float = 240.0,
    overall_timeout: float = 360.0,
    interval: float = 8.0,
    max_frames: int = 1,
    png: bool = False,
) -> dict[str, Path]:
    """Run one live prompt and write its artifacts; safe to call unattended.

    With ``max_frames > 1`` it writes a filmstrip (``frame_00`` ...) sampled every
    ``interval`` seconds; the last frame is the settled end state.
    """

    async def _run() -> tuple[list[str], str]:
        return await asyncio.wait_for(
            _live_render(
                prompt,
                workspace=workspace,
                simulator=simulator,
                model=model,
                size=size,
                settle_timeout=settle_timeout,
                interval=interval,
                max_frames=max(1, max_frames),
            ),
            timeout=overall_timeout,
        )

    frames, transcript = asyncio.run(_run())
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "live.transcript.md").write_text(transcript, encoding="utf-8")

    from jutul_agent.lab.rasterize import svg_to_png

    paths: dict[str, Path] = {"transcript": out_dir / "live.transcript.md"}
    single = len(frames) == 1
    for i, svg in enumerate(frames):
        stem = "live" if single else f"frame_{i:02d}"
        svg_path = out_dir / f"{stem}.svg"
        svg_path.write_text(svg, encoding="utf-8")
        (out_dir / f"{stem}.txt").write_text(svg_to_text(svg), encoding="utf-8")
        if png and svg_to_png(svg_path, out_dir / f"{stem}.png"):
            paths.setdefault("png", out_dir / f"{stem}.png")
    paths["svg"] = out_dir / ("live.svg" if single else "frame_00.svg")
    paths["last"] = out_dir / ("live.png" if single else f"frame_{len(frames) - 1:02d}.png")
    return paths


def _cli(argv: list[str] | None = None) -> int:
    from jutul_agent.credentials import load_user_credentials

    parser = argparse.ArgumentParser(prog="python -m jutul_agent.lab.live")
    parser.add_argument("prompt")
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--simulator", default="jutuldarcy")
    parser.add_argument("--model", default="openai:gpt-5.4")
    parser.add_argument("-o", "--out", default=None)
    parser.add_argument("--png", action="store_true")
    parser.add_argument("--size", default="110x32")
    parser.add_argument("--settle", type=float, default=240.0, help="Seconds to wait for the turn.")
    parser.add_argument("--timeout", type=float, default=360.0, help="Overall hard timeout.")
    parser.add_argument("--frames", type=int, default=1, help="Snapshot up to N frames over time.")
    parser.add_argument("--interval", type=float, default=8.0, help="Seconds between frames.")
    args = parser.parse_args(argv)

    load_user_credentials()
    from dotenv import load_dotenv

    load_dotenv()

    w, h = (int(v) for v in args.size.lower().split("x"))
    out = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="jutul-live-out-"))
    try:
        paths = capture_live(
            args.prompt,
            out,
            workspace=args.workspace.resolve(),
            simulator=args.simulator,
            model=args.model,
            size=(w, h),
            settle_timeout=args.settle,
            overall_timeout=args.timeout,
            interval=args.interval,
            max_frames=args.frames,
            png=args.png,
        )
    except TimeoutError:
        print(f"timed out after {args.timeout}s", file=sys.stderr)
        return 1
    print(f"Live capture -> {paths.get('png') or paths['svg']}")
    if "last" in paths and paths["last"].exists():
        print(f"Final frame -> {paths['last']}")
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
