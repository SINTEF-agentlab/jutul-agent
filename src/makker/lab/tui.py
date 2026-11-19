"""Render TUI scenarios headlessly and capture what they look like.

``capture`` drives one scenario through the real Textual app (with a scripted agent
and a fake Julia process) and writes three artifacts: an SVG screenshot, a plain-text
snapshot pulled from the SVG (for grep and diffs), and the session transcript. With
``--png`` it also rasterises the SVG so an agent can view it directly.

Run it:

    python -m makker.lab.tui list
    python -m makker.lab.tui run tool_call --png
    python -m makker.lab.tui all --png -o /tmp/tui
"""

from __future__ import annotations

import argparse
import asyncio
import html
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

from makker.lab.fakes import FakeJulia, make_fake_adapter
from makker.lab.scenarios import Scenario, all_scenarios, get


async def _settle(app: Any, timeout: float = 6.0) -> None:
    """Wait until the worker is idle or an approval prompt is up; never raises."""
    from makker.interfaces.tui.approval_menu import ApprovalMenu
    from makker.interfaces.tui.prompt import PromptTextArea

    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(0.05)
        prompt = app.query_one("#prompt", PromptTextArea)
        approving = bool(app.query(ApprovalMenu))
        resize_pending = getattr(app, "_resize_timer", None) is not None
        if approving:
            return
        if not prompt.disabled and not getattr(app, "_busy", True) and not resize_pending:
            return


async def _submit(app: Any, pilot: Any, text: str) -> None:
    from makker.interfaces.tui.prompt import PromptTextArea

    prompt = app.query_one("#prompt", PromptTextArea)
    prompt.text = text
    prompt.focus()
    await pilot.pause()
    await pilot.press("enter")
    await _settle(app)


async def _drive(app: Any, pilot: Any, steps: tuple) -> None:
    for step in steps:
        if isinstance(step, str):
            await _submit(app, pilot, step)
        else:
            await step(pilot)
            await _settle(app)


_TEXT_RE = re.compile(r'<text[^>]*\sy="([\d.]+)"[^>]*>(.*?)</text>', re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")


def svg_to_text(svg: str) -> str:
    """A plain-text view of the screen, recovered from the SVG text runs by row."""
    rows: dict[int, list[str]] = {}
    for y, body in _TEXT_RE.findall(svg):
        text = html.unescape(_TAG_RE.sub("", body))
        rows.setdefault(round(float(y)), []).append(text)
    lines = ["".join(rows[y]).rstrip() for y in sorted(rows)]
    return "\n".join(lines).rstrip() + "\n"


# The session id (date plus a random suffix) is the only per-run value in a render.
_SESSION_RE = re.compile(r"\d{4}-\d{2}-\d{2}-\d{4}-[0-9a-f]{4}")


def normalize_screen(text: str) -> str:
    """Mask the per-run session id so a screen can be snapshot-compared."""
    return _SESSION_RE.sub("<session>", text)


async def _render(scenario: Scenario) -> tuple[str, str]:
    """Drive a scenario and return ``(svg, transcript_markdown)``."""
    from makker.interfaces.tui import TUIApp
    from makker.paths import set_workspace_root
    from makker.session import Session
    from makker.transcript import render_markdown

    tmp = Path(tempfile.mkdtemp(prefix="jutul-lab-"))
    # Keep session output (artifacts, gitignore) inside the scratch dir.
    set_workspace_root(tmp)
    session = Session.create(julia=FakeJulia(), state_root=tmp, simulator=make_fake_adapter(tmp))
    agent = scenario.build_agent() if scenario.build_agent else None
    app = TUIApp(agent=agent, session=session, model_label=scenario.model_label)
    async with app.run_test(size=scenario.size) as pilot:
        await _settle(app)
        await _drive(app, pilot, scenario.steps)
        # Pin a deterministic scroll/layout state before the screenshot. The app
        # anchors to the bottom and jumps to the latest on new content, but _settle
        # returns as soon as the approval menu appears, before that scroll lands, so
        # the captured viewport would otherwise depend on machine speed: CI caught
        # the pre-scroll top, a fast laptop the settled bottom. Drain pending layout,
        # then force the latest into view (twice, in case the drain mounted more
        # content) to match what a user sees once the turn has settled.
        if app.is_mounted:
            log = app.query_one("#log")
            for _ in range(5):
                await pilot.pause()
            log.scroll_end(animate=False)
            for _ in range(2):
                await pilot.pause()
            log.scroll_end(animate=False)
            await pilot.pause()
        svg = app.export_screenshot()
    transcript = render_markdown(session.trace.iter_events())
    return svg, transcript


def capture(scenario: Scenario, out_dir: Path, *, png: bool = False) -> dict[str, Path]:
    """Render a scenario and write its artifacts under ``out_dir/<name>/``."""
    svg, transcript = asyncio.run(_render(scenario))
    target = out_dir / scenario.name
    target.mkdir(parents=True, exist_ok=True)
    paths = {
        "svg": target / "screen.svg",
        "txt": target / "screen.txt",
        "transcript": target / "transcript.md",
    }
    paths["svg"].write_text(svg, encoding="utf-8")
    paths["txt"].write_text(svg_to_text(svg), encoding="utf-8")
    paths["transcript"].write_text(transcript, encoding="utf-8")
    if png:
        from makker.lab.rasterize import svg_to_png

        out_png = target / "screen.png"
        if svg_to_png(paths["svg"], out_png):
            paths["png"] = out_png
    return paths


def _default_out() -> Path:
    from makker.paths import state_home

    return state_home() / "lab" / "tui"


# ---- CLI -------------------------------------------------------------------


def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m makker.lab.tui")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("list", help="List scenarios.")
    p_run = sub.add_parser("run", help="Render one scenario.")
    p_run.add_argument("name")
    p_all = sub.add_parser("all", help="Render every scenario.")
    for p in (p_run, p_all):
        p.add_argument("-o", "--out", default=None, help="Output directory.")
        p.add_argument(
            "--png", action="store_true", help="Also rasterise to PNG (needs a browser)."
        )
        p.add_argument("--size", default=None, help="Override terminal size, e.g. 120x36.")

    args = parser.parse_args(argv)
    if args.cmd == "list" or args.cmd is None:
        for s in all_scenarios():
            tags = f"  [{', '.join(s.tags)}]" if s.tags else ""
            print(f"{s.name:16} {s.description}{tags}")
        return 0

    out = Path(args.out) if args.out else _default_out()
    scenarios = [get(args.name)] if args.cmd == "run" else all_scenarios()
    if args.size:
        w, h = (int(v) for v in args.size.lower().split("x"))
        scenarios = [Scenario(**{**s.__dict__, "size": (w, h)}) for s in scenarios]

    for s in scenarios:
        paths = capture(s, out, png=args.png)
        shown = paths.get("png") or paths["svg"]
        print(f"{s.name:16} -> {shown}")
    print(f"\nArtifacts under {out}")
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
