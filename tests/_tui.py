"""Shared Textual TUI test helpers."""

from __future__ import annotations

import asyncio

from makker.interfaces.tui import TUIApp


async def wait_until_ready(app: TUIApp, timeout: float = 5.0) -> None:
    """Block until the TUI worker has finished and the prompt re-enables."""
    from makker.interfaces.tui.prompt import PromptTextArea

    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        prompt_input = app.query_one("#prompt", PromptTextArea)
        resize_pending = getattr(app, "_resize_timer", None) is not None
        if not prompt_input.disabled and not getattr(app, "_busy", True) and not resize_pending:
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"tui worker did not finish within {timeout}s")


async def submit_prompt(pilot, text: str) -> None:
    await pilot.press(*text)
    await pilot.press("enter")
    await wait_until_ready(pilot.app)
