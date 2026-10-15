"""The front ends ("surfaces") a session can be driven from, and how they show results.

A session records the surface it was started for (``Session.surface``) and the
plotting and report tools consult it. ``tui`` and ``cli`` show figures as desktop
windows and saved PNGs. A browser surface serves each figure live from the
session's Bonito server and exports HTML instead: ``web`` is the bundled browser
UI, and ``mcp`` is the MCP server, whose loopback viewer serves the same pages to
whichever browser the user opens them in.
"""

from __future__ import annotations

BROWSER_SURFACES = frozenset({"web", "mcp"})


def renders_in_browser(surface: str | None) -> bool:
    """Whether figures and reports for ``surface`` are viewed in a browser."""
    return surface in BROWSER_SURFACES
