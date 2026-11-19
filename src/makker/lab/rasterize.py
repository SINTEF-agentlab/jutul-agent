"""Rasterise an SVG to PNG with a headless browser, so an agent can view it.

Textual exports screenshots as SVG; an agent's vision needs a raster image. Chrome,
Chromium, or Edge in headless mode renders the SVG and screenshots it. No new Python
dependency. Returns ``False`` (rather than raising) when no browser is found, so a
capture run still produces the SVG and text artifacts.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

_BROWSER_NAMES = ("google-chrome", "chromium", "chromium-browser", "chrome", "msedge")
_WINDOWS_CANDIDATES = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
)
_MAC_CANDIDATES = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
)


def find_browser() -> str | None:
    """A path to a headless-capable browser, or ``None``."""
    for name in _BROWSER_NAMES:
        found = shutil.which(name)
        if found:
            return found
    for candidate in (*_WINDOWS_CANDIDATES, *_MAC_CANDIDATES):
        if Path(candidate).exists():
            return candidate
    return None


# Textual exports with a viewBox rather than width/height attributes.
_SIZE_RE = re.compile(r'<svg[^>]*\bviewBox="0 0 ([\d.]+) ([\d.]+)"')


def _svg_size(svg: str) -> tuple[int, int]:
    match = _SIZE_RE.search(svg)
    if not match:
        return 1280, 800
    return round(float(match.group(1))) + 4, round(float(match.group(2))) + 4


def svg_to_png(svg_path: Path, png_path: Path, *, browser: str | None = None) -> bool:
    """Render ``svg_path`` to ``png_path``. Returns whether a PNG was produced."""
    browser = browser or find_browser()
    if browser is None:
        return False
    width, height = _svg_size(svg_path.read_text(encoding="utf-8"))
    url = svg_path.resolve().as_uri()
    cmd = [
        browser,
        "--headless=new",
        "--disable-gpu",
        "--hide-scrollbars",
        "--force-device-scale-factor=1",
        f"--window-size={width},{height}",
        f"--screenshot={png_path.resolve()}",
        url,
    ]
    try:
        subprocess.run(cmd, check=False, capture_output=True, timeout=60, env={**os.environ})
    except (OSError, subprocess.SubprocessError):
        return False
    return png_path.exists()
