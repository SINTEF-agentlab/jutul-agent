"""Zip a rendered HTML transcript with its sibling artifacts for sharing."""

from __future__ import annotations

import zipfile
from pathlib import Path


def bundle_transcript(session_dir: Path, html_path: Path, bundle_path: Path) -> None:
    """Write ``bundle_path`` containing ``transcript.html`` and ``artifacts/``."""
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(html_path, arcname="transcript.html")
        artifacts = session_dir / "artifacts"
        if artifacts.is_dir():
            for path in artifacts.rglob("*"):
                if path.is_file():
                    archive.write(path, arcname=str(path.relative_to(session_dir)))
