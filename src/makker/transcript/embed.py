"""Resolve and inline artifact files, contained to the run's directories.

Transcripts and reports inline file *bytes* (base64 data URIs) so they stay
single, shareable files. That makes containment load-bearing: a recorded or
agent-authored path must resolve to a file inside one of the run's artifact
dirs, and an absolute or ``..`` path that escapes every dir is rejected, so a
stray or crafted reference can never pull arbitrary files into a shared
document. One resolver, used by every renderer, mirroring the server's
``_resolve_artifact`` guard.
"""

from __future__ import annotations

import base64
import mimetypes
from collections.abc import Sequence
from pathlib import Path

# Largest artifact to base64-inline; above this, keep the path reference.
MAX_INLINE_BYTES = 10 * 1024 * 1024


def resolve_artifact_file(path: str, artifact_dirs: Sequence[Path]) -> Path | None:
    """The on-disk file for a recorded artifact path, contained to the run dirs."""
    raw = Path(path)
    for root in artifact_dirs:
        base = root.resolve()
        candidate = (base / raw).resolve()
        if candidate.is_file() and candidate.is_relative_to(base):
            return candidate
        if path.startswith("artifacts/"):  # the document may sit inside artifacts/
            tail = (base / raw.name).resolve()
            if tail.is_file() and tail.is_relative_to(base):
                return tail
    return None


def image_data_uri(path: str, artifact_dirs: Sequence[Path]) -> str | None:
    """A ``data:`` URI for a contained artifact, or ``None`` when it must not inline.

    ``None`` means the path escaped the run dirs, the file is missing or
    unreadable, or it is too large to inline; the caller decides the fallback
    (keep the path, or show a missing-plot note).
    """
    resolved = resolve_artifact_file(path, artifact_dirs)
    if resolved is None:
        return None
    try:
        if resolved.stat().st_size > MAX_INLINE_BYTES:
            return None
        data = base64.standard_b64encode(resolved.read_bytes()).decode("ascii")
    except OSError:
        return None
    mime = mimetypes.guess_type(resolved.name)[0] or "application/octet-stream"
    return f"data:{mime};base64,{data}"
