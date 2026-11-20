"""Typed views over trace payloads shared by transcript renderers.

The trace log keeps ``payload`` as untyped JSON dicts so the schema can evolve
without migrations. Renderers reach for the same handful of fields, though,
so a thin parsing layer keeps that shared knowledge in one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ArtifactPayload:
    """Decoded view of an ``artifact`` trace event."""

    path: str
    caption: str
    mime: str
    slot: str | None
    format: str | None
    size_px: tuple[int, int] | None
    dpi: int | None
    source_code: str | None
    tool_call_id: str | None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> ArtifactPayload:
        size_raw = payload.get("size_px")
        size_px: tuple[int, int] | None = None
        if isinstance(size_raw, (list, tuple)) and len(size_raw) == 2:
            size_px = (int(size_raw[0]), int(size_raw[1]))

        path = str(payload.get("path") or "")
        return cls(
            path=path,
            caption=str(payload.get("caption") or _basename(path) or "artifact"),
            mime=str(payload.get("mime") or ""),
            slot=_str_or_none(payload.get("slot")),
            format=_str_or_none(payload.get("format")),
            size_px=size_px,
            dpi=_int_or_none(payload.get("dpi")),
            source_code=_text_or_none(payload.get("source_code")),
            tool_call_id=_str_or_none(payload.get("tool_call_id")),
        )

    @property
    def is_image(self) -> bool:
        return self.mime.startswith("image/")


def _basename(path: str) -> str:
    """The final path segment, separator-agnostic (paths may be POSIX or Windows)."""
    return path.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _text_or_none(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value
