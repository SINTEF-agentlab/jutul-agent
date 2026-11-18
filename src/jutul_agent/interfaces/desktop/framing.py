"""The frame format the desktop shell and its backend exchange over pipes.

One frame is an 8-byte prefix (two big-endian ``uint32``: header length, body
length), a UTF-8 JSON header, and a raw body. The header says what the frame is
(``{"t": "http", "id": 3, ...}``); the body carries bytes that would be wasteful
or lossy to put in JSON (a request or response body, a binary WebSocket
message). ``electron/backend.js`` is the other end and must agree on this.
"""

from __future__ import annotations

import json
import struct
from typing import Any, BinaryIO

_PREFIX = struct.Struct(">II")
# Sanity bounds: a torn or foreign stream reads as an absurd length long before
# it reads as a valid frame, and failing then beats allocating gigabytes.
MAX_HEADER = 4 * 1024 * 1024
MAX_BODY = 1024 * 1024 * 1024


class FrameError(RuntimeError):
    """The stream does not hold a well-formed frame."""


def encode(header: dict[str, Any], body: bytes = b"") -> bytes:
    """One frame as bytes, ready for a single write."""
    head = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return _PREFIX.pack(len(head), len(body)) + head + body


def _read_exact(stream: BinaryIO, n: int) -> bytes | None:
    """Exactly ``n`` bytes, or ``None`` if the stream ended first."""
    chunks: list[bytes] = []
    while n > 0:
        chunk = stream.read(n)
        if not chunk:
            return None
        chunks.append(chunk)
        n -= len(chunk)
    return b"".join(chunks)


def read(stream: BinaryIO) -> tuple[dict[str, Any], bytes] | None:
    """The next frame from a blocking stream, or ``None`` at a clean end."""
    prefix = _read_exact(stream, _PREFIX.size)
    if prefix is None:
        return None
    head_len, body_len = _PREFIX.unpack(prefix)
    if head_len > MAX_HEADER or body_len > MAX_BODY:
        raise FrameError(f"implausible frame ({head_len} + {body_len} bytes)")
    head = _read_exact(stream, head_len)
    body = _read_exact(stream, body_len) if body_len else b""
    if head is None or body is None:
        raise FrameError("the stream ended inside a frame")
    try:
        header = json.loads(head)
    except ValueError as exc:
        raise FrameError(f"bad frame header: {exc}") from exc
    if not isinstance(header, dict):
        raise FrameError("a frame header must be a JSON object")
    return header, body
