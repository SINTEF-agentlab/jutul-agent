"""Render captured Julia output to its final on-screen state.

Progress libraries (e.g. ProgressMeter.jl) emit carriage returns and cursor-
movement codes to overwrite a progress block in place. We own the output pipe,
but the bytes still carry those control sequences; a naive strip leaves every
intermediate update stacked. ``render_terminal_output`` replays the cursor moves
through a tiny screen buffer so the result matches what a real terminal would
show — a single final bar instead of hundreds of stacked ones.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import ClassVar

# CSI (Control Sequence Introducer): ESC '[' params final
_CSI_RE = re.compile(r"\x1b\[([0-9;?]*)([A-Za-z])")
# OSC (Operating System Command): ESC ']' ... BEL or ESC '\'
_OSC_RE = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")
# Standalone single-character escapes (ESC followed by one byte that isn't `[` or `]`)
_OTHER_ESC_RE = re.compile(r"\x1b[^\[\]]")
# Cursor-control bytes that need the screen model. Plain output has none of these,
# so it skips the per-character emulation below entirely.
_NEEDS_SCREEN_RE = re.compile("[\r\x1b\b\t]")


def render_terminal_output(text: str) -> str:
    """Render ``text`` to its final on-screen state, like a terminal would.

    SGR (colour) codes are dropped; CSI sequences we don't model are skipped.
    Tabs expand to 8-column stops, ``\\b`` moves the cursor back, and the result
    is joined with ``\\n`` with trailing whitespace stripped.
    """

    if _NEEDS_SCREEN_RE.search(text) is None:
        # No cursor control (the common case): just trim trailing whitespace per
        # line and drop trailing blank lines, matching what the screen model below
        # would produce for plain text.
        return "\n".join(line.rstrip() for line in text.split("\n")).rstrip("\n")

    text = _OSC_RE.sub("", text)
    text = _OTHER_ESC_RE.sub("", text)

    screen = _Screen()
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "\x1b":
            match = _CSI_RE.match(text, i)
            if match is None:
                i += 1
                continue
            screen.apply_csi(match.group(1), match.group(2))
            i = match.end()
            continue
        if ch == "\r":
            screen.carriage_return()
        elif ch == "\n":
            screen.newline()
        elif ch == "\b":
            screen.backspace()
        elif ch == "\t":
            screen.tab()
        else:
            screen.write(ch)
        i += 1

    return screen.render()


class _Screen:
    """Tiny in-memory screen buffer for ``render_terminal_output``.

    Handles the CSI sequences Julia output actually uses: cursor up/down/left/
    right, absolute positioning, line erase (``K``), display erase (``J``), and
    the column-1 reset on row change. SGR (``m``) is intentionally ignored.
    """

    def __init__(self) -> None:
        self._lines: list[list[str]] = [[]]
        self._row = 0
        self._col = 0

    # ---- character ops -----------------------------------------------------

    def write(self, ch: str) -> None:
        self._ensure_col(self._col)
        self._lines[self._row][self._col] = ch
        self._col += 1

    def carriage_return(self) -> None:
        self._col = 0

    def newline(self) -> None:
        self._row += 1
        self._col = 0
        self._ensure_row(self._row)

    def backspace(self) -> None:
        self._col = max(0, self._col - 1)

    def tab(self) -> None:
        stop = self._col + (8 - (self._col % 8))
        self._ensure_col(stop - 1)
        self._col = stop

    # ---- CSI dispatch ------------------------------------------------------

    def apply_csi(self, params: str, final: str) -> None:
        if final == "m":  # SGR (colour) — ignored
            return
        n = self._parse_n(params)
        handler = self._CSI_HANDLERS.get(final)
        if handler is not None:
            handler(self, params, n)

    def _csi_up(self, _params: str, n: int) -> None:
        self._row = max(0, self._row - n)

    def _csi_down(self, _params: str, n: int) -> None:
        self._row += n
        self._ensure_row(self._row)

    def _csi_right(self, _params: str, n: int) -> None:
        self._col += n

    def _csi_left(self, _params: str, n: int) -> None:
        self._col = max(0, self._col - n)

    def _csi_next_line(self, _params: str, n: int) -> None:
        self._row += n
        self._col = 0
        self._ensure_row(self._row)

    def _csi_prev_line(self, _params: str, n: int) -> None:
        self._row = max(0, self._row - n)
        self._col = 0

    def _csi_column(self, _params: str, n: int) -> None:
        self._col = max(0, n - 1)

    def _csi_position(self, params: str, _n: int) -> None:
        parts = params.split(";") if params else []
        r = int(parts[0]) if parts and parts[0].isdigit() else 1
        c = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
        self._row = max(0, r - 1)
        self._col = max(0, c - 1)
        self._ensure_row(self._row)

    def _csi_erase_display(self, params: str, _n: int) -> None:
        mode = int(params) if params and params.isdigit() else 0
        if mode == 0:  # cursor to end of screen
            del self._lines[self._row][self._col :]
            del self._lines[self._row + 1 :]
        elif mode == 1:  # start of screen to cursor
            for r in range(self._row):
                self._lines[r] = []
            self._lines[self._row][: self._col] = [" "] * self._col
        else:  # mode 2 or 3 — whole screen
            self._lines = [[]]
            self._row = 0
            self._col = 0

    def _csi_erase_line(self, params: str, _n: int) -> None:
        mode = int(params) if params and params.isdigit() else 0
        line = self._lines[self._row]
        if mode == 0:  # cursor to end of line
            del line[self._col :]
        elif mode == 1:  # start of line to cursor
            line[: self._col] = [" "] * self._col
        else:  # whole line
            line.clear()

    _CSI_HANDLERS: ClassVar[dict[str, Callable[[_Screen, str, int], None]]] = {
        "A": _csi_up,
        "B": _csi_down,
        "C": _csi_right,
        "D": _csi_left,
        "E": _csi_next_line,
        "F": _csi_prev_line,
        "G": _csi_column,
        "H": _csi_position,
        "f": _csi_position,
        "J": _csi_erase_display,
        "K": _csi_erase_line,
    }

    # ---- buffer management -------------------------------------------------

    def _ensure_row(self, target: int) -> None:
        while target >= len(self._lines):
            self._lines.append([])

    def _ensure_col(self, target: int) -> None:
        line = self._lines[self._row]
        while target >= len(line):
            line.append(" ")

    @staticmethod
    def _parse_n(params: str) -> int:
        if not params or not params.isdigit():
            return 1
        try:
            return int(params)
        except ValueError:
            return 1

    def render(self) -> str:
        return "\n".join("".join(line).rstrip() for line in self._lines).rstrip("\n")
