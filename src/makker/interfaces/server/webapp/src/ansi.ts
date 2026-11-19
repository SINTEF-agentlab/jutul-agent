// Render raw kernel/terminal output (run_julia, execute) for display. The server
// already terminal-renders streamed tool deltas; these passes also tidy raw output
// from non-kernel tools. Everything returns plain data (no HTML), so the React
// layer renders it with no `dangerouslySetInnerHTML`.

/** Standard ANSI SGR foreground colors, tuned to read on the output background. */
const ANSI_FG: Record<number, string> = {
  30: "#3b4252", 31: "#c0392b", 32: "#2f9e44", 33: "#b8860b", 34: "#1f6feb",
  35: "#a626a4", 36: "#0b7285", 37: "#9aa0a8", 90: "#7a828e", 91: "#e05561",
  92: "#37b24d", 93: "#d6a200", 94: "#4098ff", 95: "#c678dd", 96: "#56b6c2", 97: "#cfd3da",
};

export interface AnsiSegment {
  text: string;
  bold?: boolean;
  color?: string;
}

const OSC = /\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)/g;
const CSI = /\x1b\[([0-9;?]*)([@-~])/g;

/** Strip every ANSI escape (colors, cursor moves, OSC) — used to test blankness. */
export function stripAnsi(s: string): string {
  return String(s).replace(OSC, "").replace(/\x1b\[[0-9;?]*[@-~]/g, "");
}

/**
 * Parse ANSI SGR color codes into styled text segments so run_julia output reads
 * like a REPL (errors red, types colored). SGR is stateful: styles accumulate
 * until a reset (code 0). Non-SGR control sequences (cursor moves, line clears
 * from progress bars) are dropped so they don't render as escape soup.
 */
export function parseAnsi(input: string): AnsiSegment[] {
  const text = input.replace(OSC, "");
  const segments: AnsiSegment[] = [];
  let bold = false;
  let color: string | undefined;
  let last = 0;
  let m: RegExpExecArray | null;
  const push = (slice: string) => {
    if (!slice) return;
    segments.push(bold || color ? { text: slice, bold: bold || undefined, color } : { text: slice });
  };
  CSI.lastIndex = 0;
  while ((m = CSI.exec(text))) {
    push(text.slice(last, m.index));
    last = CSI.lastIndex;
    if (m[2] !== "m") continue; // non-SGR CSI (cursor move, clear): drop it
    const codes = m[1]
      .split(";")
      .filter((s) => s !== "")
      .map(Number);
    if (codes.length === 0) {
      // A bare `\x1b[m` is a reset, same as `\x1b[0m`.
      bold = false;
      color = undefined;
      continue;
    }
    // Apply each code in order: a reset (0) clears, then later codes in the SAME
    // sequence still take effect — e.g. `\x1b[0;31m` (reset then red) must end red,
    // not uncolored.
    for (const c of codes) {
      if (c === 0) {
        bold = false;
        color = undefined;
      } else if (c === 1) {
        bold = true;
      } else if (ANSI_FG[c]) {
        color = ANSI_FG[c];
      }
    }
  }
  push(text.slice(last));
  return segments;
}

/**
 * Apply each line's last carriage return: progress bars overwrite their line with
 * `\r`, so only the text after the final `\r` is the line's settled state.
 */
export function applyCarriageReturns(raw: string): string {
  return String(raw)
    .replace(/\r\n/g, "\n") // CRLF is a normal newline, not a cursor overwrite
    .split("\n")
    .map((line) => {
      const cr = line.lastIndexOf("\r");
      return cr >= 0 ? line.slice(cr + 1) : line;
    })
    .join("\n");
}

/**
 * Collapse runs of visually-blank lines to a single blank one. Cursor-redraw
 * progress output leaves lines that look empty but still carry cursor-move/erase
 * escapes, so each line is tested with ANSI and whitespace stripped.
 */
export function collapseBlankRuns(text: string): string {
  const out: string[] = [];
  let blanks = 0;
  for (const line of text.split("\n")) {
    if (stripAnsi(line).trim() === "") {
      if (++blanks === 1) out.push("");
    } else {
      blanks = 0;
      out.push(line);
    }
  }
  return out.join("\n");
}

const OUTPUT_MAX_LINES = 1000;
const OUTPUT_MAX_CHARS = 100000;

/**
 * Tidy kernel output for display. The box scrolls, so this keeps far more than a
 * terminal would and only trims genuinely huge dumps — keeping the newest lines
 * (like a terminal) with a single marker on top, never splicing the middle (which
 * would break tables/stacktraces and read as a gap).
 */
export function clampOutput(input: string): string {
  let text = collapseBlankRuns(input);
  let trimmed = false;
  let lines = text.split("\n");
  if (lines.length > OUTPUT_MAX_LINES) {
    lines = lines.slice(-OUTPUT_MAX_LINES);
    trimmed = true;
  }
  text = lines.join("\n");
  if (text.length > OUTPUT_MAX_CHARS) {
    text = text.slice(-OUTPUT_MAX_CHARS);
    text = text.slice(text.indexOf("\n") + 1); // drop the partial first line
    trimmed = true;
  }
  return trimmed ? "  … earlier output trimmed …\n" + text : text;
}

/** The full display pipeline: settle carriage returns, clamp, then parse ANSI. */
export function terminalSegments(raw: string): AnsiSegment[] {
  return parseAnsi(clampOutput(applyCarriageReturns(raw)));
}
