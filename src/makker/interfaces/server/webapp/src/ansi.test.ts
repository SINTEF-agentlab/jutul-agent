import { describe, expect, it } from "vitest";

import {
  applyCarriageReturns,
  clampOutput,
  collapseBlankRuns,
  parseAnsi,
  stripAnsi,
} from "./ansi";

describe("applyCarriageReturns", () => {
  it("keeps only the text after the last carriage return on each line", () => {
    expect(applyCarriageReturns("abc\rdef")).toBe("def");
    expect(applyCarriageReturns("one\ntwo\rTWO\nthree")).toBe("one\nTWO\nthree");
  });
  it("treats CRLF as a plain newline, not a cursor overwrite", () => {
    // Windows / non-kernel tool output uses \r\n; the \r must not blank the line.
    expect(applyCarriageReturns("hello\r\nworld")).toBe("hello\nworld");
    expect(applyCarriageReturns("done\r\n")).toBe("done\n");
  });
});

describe("collapseBlankRuns", () => {
  it("collapses runs of blank lines to a single blank line", () => {
    expect(collapseBlankRuns("a\n\n\n\nb")).toBe("a\n\nb");
  });
  it("treats lines that are blank after stripping ANSI as blank", () => {
    expect(collapseBlankRuns("a\n\x1b[2K\n\x1b[2K\nb")).toBe("a\n\nb");
  });
});

describe("clampOutput", () => {
  it("leaves small output untouched (aside from blank-run collapse)", () => {
    expect(clampOutput("hello\nworld")).toBe("hello\nworld");
  });
  it("trims to the newest lines with a marker when very long", () => {
    const text = Array.from({ length: 1500 }, (_, i) => `line ${i}`).join("\n");
    const out = clampOutput(text);
    expect(out.startsWith("  … earlier output trimmed …\n")).toBe(true);
    expect(out).toContain("line 1499");
    expect(out).not.toContain("line 100\n");
  });
});

describe("parseAnsi", () => {
  it("turns SGR color codes into styled segments", () => {
    expect(parseAnsi("\x1b[31mred\x1b[0m plain")).toEqual([
      { text: "red", color: "#c0392b" },
      { text: " plain" },
    ]);
  });
  it("marks bold and accumulates style until reset", () => {
    expect(parseAnsi("\x1b[1mbold")).toEqual([{ text: "bold", bold: true }]);
  });
  it("applies a color that follows a reset in the same sequence", () => {
    // `\x1b[0;31m` (reset then red) is a common reset-then-set idiom; the color
    // must survive the reset and the text end up red, not uncolored.
    expect(parseAnsi("\x1b[0;31mred")).toEqual([{ text: "red", color: "#c0392b" }]);
  });
  it("drops non-SGR control sequences (cursor moves, clears) and OSC", () => {
    expect(parseAnsi("\x1b[2K\x1b[Atext")).toEqual([{ text: "text" }]);
    expect(parseAnsi("\x1b]0;title\x07body")).toEqual([{ text: "body" }]);
  });
});

describe("stripAnsi", () => {
  it("removes every escape sequence", () => {
    expect(stripAnsi("\x1b[31mred\x1b[0m")).toBe("red");
  });
});
