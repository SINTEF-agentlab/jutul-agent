import { describe, expect, it } from "vitest";

import { parseServerMessage, SIDE_OUTPUT_TYPES } from "./protocol";

describe("parseServerMessage", () => {
  it("parses a typed message", () => {
    expect(parseServerMessage('{"type":"text","text":"hi"}')).toEqual({ type: "text", text: "hi" });
  });
  it("rejects non-JSON and untyped payloads", () => {
    expect(parseServerMessage("not json")).toBeNull();
    expect(parseServerMessage('{"no":"type"}')).toBeNull();
  });
});

describe("SIDE_OUTPUT_TYPES", () => {
  it("covers mid-turn outputs but not agent content or command results", () => {
    for (const t of ["usage", "viz", "artifact", "ui"] as const) {
      expect(SIDE_OUTPUT_TYPES.has(t)).toBe(true);
    }
    expect(SIDE_OUTPUT_TYPES.has("text")).toBe(false);
    expect(SIDE_OUTPUT_TYPES.has("tool")).toBe(false);
    // A notice ends a command's working state (e.g. /compact), so it is not a
    // passive mid-turn side output.
    expect(SIDE_OUTPUT_TYPES.has("notice")).toBe(false);
  });
});
