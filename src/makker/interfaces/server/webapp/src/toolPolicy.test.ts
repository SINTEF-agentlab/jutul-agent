import { describe, expect, it } from "vitest";

import { argPreview, asObject, listingNote, summarizeArgs, unitNote } from "./toolPolicy";

describe("argPreview", () => {
  it("shows the active todo, else the count", () => {
    expect(
      argPreview({ todos: [{ content: "do it", status: "in_progress" }] }, "write_todos"),
    ).toBe("do it");
    expect(argPreview({ todos: [{}, {}] }, "write_todos")).toBe("2 items");
    expect(argPreview({ todos: [{}] }, "write_todos")).toBe("1 item");
  });

  it("prefers known keys, falling back to the first scalar value", () => {
    expect(argPreview({ code: "line1\nline2" })).toBe("line1");
    expect(argPreview({ command: "ls -la" })).toBe("ls -la");
    expect(argPreview({ anything: "value" })).toBe("value");
  });

  it("skips object-valued first args and empty args", () => {
    expect(argPreview({ nested: { a: 1 } })).toBe("");
    expect(argPreview(null)).toBe("");
  });
});

describe("notes", () => {
  it("counts non-blank lines", () => {
    expect(unitNote("a\nb\nc", "line")).toBe("3 lines");
    expect(unitNote("only", "match", "matches")).toBe("1 match");
  });
  it("counts quoted entries in an ls repr", () => {
    expect(listingNote("['a/', 'b/', 'c.jl']")).toBe("3 entries");
    expect(listingNote("['solo/']")).toBe("1 entry");
    expect(listingNote("[]")).toBe("0 entries");
    // A filename with an apostrophe is repr'd with double quotes; the count must
    // not be thrown off by the apostrophe inside it.
    expect(listingNote("[\"it's.jl\", 'b/']")).toBe("2 entries");
  });
});

describe("asObject", () => {
  it("accepts objects and JSON strings, rejects arrays and junk", () => {
    expect(asObject({ a: 1 })).toEqual({ a: 1 });
    expect(asObject('{"b":2}')).toEqual({ b: 2 });
    expect(asObject("[1,2]")).toBeNull();
    expect(asObject("nope")).toBeNull();
    expect(asObject(42)).toBeNull();
  });
});

describe("summarizeArgs", () => {
  it("renders key: value lines", () => {
    expect(summarizeArgs({ a: "x", b: 2 })).toBe("a: x\nb: 2");
  });
});
