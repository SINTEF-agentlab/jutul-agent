import { describe, expect, it } from "vitest";

import { fmtNum, formatElapsed, formatTokens, timeAgo } from "./format";

describe("formatTokens", () => {
  it("compacts thousands", () => {
    expect(formatTokens(500)).toBe("500");
    expect(formatTokens(1500)).toBe("1.5k");
    expect(formatTokens(10000)).toBe("10k");
    expect(formatTokens(24000)).toBe("24k");
  });
});

describe("fmtNum", () => {
  it("rounds floats to four significant digits, passes through non-numbers", () => {
    expect(fmtNum(3.14159)).toBe("3.142");
    expect(fmtNum(42)).toBe("42");
    expect(fmtNum("label")).toBe("label");
  });
});

describe("formatElapsed", () => {
  it("counts seconds, then minutes and seconds", () => {
    expect(formatElapsed(0)).toBe("0s");
    expect(formatElapsed(12_400)).toBe("12s");
    expect(formatElapsed(59_999)).toBe("59s");
    expect(formatElapsed(65_000)).toBe("1:05");
    expect(formatElapsed(750_000)).toBe("12:30");
  });

  it("never counts backwards from a clock that jumped", () => {
    expect(formatElapsed(-5000)).toBe("0s");
  });
});

describe("timeAgo", () => {
  it("describes recent times relatively", () => {
    expect(timeAgo(new Date().toISOString())).toBe("just now");
    expect(timeAgo(new Date(Date.now() - 5 * 60_000).toISOString())).toBe("5m ago");
    expect(timeAgo(new Date(Date.now() - 3 * 3_600_000).toISOString())).toBe("3h ago");
    expect(timeAgo(new Date(Date.now() - 2 * 86_400_000).toISOString())).toBe("2d ago");
  });
});
