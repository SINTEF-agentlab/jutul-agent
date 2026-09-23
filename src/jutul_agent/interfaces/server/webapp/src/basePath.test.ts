import { afterEach, describe, expect, it } from "vitest";

import { basePath, normalizeBasePath, withBase } from "./basePath";

describe("normalizeBasePath", () => {
  it("returns empty for blank or root", () => {
    expect(normalizeBasePath("")).toBe("");
    expect(normalizeBasePath("/")).toBe("");
    expect(normalizeBasePath(null)).toBe("");
    expect(normalizeBasePath(undefined)).toBe("");
  });

  it("normalizes a leading slash and strips a trailing one", () => {
    expect(normalizeBasePath("restricted")).toBe("/restricted");
    expect(normalizeBasePath("/restricted/")).toBe("/restricted");
    expect(normalizeBasePath("/restricted")).toBe("/restricted");
  });
});

describe("withBase", () => {
  afterEach(() => {
    delete window.__JUTUL_BASE_PATH__;
  });

  it("leaves paths alone when no base is set", () => {
    expect(withBase("/sessions")).toBe("/sessions");
  });

  it("prefixes site-relative paths", () => {
    window.__JUTUL_BASE_PATH__ = "/restricted";
    expect(withBase("/sessions")).toBe("/restricted/sessions");
    expect(withBase("/live/s1/viz/x")).toBe("/restricted/live/s1/viz/x");
  });

  it("leaves absolute URLs alone", () => {
    window.__JUTUL_BASE_PATH__ = "/restricted";
    expect(withBase("http://127.0.0.1:9/viz")).toBe("http://127.0.0.1:9/viz");
  });
});

describe("basePath", () => {
  afterEach(() => {
    delete window.__JUTUL_BASE_PATH__;
  });

  it("reads the injected window value", () => {
    window.__JUTUL_BASE_PATH__ = "/restricted/";
    expect(basePath()).toBe("/restricted");
  });
});
