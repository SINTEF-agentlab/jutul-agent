// stageSize answers "how big is the plot area", measured from a visible stage
// when there is one, otherwise computed the way the CSS will lay the canvas out.

import { afterEach, describe, expect, it } from "vitest";

import { CANVAS_W_KEY, stageSize } from "./stageSize";

afterEach(() => {
  document.body.innerHTML = "";
  localStorage.clear();
});

function setViewport(w: number, h: number) {
  Object.defineProperty(window, "innerWidth", { value: w, configurable: true });
  Object.defineProperty(window, "innerHeight", { value: h, configurable: true });
}

describe("stageSize", () => {
  it("prefers a visible stage's own measurement", () => {
    const el = document.createElement("div");
    el.className = "canvas-stage active";
    Object.defineProperty(el, "clientWidth", { value: 640 });
    Object.defineProperty(el, "clientHeight", { value: 720 });
    document.body.appendChild(el);
    expect(stageSize()).toEqual({ width: 640, height: 720 });
  });

  it("computes the prospective split when no stage is visible", () => {
    setViewport(2000, 1000);
    localStorage.setItem(CANVAS_W_KEY, "0.5");
    expect(stageSize()).toEqual({ width: 1000, height: 952 });
  });

  it("clamps a wild persisted split and defaults without one", () => {
    setViewport(2000, 1000);
    localStorage.setItem(CANVAS_W_KEY, "0.95");
    expect(stageSize().width).toBe(Math.round(2000 * 0.62));
    localStorage.removeItem(CANVAS_W_KEY);
    expect(stageSize().width).toBe(Math.round(2000 * 0.46));
  });

  it("uses the overlay width on a narrow viewport", () => {
    setViewport(900, 800);
    expect(stageSize()).toEqual({ width: 640, height: 752 });
  });

  it("ignores a degenerate (hidden) stage measurement", () => {
    const el = document.createElement("div");
    el.className = "canvas-stage active"; // mounted but effectively unmeasured
    document.body.appendChild(el);
    setViewport(2000, 1000);
    expect(stageSize().width).toBe(Math.round(2000 * 0.46));
  });
});
