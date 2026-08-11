import { act, render } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { View } from "../store";
import { IframePanel, stageGeometry } from "./registry";

// A freshly-registered live plot's Bonito route can take a couple of seconds
// to start answering (see the comment in registry.tsx). A failed iframe
// navigation doesn't fire a DOM `error` event, so readiness is checked with a
// `fetch` ping instead. These tests drive that ping directly (mocking global
// fetch) rather than the iframe's load/error events.

const plotView: View = { id: "v1", url: "http://127.0.0.1:1/viz/abc", title: "plot", kind: "plot", nonce: 0 };
const reportView: View = { id: "v2", url: "http://127.0.0.1:1/report", title: "report", kind: "report", nonce: 0 };
const DELAYS = [400, 800, 1200, 2000, 3000];

describe("IframePanel live readiness", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it("does not mount the iframe until the readiness ping succeeds", async () => {
    const fetchMock = vi.fn().mockRejectedValueOnce(new TypeError("network error")).mockResolvedValueOnce({});
    vi.stubGlobal("fetch", fetchMock);
    const onLoaded = vi.fn();
    const { container } = render(
      <IframePanel view={plotView} active reloadToken={0} onLoaded={onLoaded} />,
    );
    expect(container.querySelector("iframe")).toBeNull();

    await act(async () => {
      await Promise.resolve(); // let the first ping's rejection settle
    });
    expect(container.querySelector("iframe")).toBeNull();
    expect(fetchMock).toHaveBeenCalledTimes(1);

    await act(async () => {
      vi.advanceTimersByTime(400);
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(container.querySelector("iframe")).not.toBeNull();
  });

  it("gives up after the retry budget and mounts the iframe anyway", async () => {
    const fetchMock = vi.fn().mockRejectedValue(new TypeError("network error"));
    vi.stubGlobal("fetch", fetchMock);
    const { container } = render(
      <IframePanel view={plotView} active reloadToken={0} onLoaded={vi.fn()} />,
    );

    for (const delay of DELAYS) {
      await act(async () => {
        await Promise.resolve();
      });
      expect(container.querySelector("iframe")).toBeNull();
      await act(async () => {
        vi.advanceTimersByTime(delay);
      });
    }
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(container.querySelector("iframe")).not.toBeNull();
  });

  it("re-probes when reloadToken changes", async () => {
    const fetchMock = vi.fn().mockResolvedValue({});
    vi.stubGlobal("fetch", fetchMock);
    const { container, rerender } = render(
      <IframePanel view={plotView} active reloadToken={0} onLoaded={vi.fn()} />,
    );
    await act(async () => {
      await Promise.resolve();
    });
    expect(container.querySelector("iframe")).not.toBeNull();
    expect(fetchMock).toHaveBeenCalledTimes(1);

    await act(async () => {
      rerender(<IframePanel view={plotView} active reloadToken={1} onLoaded={vi.fn()} />);
      await Promise.resolve();
    });
    // A fresh probe ran (not just reusing the earlier "ready" result), and
    // the iframe is back once it resolves.
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(container.querySelector("iframe")).not.toBeNull();
  });

  it("skips the readiness ping for non-plot iframes (static reports)", () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    const { container } = render(
      <IframePanel view={reportView} active reloadToken={0} onLoaded={vi.fn()} />,
    );
    expect(fetchMock).not.toHaveBeenCalled();
    expect(container.querySelector("iframe")).not.toBeNull();
  });
});

describe("stageGeometry", () => {
  it("scale fits the figure inside the box, centered, aspect preserved", () => {
    // A 1600x900 figure in a 736x952 panel: width-limited, so scale = 736/1600.
    const g = stageGeometry("scale", 1600, 900, 736, 952);
    expect(g.scale).toBeCloseTo(0.46, 2);
    expect(g.width).toBe(736);
    expect(g.height).toBe(Math.round(900 * (736 / 1600)));
    expect(g.left).toBe(0);
    expect(g.top).toBe(Math.round((952 - g.height) / 2));
  });

  it("scale never upscales past the figure's own size", () => {
    // A canvas grown by CSS blurs, so a panel larger than the figure centers it 1:1.
    const g = stageGeometry("scale", 800, 600, 2000, 1500);
    expect(g.scale).toBe(1);
    expect(g.width).toBe(800);
    expect(g.height).toBe(600);
    expect(g.left).toBe(600);
    expect(g.top).toBe(450);
  });

  it("fill hands the figure the whole box", () => {
    expect(stageGeometry("fill", 1600, 900, 736, 952)).toEqual({
      left: 0,
      top: 0,
      width: 736,
      height: 952,
      scale: 1,
    });
  });

  it("an unmeasured box degrades to fill-at-zero, not NaN", () => {
    const g = stageGeometry("scale", 1600, 900, 0, 0);
    expect(g).toEqual({ left: 0, top: 0, width: 0, height: 0, scale: 1 });
  });
});

describe("IframePanel stage", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    // jsdom lays nothing out, so the stage would measure 0x0; give every element
    // a panel-sized client box for these tests.
    Object.defineProperty(HTMLElement.prototype, "clientWidth", { configurable: true, value: 736 });
    Object.defineProperty(HTMLElement.prototype, "clientHeight", { configurable: true, value: 952 });
  });
  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
    delete (HTMLElement.prototype as { clientWidth?: number }).clientWidth;
    delete (HTMLElement.prototype as { clientHeight?: number }).clientHeight;
  });

  it("a sized live plot renders at its design size inside a scaled frame", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({}));
    const sized: View = { ...plotView, live: true, width: 1600, height: 900 };
    const { container } = render(
      <IframePanel view={sized} active reloadToken={0} onLoaded={() => {}} />,
    );
    await act(async () => {
      await Promise.resolve();
    });
    const iframe = container.querySelector("iframe")!;
    // The frame's inner viewport is the figure's design size; the CSS transform
    // does the shrinking, so the served layout is never crushed.
    expect(iframe.style.width).toBe("1600px");
    expect(iframe.style.height).toBe("900px");
    expect(iframe.style.transform).toContain("scale(");
    expect(container.querySelector(".canvas-stage .stage-frame")).not.toBeNull();
  });

  it("a plot without a recorded size fills the stage", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({}));
    const { container } = render(
      <IframePanel view={plotView} active reloadToken={0} onLoaded={() => {}} />,
    );
    await act(async () => {
      await Promise.resolve();
    });
    const iframe = container.querySelector("iframe")!;
    expect(iframe.style.transform).toBe("");
  });

  it("never first-mounts hidden; stays mounted once shown", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({}));
    const sized: View = { ...plotView, live: true, width: 1600, height: 900 };
    const { container, rerender } = render(
      <IframePanel view={sized} active={false} reloadToken={0} onLoaded={() => {}} />,
    );
    await act(async () => {
      await Promise.resolve();
    });
    expect(container.querySelector("iframe")).toBeNull(); // hidden: not mounted yet
    rerender(<IframePanel view={sized} active reloadToken={0} onLoaded={() => {}} />);
    await act(async () => {
      await Promise.resolve();
    });
    expect(container.querySelector("iframe")).not.toBeNull();
    rerender(<IframePanel view={sized} active={false} reloadToken={0} onLoaded={() => {}} />);
    expect(container.querySelector("iframe")).not.toBeNull(); // shown once: stays mounted
  });
});
