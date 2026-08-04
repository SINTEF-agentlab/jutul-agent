import { act, render } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { View } from "../store";
import { IframePanel } from "./registry";

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
