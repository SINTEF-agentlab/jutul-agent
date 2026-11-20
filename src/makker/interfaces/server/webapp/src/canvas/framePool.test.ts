import { afterEach, describe, expect, it } from "vitest";

import { LIVE_VIEW_CAP, framePool } from "./framePool";

const frame = (viewId: string, nonce = 0, url = `/live/s/viz/${viewId}`) => ({
  sessionId: "s1",
  viewId,
  url,
  title: viewId,
  nonce,
});

afterEach(() => framePool.clear());

describe("framePool", () => {
  it("a higher nonce replaces the view's frame; an equal one adopts it", () => {
    framePool.register(frame("res", 0));
    const k0 = framePool.snapshot()[0].key;
    // Replay after a session switch registers nonce 0 again: adopt, no remount.
    framePool.register(frame("res", 0));
    expect(framePool.snapshot()).toHaveLength(1);
    expect(framePool.snapshot()[0].key).toBe(k0);
    // A deliberate same-slot re-plot bumps the nonce: the frame must reload.
    framePool.register(frame("res", 1));
    expect(framePool.snapshot()).toHaveLength(1);
    expect(framePool.snapshot()[0].key).not.toBe(k0);
  });

  it("adoption refreshes title and size but keeps the mounted frame", () => {
    framePool.register(frame("res", 2));
    framePool.register({ ...frame("res", 0), title: "renamed", width: 800, height: 600 });
    const [f] = framePool.snapshot();
    expect(f.nonce).toBe(2); // the mounted frame, not the replayed nonce
    expect(f.title).toBe("renamed");
    expect(f.width).toBe(800);
    expect(framePool.nonceOf("s1", "res", f.url)).toBe(2);
  });

  it("releases the oldest frames past the cap and reports them", () => {
    for (let i = 0; i < LIVE_VIEW_CAP; i++) framePool.register(frame(`v${i}`));
    expect(framePool.snapshot()).toHaveLength(LIVE_VIEW_CAP);
    const released = framePool.register(frame("one-more"));
    expect(released.map((f) => f.viewId)).toEqual(["v0"]);
    expect(framePool.snapshot()).toHaveLength(LIVE_VIEW_CAP);
    expect(framePool.has("s1", "v0", "/live/s/viz/v0")).toBe(false);
  });

  it("release drops every frame of a view; other sessions are untouched", () => {
    framePool.register(frame("res"));
    framePool.register({ ...frame("res"), sessionId: "s2" });
    framePool.release("s1", "res");
    expect(framePool.has("s1", "res", "/live/s/viz/res")).toBe(false);
    expect(framePool.has("s2", "res", "/live/s/viz/res")).toBe(true);
  });
});
