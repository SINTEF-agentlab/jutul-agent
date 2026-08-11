import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { framePool } from "./canvas/framePool";
import { createSessionStore } from "./store";
import type { SessionStore, ThreadItem } from "./store";
import type { StoreApi } from "zustand/vanilla";

let store: StoreApi<SessionStore>;
const state = () => store.getState();
const items = () => state().items;
const byKind = <K extends ThreadItem["kind"]>(kind: K) =>
  items().filter((it): it is Extract<ThreadItem, { kind: K }> => it.kind === kind);

beforeEach(() => {
  store = createSessionStore();
});

describe("assistant text streaming", () => {
  it("coalesces deltas into one assistant item and clears the thinking indicator", () => {
    state().beginWorking();
    state().handle({ type: "text", text: "Hel" });
    state().handle({ type: "text", text: "lo" });
    const msgs = byKind("assistant");
    expect(msgs).toHaveLength(1);
    expect(msgs[0].text).toBe("Hello");
    expect(state().working).toBe(false);
  });
});

describe("reasoning", () => {
  it("streams into a live block and collapses it when assistant prose starts", () => {
    state().handle({ type: "reasoning", text: "thinking " });
    state().handle({ type: "reasoning", text: "hard" });
    expect(byKind("reasoning")[0]).toMatchObject({ text: "thinking hard", live: true });
    state().handle({ type: "text", text: "Answer" });
    expect(byKind("reasoning")[0].live).toBe(false);
    expect(byKind("assistant")[0].text).toBe("Answer");
  });

  it("starts a fresh assistant bubble after interleaved reasoning, in order", () => {
    // text -> reasoning -> text must render the second text AFTER the reasoning,
    // not appended back into the first bubble.
    state().handle({ type: "text", text: "first" });
    state().handle({ type: "reasoning", text: "rethink" });
    state().handle({ type: "text", text: "second" });
    const assistants = byKind("assistant");
    expect(assistants.map((a) => a.text)).toEqual(["first", "second"]);
    // The reasoning sits between the two assistant items in the thread.
    const kinds = items().map((it) => it.kind);
    expect(kinds).toEqual(["assistant", "reasoning", "assistant"]);
  });
});

describe("tool lifecycle", () => {
  it("creates one card and ends with the final terminal output", () => {
    state().handle({ type: "tool", event: "requested", name: "run_julia", tool_call_id: "1", args: { code: "1+1" } });
    state().handle({ type: "tool", event: "delta", name: "run_julia", tool_call_id: "1", content: "running…" });
    state().handle({ type: "tool", event: "finished", name: "run_julia", tool_call_id: "1", content: "2" });
    const tools = byKind("tool");
    expect(tools).toHaveLength(1);
    expect(tools[0]).toMatchObject({ status: "done", output: "2", args: { code: "1+1" } });
  });

  it("honours per-tool policy: read_file shows a line note, not raw output", () => {
    state().handle({ type: "tool", event: "requested", name: "read_file", tool_call_id: "r", args: { file_path: "a.jl" } });
    state().handle({ type: "tool", event: "finished", name: "read_file", tool_call_id: "r", content: "l1\nl2\nl3" });
    expect(byKind("tool")[0]).toMatchObject({ status: "done", output: "", note: "3 lines" });
  });

  it("always surfaces error output", () => {
    state().handle({ type: "tool", event: "error", name: "ls", tool_call_id: "e", content: "boom" });
    expect(byKind("tool")[0]).toMatchObject({ status: "error", output: "boom" });
  });
});

describe("canvas views", () => {
  it("opens a plot view and adds an inline chip; same slot refreshes in place", () => {
    state().handle({ type: "viz", url: "/a", kind: "plot", slot: "fig", title: "First" });
    expect(state().canvasOpen).toBe(true);
    expect(state().activeView).toBe("slot:fig");
    expect(state().viewOrder).toEqual(["slot:fig"]);
    const nonce0 = state().views["slot:fig"].nonce;

    state().handle({ type: "viz", url: "/b", kind: "plot", slot: "fig", title: "Updated" });
    expect(state().viewOrder).toEqual(["slot:fig"]); // not stacked
    expect(state().views["slot:fig"].url).toBe("/b");
    expect(state().views["slot:fig"].nonce).toBe(nonce0 + 1); // forces a reload
    expect(byKind("viz-chip")).toHaveLength(2); // a fresh reference each time
  });

  it("removing the active view closes the canvas", () => {
    state().handle({ type: "viz", url: "/a", kind: "plot", slot: "fig" });
    state().removeView("slot:fig");
    expect(state().canvasOpen).toBe(false);
    expect(state().activeView).toBeNull();
  });
});

describe("artifacts", () => {
  it("an image artifact is an inline image item and claims no canvas tab", () => {
    // Its card shows the image in full, so a tab up front is one the user never
    // asked for, sitting next to the live views the canvas exists for.
    state().handle({ type: "artifact", url: "/p.png", mime: "image/png", caption: "Fig" });
    expect(byKind("artifact-image")[0]).toMatchObject({ url: "/p.png", title: "Fig" });
    expect(Object.keys(state().views)).toHaveLength(0);
    expect(state().canvasOpen).toBe(false);
  });
  it("opening an image registers its view on demand and shows it", () => {
    state().handle({ type: "artifact", url: "/p.png", mime: "image/png", caption: "Fig" });
    state().openImage("/p.png", "Fig");
    expect(state().views["url:/p.png"].kind).toBe("image");
    expect(state().activeView).toBe("url:/p.png");
    expect(state().canvasOpen).toBe(true);
    // Opening the same image again must not stack a duplicate tab.
    state().openImage("/p.png", "Fig");
    expect(state().viewOrder).toEqual(["url:/p.png"]);
  });
  it("a non-image artifact is just a file link, no view", () => {
    state().handle({ type: "artifact", url: "/f.csv", mime: "text/csv", caption: "Data" });
    expect(byKind("artifact-file")[0]).toMatchObject({ url: "/f.csv", caption: "Data" });
    expect(Object.keys(state().views)).toHaveLength(0);
  });
});

describe("interrupts", () => {
  it("records the pending approval and frees the composer", () => {
    state().beginWorking();
    state().handle({
      type: "interrupt",
      interrupt_id: "x",
      actions: [{ name: "execute" }],
      allowed_decisions: ["approve", "reject"],
      allowlist: [],
    });
    expect(state().busy).toBe(false);
    expect(state().pending?.actions[0].name).toBe("execute");
    state().clearInterrupt();
    expect(state().pending).toBeNull();
  });
});

describe("usage", () => {
  it("reports a percentage of the context window when known", () => {
    state().setContextWindow(1000);
    state().handle({ type: "usage", input_tokens: 250, output_tokens: 10, total_tokens: 260, model_calls: 1 });
    expect(state().usageLabel).toBe("25% ctx");
  });
  it("falls back to a token count without a window", () => {
    state().handle({ type: "usage", input_tokens: 2000, output_tokens: 0, total_tokens: 2000, model_calls: 1 });
    expect(state().usageLabel).toBe("2.0k ctx");
  });
});

describe("side outputs do not clear the thinking indicator", () => {
  it("keeps working through usage and viz, clears on text", () => {
    state().beginWorking();
    state().handle({ type: "usage", input_tokens: 1, output_tokens: 0, total_tokens: 1, model_calls: 1 });
    expect(state().working).toBe(true);
    state().handle({ type: "viz", url: "/a", kind: "plot" });
    expect(state().working).toBe(true);
    state().handle({ type: "text", text: "done" });
    expect(state().working).toBe(false);
  });
});

describe("turn end / error / notice", () => {
  it("turn_end clears busy and collapses live reasoning", () => {
    state().handle({ type: "reasoning", text: "mid" });
    state().beginWorking();
    state().handle({ type: "turn_end", text: "" });
    expect(state().busy).toBe(false);
    expect(state().working).toBe(false);
    expect(byKind("reasoning")[0].live).toBe(false);
  });

  it("error offers retry only after a prompt was sent", () => {
    state().handle({ type: "error", message: "first" });
    expect(byKind("error")[0].canRetry).toBe(false);
    state().startTurn("hello");
    state().handle({ type: "error", message: "second" });
    expect(byKind("error").at(-1)?.canRetry).toBe(true);
  });

  it("notice clears busy and adds a system note", () => {
    state().beginWorking();
    state().handle({ type: "notice", text: "Compacted." });
    expect(state().busy).toBe(false);
    expect(byKind("sys-note")[0].text).toBe("Compacted.");
  });
});

describe("ui history_changed is internal (no thread item)", () => {
  it("does not surface as a ui-note", () => {
    state().handle({ type: "ui", action: "history_changed", payload: { title: "x" } });
    expect(byKind("ui-note")).toHaveLength(0);
    state().handle({ type: "ui", action: "well_placed", payload: { id: 3 } });
    expect(byKind("ui-note")[0].action).toBe("well_placed");
  });
});

describe("replay", () => {
  it("reconstructs a recorded conversation and opens the last view", () => {
    state().replay([
      { type: "user", text: "hi" },
      { type: "assistant", text: "hello" },
      { type: "tool", event: "requested", name: "ls", tool_call_id: "1" },
      { type: "tool", event: "finished", name: "ls", tool_call_id: "1", content: "['a/']" },
      { type: "viz", url: "/p.png", kind: "plot", slot: "fig" },
    ]);
    expect(byKind("user")[0].text).toBe("hi");
    expect(byKind("assistant")[0].text).toBe("hello");
    expect(byKind("tool")[0].status).toBe("done");
    expect(state().canvasOpen).toBe(true);
    expect(state().liveAssistantId).toBeNull();
  });
});

describe("reset", () => {
  it("clears the thread and canvas but keeps config", () => {
    state().setConfig({ sim: "battmo", model: "anthropic:claude" });
    state().startTurn("hi");
    state().handle({ type: "viz", url: "/a", kind: "plot" });
    state().reset();
    expect(state().items).toHaveLength(0);
    expect(state().canvasOpen).toBe(false);
    expect(Object.keys(state().views)).toHaveLength(0);
    expect(state().sim).toBe("battmo");
    expect(state().model).toBe("anthropic:claude");
  });

  it("keeps the provider key status across a reset (it is account-wide)", () => {
    state().setCredentials([
      {
        provider: "openai",
        label: "OpenAI",
        env_var: "OPENAI_API_KEY",
        is_set: true,
        masked: "sk-***z",
        source: "file",
        shadowed: false,
      },
    ]);
    state().reset();
    expect(state().credentials).toHaveLength(1);
  });
});

describe("api-keys modal", () => {
  it("opens with a required provider and closes back to empty", () => {
    const required = { provider: "openai", label: "OpenAI", env_var: "OPENAI_API_KEY" };
    state().openApiKeys(required);
    expect(state().apiKeys).toEqual({ open: true, required });
    state().closeApiKeys();
    expect(state().apiKeys).toEqual({ open: false, required: null });
  });
});

describe("view sizing mode", () => {
  it("a viz carries its live/record/size fields into the view", () => {
    state().handle({
      type: "viz",
      url: "/live/s/viz/res",
      kind: "plot",
      slot: "res",
      live: true,
      record: "artifacts/res.png",
      width: 1600,
      height: 900,
    });
    const v = state().views["slot:res"];
    expect(v.live).toBe(true);
    expect(v.record).toBe("artifacts/res.png");
    expect([v.width, v.height]).toEqual([1600, 900]);
    expect(v.mode).toBeUndefined(); // unset: the stage defaults to "scale"
  });

  it("setViewMode flips the mode and a same-slot refresh keeps the choice", () => {
    state().handle({ type: "viz", url: "/a", kind: "plot", slot: "res", width: 800, height: 600 });
    state().setViewMode("slot:res", "fill");
    expect(state().views["slot:res"].mode).toBe("fill");
    state().handle({ type: "viz", url: "/b", kind: "plot", slot: "res", width: 800, height: 600 });
    expect(state().views["slot:res"].mode).toBe("fill"); // the user's choice survives
    expect(state().views["slot:res"].url).toBe("/b");
  });
});

describe("replayed live views and the frame pool", () => {
  afterEach(() => framePool.clear());

  const liveViz = {
    type: "viz" as const,
    url: "/live/s1/viz/res",
    kind: "plot",
    slot: "res",
    live: true,
    poster: "/sessions/s1/artifacts/res.png",
    record: "artifacts/res.png",
    width: 1600,
    height: 900,
  };

  it("a replayed live view stays live only when its frame survives in the pool", () => {
    state().setSession("s1", "");
    framePool.register({
      sessionId: "s1",
      viewId: "slot:res",
      url: liveViz.url,
      title: "res",
      nonce: 2,
    });
    state().replay([liveViz]);
    const v = state().views["slot:res"];
    expect(v.live).toBe(true);
    expect(v.url).toBe(liveViz.url);
    expect(v.nonce).toBe(2); // aligned with the mounted frame's reload token
  });

  it("a replayed live view without its frame falls back to the poster", () => {
    state().setSession("s1", "");
    state().replay([liveViz]);
    const v = state().views["slot:res"];
    expect(v.live).toBe(false);
    expect(v.url).toBe(liveViz.poster);
    expect(v.record).toBe("artifacts/res.png"); // regenerate stays available
  });

  it("a replayed live view with neither frame nor poster is marked expired", () => {
    state().setSession("s1", "");
    state().replay([{ ...liveViz, poster: null }]);
    const v = state().views["slot:res"];
    expect(v.live).toBe(false);
    expect(v.expired).toBe(true);
  });

  it("downgradeView releases the frame and falls back to the poster", () => {
    state().setSession("s1", "");
    state().handle(liveViz);
    framePool.register({
      sessionId: "s1",
      viewId: "slot:res",
      url: liveViz.url,
      title: "res",
      nonce: 0,
    });
    state().downgradeView("slot:res");
    expect(framePool.has("s1", "slot:res", liveViz.url)).toBe(false);
    const v = state().views["slot:res"];
    expect(v.live).toBe(false);
    expect(v.url).toBe(liveViz.poster);
  });

  it("removing a view releases its pooled frame", () => {
    state().setSession("s1", "");
    state().handle(liveViz);
    framePool.register({
      sessionId: "s1",
      viewId: "slot:res",
      url: liveViz.url,
      title: "res",
      nonce: 0,
    });
    state().removeView("slot:res");
    expect(framePool.has("s1", "slot:res", liveViz.url)).toBe(false);
  });
});
