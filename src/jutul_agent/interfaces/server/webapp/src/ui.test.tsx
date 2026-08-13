import { act, fireEvent, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { api, type HistoryEntry } from "./api";
import { Canvas } from "./components/Canvas";
import { Sidebar } from "./components/Sidebar";
import { Thread } from "./components/Thread";
import type { ServerMessage } from "./protocol";
import { renderWithStore } from "./test/util";

function drive(store: { getState: () => { handle: (m: ServerMessage) => void } }, ...msgs: ServerMessage[]) {
  act(() => {
    for (const m of msgs) store.getState().handle(m);
  });
}

describe("Thread rendering", () => {
  it("shows the welcome screen when empty", () => {
    renderWithStore(<Thread />);
    expect(screen.getByText(/What would you like to explore/)).toBeInTheDocument();
  });

  it("lets an installed capability name the welcome screen and set its prompts", () => {
    const { store } = renderWithStore(<Thread />);
    act(() =>
      store.getState().setConfig({
        sim: "jutuldarcy",
        simDetails: { jutuldarcy: { display_name: "JutulDarcy", examples: ["Sim prompt."] } },
        branding: { display_name: "MyApp", tagline: "Build a model.", examples: ["Build the model."] },
      }),
    );
    expect(screen.getByText(/explore with MyApp/)).toBeInTheDocument();
    expect(screen.getByText("Build a model.")).toBeInTheDocument();
    expect(screen.getByText("Build the model.")).toBeInTheDocument();
    expect(screen.queryByText("Sim prompt.")).toBeNull();
  });

  it("falls back to the simulator when nothing is branded", () => {
    const { store } = renderWithStore(<Thread />);
    act(() =>
      store.getState().setConfig({
        sim: "jutuldarcy",
        simDetails: { jutuldarcy: { display_name: "JutulDarcy", examples: ["Sim prompt."] } },
      }),
    );
    expect(screen.getByText(/explore with JutulDarcy/)).toBeInTheDocument();
    expect(screen.getByText("Sim prompt.")).toBeInTheDocument();
  });

  it("renders a streamed exchange: user, assistant, tool", () => {
    const { store } = renderWithStore(<Thread />);
    act(() => store.getState().addUser("run a sim"));
    drive(
      store,
      { type: "text", text: "On it." },
      { type: "tool", event: "requested", name: "run_julia", label: "run_julia", tool_call_id: "1", args: { code: "1+1" } },
      { type: "tool", event: "finished", name: "run_julia", tool_call_id: "1", content: "2" },
    );
    expect(screen.getByText("run a sim")).toBeInTheDocument();
    expect(screen.getByText("On it.")).toBeInTheDocument();
    expect(screen.getByText("run_julia")).toBeInTheDocument();
    expect(screen.getByText("done")).toBeInTheDocument();
  });

  it("counts up on a tool that is still running, and stops when it finishes", () => {
    vi.useFakeTimers();
    try {
      const { store, container } = renderWithStore(<Thread />);
      drive(store, {
        type: "tool",
        event: "requested",
        name: "setup_mesh",
        label: "setup_mesh",
        tool_call_id: "1",
        args: {},
      });
      // Nothing yet: a number that flashes up and vanishes is noise.
      expect(container.querySelector(".elapsed")).toBeNull();
      act(() => void vi.advanceTimersByTime(65_000));
      expect(container.querySelector(".elapsed")?.textContent).toBe("1:05");
      drive(store, { type: "tool", event: "finished", name: "setup_mesh", tool_call_id: "1", content: "ok" });
      expect(container.querySelector(".elapsed")).toBeNull();
    } finally {
      vi.useRealTimers();
    }
  });

  it("shows the thinking indicator only while working", () => {
    const { store, container } = renderWithStore(<Thread />);
    act(() => store.getState().beginWorking());
    expect(container.querySelector(".working")).toBeInTheDocument();
    drive(store, { type: "text", text: "hi" });
    expect(container.querySelector(".working")).toBeNull();
  });
});

describe("canvas interaction", () => {
  it("a viz chip opens the canvas to that view", () => {
    const { store } = renderWithStore(
      <>
        <Thread />
        <Canvas />
      </>,
    );
    drive(store, { type: "viz", url: "/plot.html", kind: "plot", slot: "fig", title: "My figure" });
    // Opening happens in the store on viz; closing then re-opening via the chip.
    act(() => store.getState().closeCanvas());
    expect(store.getState().canvasOpen).toBe(false);
    fireEvent.click(screen.getByText("My figure", { selector: ".viz-chip .t" }));
    expect(store.getState().canvasOpen).toBe(true);
    expect(store.getState().activeView).toBe("slot:fig");
  });

  it("a resumed session's poster clears the loader without looping", () => {
    // Resuming demotes a live figure to its PNG poster, and a poster the browser
    // has already decoded reports its load from the panel's mount check. Since
    // that report re-renders the canvas, the whole path has to settle rather than
    // feed itself another report.
    const complete = vi.spyOn(HTMLImageElement.prototype, "complete", "get").mockReturnValue(true);
    const { store, container } = renderWithStore(<Canvas />);
    act(() => store.getState().setSession("s1", ""));
    act(() =>
      store.getState().replay([
        {
          type: "viz",
          url: "/sessions/s1/viz/abc",
          kind: "plot",
          slot: "fig",
          title: "My figure",
          live: true,
          poster: "/sessions/s1/artifacts/fig.png",
        },
      ]),
    );
    expect(container.querySelector("img")?.getAttribute("src")).toBe("/sessions/s1/artifacts/fig.png");
    expect(container.querySelector(".canvas-loading")).toBeNull();
    complete.mockRestore();
  });
});

describe("approval", () => {
  it("renders the request and clears it on approve", () => {
    const { store } = renderWithStore(<Thread />);
    drive(store, {
      type: "interrupt",
      interrupt_id: "x",
      actions: [{ name: "execute", label: "execute", args: { command: "ls" } }],
      allowed_decisions: ["approve", "reject"],
      allowlist: [],
    });
    expect(screen.getByText(/Approve execute\?/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "approve" }));
    expect(store.getState().pending).toBeNull();
    expect(screen.queryByText(/Approve execute\?/)).toBeNull();
  });
});

describe("session list live markers", () => {
  const entry = (id: string, extra: Partial<HistoryEntry> = {}): HistoryEntry => ({
    id,
    title: `Session ${id}`,
    started: "2026-01-01T00:00:00Z",
    last_active: "2026-01-01T00:00:00Z",
    sim: "jutuldarcy",
    ...extra,
  });

  function renderList(history: HistoryEntry[]) {
    const rendered = renderWithStore(<Sidebar collapsed={false} />);
    act(() => rendered.store.getState().setHistory(history));
    return rendered;
  }

  it("marks only the sessions the server calls live", () => {
    const { container } = renderList([
      entry("a", { live: true }),
      entry("b"),
      entry("c", { live: false }),
    ]);
    const items = container.querySelectorAll(".history-item");
    expect(items.length).toBe(3);
    expect(items[0].querySelector(".h-live")).not.toBeNull();
    expect(items[1].querySelector(".h-live")).toBeNull();
    expect(items[2].querySelector(".h-live")).toBeNull();
  });

  it("labels the marker for screen readers and on hover", () => {
    const { container } = renderList([entry("a", { live: true })]);
    const dot = container.querySelector(".h-live");
    expect(dot?.getAttribute("aria-label")).toMatch(/Julia session intact/);
    expect(dot?.getAttribute("title")).toBe(dot?.getAttribute("aria-label"));
  });

  it("keeps the title truncatable next to the marker", () => {
    // The dot turned .h-title into a flex row, which cannot ellipsis a bare text
    // node, so the title has to keep its own element.
    const { container } = renderList([entry("a", { live: true })]);
    expect(container.querySelector(".h-title-text")?.textContent).toBe("Session a");
  });
});

describe("session list freshness", () => {
  const listed = (id: string): HistoryEntry => ({
    id,
    title: `Session ${id}`,
    started: "2026-01-01T00:00:00Z",
    last_active: "2026-01-01T00:00:00Z",
    sim: "jutuldarcy",
    live: true,
  });

  it("re-reads the list when a new chat replaces the current session", async () => {
    // newChat used to leave the sidebar showing whatever it had until the next
    // turn ended, so the session just left kept a stale place and marker.
    const { store, controller } = renderWithStore(<Sidebar collapsed={false} />);
    const history = vi.spyOn(api, "history").mockResolvedValue([listed("a")]);
    vi.spyOn(controller, "startSession").mockResolvedValue(undefined);
    await act(() => controller.newChat());
    expect(history).toHaveBeenCalled();
    expect(store.getState().history.map((h) => h.id)).toEqual(["a"]);
    vi.restoreAllMocks();
  });

  it("re-reads the list when the tab comes back to the front", async () => {
    // A host can be evicted while the tab is in the background, which clears a
    // live marker this client never hears about.
    renderWithStore(<Sidebar collapsed={false} />);
    const history = vi.spyOn(api, "history").mockResolvedValue([listed("a")]);
    document.dispatchEvent(new Event("visibilitychange"));
    await act(async () => {});
    expect(history).toHaveBeenCalled();
    vi.restoreAllMocks();
  });
});

describe("switching sessions", () => {
  it("marks the resumed session live without waiting for the replay", async () => {
    // The refresh used to run after the conversation had been re-fetched and laid
    // down, so on a long chat the dot appeared well after the click. Liveness is
    // already settled once the resume answers.
    const { controller } = renderWithStore(<Sidebar collapsed={false} />);
    vi.spyOn(api, "resumeSession").mockResolvedValue({
      session_id: "a",
      kernel_restarted: true,
    });
    // A replay that never settles stands in for a slow one.
    vi.spyOn(api, "messages").mockReturnValue(new Promise(() => {}));
    const history = vi.spyOn(api, "history").mockResolvedValue([]);

    void controller.resume("a");
    await act(async () => {});

    expect(history).toHaveBeenCalled();
    vi.restoreAllMocks();
  });
});
