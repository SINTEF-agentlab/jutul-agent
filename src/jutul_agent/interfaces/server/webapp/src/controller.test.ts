import { describe, expect, it, vi } from "vitest";

import { ApiError, api } from "./api";
import { Controller, preparePopoutDocument } from "./controller";
import { createSessionStore } from "./store";

describe("plot popout placeholder", () => {
  it("treats a model-provided title as text, never markup", () => {
    const doc = document.implementation.createHTMLDocument();
    preparePopoutDocument(doc, '<script data-test="payload">boom()</script>');

    expect(doc.title).toBe('<script data-test="payload">boom()</script>');
    expect(doc.querySelector("script")).toBeNull();
    expect(doc.body.textContent).toBe("Preparing an independent view…");
  });
});

describe("model catalog refresh", () => {
  it("adds API-listed models to the picker after startup", async () => {
    const staticModel = { id: "openai:gpt-5.4-mini", label: "gpt-5.4-mini", provider: "openai" };
    const liveModel = {
      id: "openai:gpt-6-luna",
      label: "gpt-6-luna",
      provider: "openai",
      note: "API-listed · tool support unverified",
    };
    const spies = [
      vi.spyOn(api, "simulators").mockResolvedValue({
        simulators: ["jutuldarcy"], default: "jutuldarcy", details: {},
      }),
      vi.spyOn(api, "models").mockResolvedValue({
        default: staticModel.id, providers: ["openai"], models: [staticModel],
      }),
      vi.spyOn(api, "credentials").mockResolvedValue([]),
      vi.spyOn(api, "liveModels").mockResolvedValue({ models: [staticModel, liveModel] }),
    ];
    const store = createSessionStore();
    const controller = new Controller(store);
    const start = vi.spyOn(controller, "startSession").mockResolvedValue();
    const history = vi.spyOn(controller, "refreshHistory").mockResolvedValue();
    const context = vi.spyOn(controller, "refreshContextWindow").mockResolvedValue();

    try {
      await controller.init();
      await vi.waitFor(() => expect(store.getState().models).toContainEqual(liveModel));
    } finally {
      for (const spy of [...spies, start, history, context]) spy.mockRestore();
    }
  });
});

describe("session startup failures", () => {
  it("shows the server refusal and leaves the composer idle", async () => {
    const create = vi
      .spyOn(api, "createSession")
      .mockRejectedValue(new ApiError(409, "Rebuild it with: jutul-agent sysimage build"));
    const store = createSessionStore();
    store.setState({ sim: "jutuldarcy", busy: true, working: true });
    const controller = new Controller(store);

    await controller.startSession();

    const state = store.getState();
    expect(state.busy).toBe(false);
    expect(state.working).toBe(false);
    expect(state.items.at(-1)).toMatchObject({
      kind: "sys-note",
      text: "Rebuild it with: jutul-agent sysimage build",
      level: "warn",
    });
    create.mockRestore();
  });

  it("retries session creation for a prompt sent after startup failed", async () => {
    const create = vi
      .spyOn(api, "createSession")
      .mockRejectedValue(new ApiError(409, "The system image is stale."));
    const store = createSessionStore();
    store.setState({ sim: "jutuldarcy" });
    const controller = new Controller(store);
    vi.spyOn(controller, "sendCanvasHint").mockImplementation(() => undefined);

    await controller.startSession();
    controller.send("Run the model");
    await vi.waitFor(() => expect(create).toHaveBeenCalledTimes(2));
    await vi.waitFor(() => expect(store.getState().busy).toBe(false));

    expect(store.getState().working).toBe(false);
    expect(store.getState().items).toContainEqual(
      expect.objectContaining({ kind: "user", text: "Run the model" }),
    );
    create.mockRestore();
  });
});
