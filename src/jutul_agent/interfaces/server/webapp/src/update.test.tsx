import { act, fireEvent, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiError, api, type UpdateJob, type UpdateStatus } from "./api";
import { UpdateDialog, describe as plan } from "./components/UpdateDialog";
import { renderWithStore } from "./test/util";

const status = (patch: Partial<UpdateStatus> = {}): UpdateStatus => ({
  current: "0.2.1",
  latest: "0.2.2",
  local: true,
  can_install: true,
  install_note: null,
  separate_window: false,
  env: { outdated: [], sysimage: false },
  job: null,
  ...patch,
});

const job = (patch: Partial<UpdateJob>): UpdateJob => ({
  state: "running",
  install: true,
  error: null,
  lines: [],
  next: 0,
  ...patch,
});

const outdated = { outdated: [{ package: "JutulDarcy", version: "0.3.12", required: "0.4" }] };

describe("what the update dialog offers", () => {
  it("updates jutul-agent, then the environment", () => {
    const p = plan(status());
    expect(p.action).toEqual({ install: true, label: "Update" });
    expect(p.lines[0]).toBe("jutul-agent 0.2.2 is available. This is 0.2.1.");
  });

  it("says how to update when this install can't do it itself", () => {
    const note = "On Windows, jutul-agent can't replace its own files while it runs.";
    const p = plan(status({ can_install: false, install_note: note }));
    expect(p.action).toBeNull();
    expect(p.note).toBe(note);
    // ...but still brings the environment up to the running version.
    const env = plan(
      status({ can_install: false, install_note: note, env: { ...outdated, sysimage: false } }),
    );
    expect(env.action).toEqual({ install: false, label: "Update environment" });
  });

  it("updates only the environment when no release is due", () => {
    const p = plan(status({ latest: null, env: { ...outdated, sysimage: false } }));
    expect(p.title).toBe("Update the Julia environment");
    expect(p.lines[0]).toContain("has JutulDarcy 0.3.12");
    expect(p.action?.install).toBe(false);
  });

  it("offers nothing to a page served to the network", () => {
    const remote = status({ latest: null, local: false, env: { ...outdated, sysimage: false } });
    expect(plan(remote).action).toBeNull();
  });
});

describe("running an update", () => {
  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it("asks about the system image, streams the output, and ends", async () => {
    vi.useFakeTimers();
    const { controller, container } = renderWithStore(<UpdateDialog />);
    vi.spyOn(api, "updateStatus")
      .mockResolvedValueOnce(status({ latest: null, env: { ...outdated, sysimage: true } }))
      .mockResolvedValueOnce(status({ job: job({ install: false, lines: ["one"], next: 1 }) }))
      .mockResolvedValueOnce(
        status({ job: job({ install: false, state: "done", lines: ["two"], next: 2 }) }),
      );
    const start = vi.spyOn(api, "startUpdate").mockResolvedValue(status({ job: job({}) }));
    vi.spyOn(api, "version").mockResolvedValue({
      jutul_agent: "0.2.1",
      update: null,
      simulator: null,
    });

    await act(async () => controller.openUpdate());
    fireEvent.click(screen.getByLabelText(/Stop using one/));
    await act(async () => {
      fireEvent.click(screen.getByText("Update environment"));
    });
    expect(start).toHaveBeenCalledWith({ install: false, sysimage: "off" });
    // No way out while it runs.
    expect(screen.queryByText("Close")).toBeNull();

    await act(async () => vi.advanceTimersByTimeAsync(2100));
    expect(container.querySelector(".update-log")?.textContent).toBe("one\ntwo");
    expect(screen.getByText(/Done/)).toBeTruthy();
    expect(screen.getByText("Close")).toBeTruthy();
  });

  it("keeps the session when the server refuses, and says why", async () => {
    const { controller, container } = renderWithStore(<UpdateDialog />);
    vi.spyOn(api, "updateStatus").mockResolvedValue(status());
    vi.spyOn(api, "startUpdate").mockRejectedValue(
      new ApiError(409, "A session is starting or working. Stop it or let it finish, then update."),
    );
    const close = vi.spyOn(controller.transport, "close");

    await act(async () => controller.openUpdate());
    await act(async () => {
      fireEvent.click(screen.getByText("Update"));
    });

    expect(close).not.toHaveBeenCalled();
    expect(container.querySelector(".update-error")?.textContent).toContain("Stop it");
    // Still on the confirm screen, so it can be tried again.
    expect(screen.getByText("Update")).toBeTruthy();
  });

  it("reloads into the new server once it is back", async () => {
    vi.useFakeTimers();
    const { controller, store } = renderWithStore(<UpdateDialog />);
    store.setState({
      versions: { jutul_agent: "0.2.1", update: "0.2.2", simulator: null, boot: "a" },
    });
    const reload = vi.fn();
    vi.spyOn(window, "location", "get").mockReturnValue({ reload } as unknown as Location);
    vi.spyOn(api, "updateStatus")
      .mockResolvedValueOnce(status({ job: job({ state: "restarting", next: 0 }) }))
      .mockResolvedValue(null); // the server went away
    vi.spyOn(api, "startUpdate").mockResolvedValue(status({ job: job({}) }));
    const version = vi
      .spyOn(api, "version")
      .mockResolvedValueOnce({ jutul_agent: null, update: null, simulator: null })
      .mockResolvedValue({ jutul_agent: "0.2.2", update: null, simulator: null, boot: "b" });

    await act(async () => {
      void controller.startUpdate(true, null);
      await vi.advanceTimersByTimeAsync(6000);
    });

    expect(version).toHaveBeenCalled();
    expect(reload).toHaveBeenCalledTimes(1);
    expect(sessionStorage.getItem("ja_updated")).toBe("0.2.2");
    sessionStorage.clear();
  });

  it("waits for an update that continues in a separate window", async () => {
    vi.useFakeTimers();
    const { controller, store } = renderWithStore(<UpdateDialog />);
    store.setState({
      versions: { jutul_agent: "0.2.1", update: "0.2.2", simulator: null, boot: "a" },
    });
    const separate = (patch: Partial<UpdateStatus> = {}) =>
      status({ separate_window: true, ...patch });
    vi.spyOn(api, "updateStatus")
      .mockResolvedValueOnce(separate())
      .mockResolvedValueOnce(separate({ job: job({ state: "restarting", next: 0 }) }))
      .mockResolvedValue(null); // the server exited
    vi.spyOn(api, "startUpdate").mockResolvedValue(separate({ job: job({}) }));
    vi.spyOn(api, "version").mockResolvedValue({ jutul_agent: null, update: null, simulator: null });

    await act(async () => controller.openUpdate());
    expect(screen.getByText(/updates in a separate window/)).toBeTruthy();
    await act(async () => {
      fireEvent.click(screen.getByText("Update"));
      await vi.advanceTimersByTimeAsync(2500);
    });
    expect(screen.getByRole("status").textContent).toContain("updating in a separate window");
    // The whole update runs before the server is back, so the page waits longer.
    await act(async () => vi.advanceTimersByTimeAsync(30 * 60_000));
    expect(store.getState().update.phase).toBe("restarting");
  });
});

describe("after the reload into the restarted server", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    sessionStorage.clear();
  });

  async function reload(outcome: UpdateJob) {
    sessionStorage.setItem("ja_updated", "0.2.2");
    const rendered = renderWithStore(<UpdateDialog />);
    const { controller } = rendered;
    vi.spyOn(api, "simulators").mockResolvedValue({
      simulators: ["jutuldarcy"],
      default: "jutuldarcy",
      details: {},
    });
    vi.spyOn(api, "models").mockResolvedValue({ default: null, providers: [], models: [] });
    vi.spyOn(api, "credentials").mockResolvedValue([]);
    vi.spyOn(api, "liveModels").mockResolvedValue({ models: [] });
    vi.spyOn(api, "version").mockResolvedValue({
      jutul_agent: "0.2.2",
      update: null,
      simulator: null,
    });
    vi.spyOn(api, "updateStatus").mockResolvedValue(status({ latest: null, job: outcome }));
    vi.spyOn(controller, "startSession").mockResolvedValue();
    vi.spyOn(controller, "refreshHistory").mockResolvedValue();
    vi.spyOn(controller, "refreshContextWindow").mockResolvedValue();
    await act(async () => controller.init());
    return rendered;
  }

  it("says it was updated", async () => {
    const { store } = await reload(job({ state: "done" }));
    expect(store.getState().update.open).toBe(false);
    expect(store.getState().items.at(-1)).toMatchObject({ text: "Updated to jutul-agent 0.2.2." });
  });

  it("shows an update that failed while no page could follow it", async () => {
    const { container } = await reload(
      job({
        state: "failed",
        error: "jutul-agent could not be upgraded. See the output above.",
        lines: ["$ uv tool upgrade jutul-agent", "error: no network"],
      }),
    );
    expect(screen.getByText("The update didn't finish.")).toBeTruthy();
    expect(container.querySelector(".update-error")?.textContent).toContain("could not be upgraded");
    expect(container.querySelector(".update-log")?.textContent).toContain("error: no network");
  });
});
