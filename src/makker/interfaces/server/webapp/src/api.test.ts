import { afterEach, describe, expect, it, vi } from "vitest";

import { api } from "./api";

afterEach(() => {
  delete window.__JUTUL_BASE_PATH__;
  vi.unstubAllGlobals();
});

describe.each(["", "/restricted", "/nested/prefix"])("API under %j", (prefix) => {
  it("routes every request through the configured prefix exactly once", async () => {
    window.__JUTUL_BASE_PATH__ = prefix;
    const fetchMock = vi.fn(async () => new Response(JSON.stringify({
      providers: [], sessions: [], messages: [],
    })));
    vi.stubGlobal("fetch", fetchMock);

    await api.simulators();
    await api.models();
    await api.liveModels();
    await api.credentials();
    await api.modelWindow("provider:model");
    await api.history(10);
    await api.messages("sid");
    await api.context("sid");
    await api.setCredential("provider", "test-key");
    await api.createSession({ sim: "demo" });
    await api.resumeSession("sid", { sim: "demo" });
    await api.deleteSession("sid");
    await api.uploadFile("sid", new File(["test"], "input.txt"));

    const requests = vi.mocked(fetch).mock.calls.map(([url, init]) => [
      url, init?.method ?? "GET",
    ]);
    expect(requests).toEqual([
      ["/simulators", "GET"],
      ["/models", "GET"],
      ["/models/live", "GET"],
      ["/credentials", "GET"],
      ["/models/window?model=provider%3Amodel", "GET"],
      ["/sessions/history?limit=10", "GET"],
      ["/sessions/sid/messages", "GET"],
      ["/sessions/sid/context", "GET"],
      ["/credentials", "POST"],
      ["/sessions", "POST"],
      ["/sessions/sid/resume", "POST"],
      ["/sessions/sid", "DELETE"],
      ["/sessions/sid/upload", "POST"],
    ].map(([path, method]) => [prefix + path, method]));

    const calls = vi.mocked(fetch).mock.calls;
    expect(calls[8][1]).toMatchObject({
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ provider: "provider", value: "test-key" }),
    });
    const upload = calls[12][1]!;
    expect(upload.body).toBeInstanceOf(FormData);
    expect((upload.body as FormData).get("file")).toBeInstanceOf(File);
    expect(upload.headers).toBeUndefined();
  });

  it("builds prefixed download URLs", () => {
    window.__JUTUL_BASE_PATH__ = prefix;
    expect(api.transcriptUrl("sid", "html")).toBe(
      `${prefix}/sessions/sid/transcript?format=html`,
    );
    expect(api.memoryUrl("sid")).toBe(`${prefix}/sessions/sid/memory`);
  });
});
