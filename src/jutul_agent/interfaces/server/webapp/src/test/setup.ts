import "@testing-library/jest-dom/vitest";

// jsdom has no ResizeObserver; the canvas stage observes its own size with one.
// A stub that never fires is enough — stage-geometry math is tested directly.
class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}
if (typeof globalThis.ResizeObserver === "undefined") {
  globalThis.ResizeObserver = ResizeObserverStub as unknown as typeof ResizeObserver;
}
