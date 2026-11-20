// React access to the store and controller. `useSel` subscribes a component to a
// slice of store state; `useController` exposes the imperative commands.

import { createContext, useContext } from "react";
import { useStore } from "zustand";
import type { StoreApi } from "zustand/vanilla";

import type { Controller } from "./controller";
import type { SessionStore } from "./store";

interface AppContext {
  store: StoreApi<SessionStore>;
  controller: Controller;
}

const Ctx = createContext<AppContext | null>(null);
export const SessionProvider = Ctx.Provider;

function useApp(): AppContext {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useApp must be used within a SessionProvider");
  return ctx;
}

export function useSel<T>(selector: (s: SessionStore) => T): T {
  return useStore(useApp().store, selector);
}

export function useController(): Controller {
  return useApp().controller;
}
