import { render } from "@testing-library/react";
import type { ReactElement, ReactNode } from "react";

import { SessionProvider } from "../context";
import { Controller } from "../controller";
import { createSessionStore } from "../store";

/** Render UI wrapped in a fresh store + controller (no network, no socket).
 *  The provider rides in `wrapper`, so testing-library's `rerender` keeps it. */
export function renderWithStore(ui: ReactElement) {
  const store = createSessionStore();
  const controller = new Controller(store);
  const wrapper = ({ children }: { children: ReactNode }) => (
    <SessionProvider value={{ store, controller }}>{children}</SessionProvider>
  );
  const utils = render(ui, { wrapper });
  return { store, controller, ...utils };
}
