// A small debug bridge on `window.jutulDebug` for the headless screenshot/live
// harnesses: it drives the app through the same store/controller a real WebSocket
// message goes through (inject scripted wire events, open views, inspect state)
// without coupling tests to React internals.

import type { Controller } from "./controller";
import type { ReplayMessage, ServerMessage } from "./protocol";
import type { SessionStore } from "./store";
import type { StoreApi } from "zustand/vanilla";

export interface JutulDebug {
  handle: (m: ServerMessage) => void;
  addUser: (text: string) => void;
  replay: (messages: ReplayMessage[]) => void;
  openView: (id: string) => void;
  closeCanvas: () => void;
  setMeta: (meta: string) => void;
  send: (text: string) => void;
  runSlash: (text: string) => void;
  state: () => SessionStore;
}

declare global {
  interface Window {
    jutulDebug?: JutulDebug;
    /** Host-app hook: return true from this to consume a `ui` action (e.g. apply
     *  it to the host's own controls) instead of surfacing it as a note. */
    onJutulUi?: (msg: Extract<ServerMessage, { type: "ui" }>) => boolean | void;
  }
}

export function installDebug(store: StoreApi<SessionStore>, controller: Controller): void {
  window.jutulDebug = {
    handle: (m) => store.getState().handle(m),
    addUser: (text) => store.getState().addUser(text),
    replay: (messages) => store.getState().replay(messages),
    openView: (id) => store.getState().openView(id),
    closeCanvas: () => store.getState().closeCanvas(),
    setMeta: (meta) => store.setState({ meta }),
    send: (text) => controller.send(text),
    runSlash: (text) => controller.runSlash(text),
    state: () => store.getState(),
  };
}
