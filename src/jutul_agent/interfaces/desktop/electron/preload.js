// Preload for every frame of the desktop window (the app, and the live-plot
// pages it embeds in iframes).
//
// The web UI is the one the browser gets, unchanged: it calls `fetch` and opens
// `WebSocket`s against its own origin. In the desktop app that origin is the
// `jutul://` scheme, which the main process answers from the backend's pipe.
// `fetch` (and iframes, images, downloads) reach it through Electron's protocol
// handler with no help. A WebSocket cannot: a custom scheme has no upgrade. So
// this file replaces `WebSocket` in the page with one that carries the same
// messages over Electron IPC to the main process, for same-origin URLs only.
'use strict';

const { contextBridge, ipcRenderer, webFrame } = require('electron');

// ---- isolated world: the IPC end -------------------------------------------

const sockets = new Map(); // id -> callback(kind, a, b)
// Ids must not collide with those of an earlier document in this frame whose
// close has not reached the main process yet.
const prefix = `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}-`;
let nextSocket = 0;

ipcRenderer.on('jutul:ws', (_event, id, kind, a, b) => {
  const cb = sockets.get(id);
  if (!cb) return;
  if (kind === 'close') sockets.delete(id);
  cb(kind, a, b);
});

// A document that goes away (reload, a closed plot iframe) must not leave its
// sockets open in the backend.
window.addEventListener('pagehide', () => {
  for (const id of sockets.keys()) ipcRenderer.send('jutul:ws-close', id, 1001);
  sockets.clear();
});

const socketApi = {
  wsOpen(url, protocols, cb) {
    const id = prefix + ++nextSocket;
    sockets.set(id, cb);
    ipcRenderer.send('jutul:ws-open', id, url, protocols);
    return id;
  },
  wsSend(id, data) {
    ipcRenderer.send('jutul:ws-send', id, data);
  },
  wsClose(id, code) {
    sockets.delete(id);
    ipcRenderer.send('jutul:ws-close', id, code);
  },
};

// What the shell's own pages (status.html) and the status banner use.
const statusListeners = new Set();
ipcRenderer.on('jutul:status', (_event, status) => {
  for (const cb of statusListeners) cb(status);
});
try {
  contextBridge.exposeInMainWorld('jutulDesktop', {
    getStatus: () => ipcRenderer.invoke('jutul:get-status'),
    onStatus: (cb) => {
      statusListeners.add(cb);
    },
    retry: (options) => ipcRenderer.invoke('jutul:retry', options || {}),
    chooseWorkspace: () => ipcRenderer.invoke('jutul:choose-workspace'),
  });
} catch {
  // An iframe keeps its window object when it leaves its initial about:blank, so
  // the name is already taken by the first run of this script. Only the shell's
  // own top-level pages use this API; nothing is lost.
}

// ---- main world: the WebSocket the page sees -------------------------------

function installWebSocketShim(ipc) {
  // Installed again on every run, over an earlier copy: that copy's `ipc` belongs
  // to a preload context that is gone.
  const Native = window.WebSocket.__jutulNative || window.WebSocket;
  const CONNECTING = 0;
  const OPEN = 1;
  const CLOSING = 2;
  const CLOSED = 3;

  class DesktopWebSocket extends EventTarget {
    constructor(url, protocols) {
      const target = new URL(url, location.href);
      // Only the app's own origin lives behind the pipe. Anything else is a
      // real network address and gets the real WebSocket.
      if (
        location.protocol !== 'jutul:' ||
        target.host !== location.host ||
        !/^(wss?|https?|jutul):$/.test(target.protocol)
      ) {
        return new Native(url, protocols);
      }
      super();
      const wanted = protocols === undefined ? [] : [].concat(protocols).map(String);
      this.url = `ws://${target.host}${target.pathname}${target.search}`;
      this.readyState = CONNECTING;
      this.protocol = '';
      this.extensions = '';
      this.bufferedAmount = 0;
      this.binaryType = 'blob';
      this.onopen = this.onmessage = this.onclose = this.onerror = null;
      this._id = ipc.wsOpen(target.pathname + target.search, wanted, (kind, a, b) => {
        if (kind === 'open') {
          this.readyState = OPEN;
          this.protocol = a || '';
          this._emit(new Event('open'));
        } else if (kind === 'message') {
          if (this.readyState !== OPEN) return;
          let data = a;
          if (typeof a !== 'string') {
            // IPC delivers a Uint8Array; hand the page what a socket would.
            const buffer = a.buffer.slice(a.byteOffset, a.byteOffset + a.byteLength);
            data = this.binaryType === 'arraybuffer' ? buffer : new Blob([buffer]);
          }
          this._emit(new MessageEvent('message', { data, origin: location.origin }));
        } else if (kind === 'close') {
          this._finish(a, b);
        }
      });
    }

    send(data) {
      if (this.readyState === CONNECTING) {
        throw new DOMException('Still in CONNECTING state.', 'InvalidStateError');
      }
      if (this.readyState !== OPEN) return;
      if (typeof data === 'string') {
        ipc.wsSend(this._id, data);
      } else if (data instanceof Blob) {
        data.arrayBuffer().then((buffer) => {
          if (this.readyState === OPEN) ipc.wsSend(this._id, new Uint8Array(buffer));
        });
      } else if (ArrayBuffer.isView(data)) {
        ipc.wsSend(this._id, new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
      } else {
        ipc.wsSend(this._id, new Uint8Array(data));
      }
    }

    close(code, reason) {
      if (this.readyState === CLOSING || this.readyState === CLOSED) return;
      const wasOpen = this.readyState === OPEN;
      this.readyState = CLOSING;
      ipc.wsClose(this._id, code === undefined ? 1000 : code);
      // The main process drops the socket without echoing; finish it here, on
      // a later tick as a real socket would.
      setTimeout(() => this._finish(code === undefined ? 1005 : code, reason || '', wasOpen), 0);
    }

    _finish(code, reason, clean) {
      if (this.readyState === CLOSED) return;
      // A close before the handshake finished is a failed connection, which a
      // socket reports as an error followed by an unclean close.
      const failed = this.readyState === CONNECTING;
      this.readyState = CLOSED;
      if (failed) this._emit(new Event('error'));
      const wasClean = clean !== undefined ? clean : !failed && code !== 1006;
      this._emit(new CloseEvent('close', { code: failed ? 1006 : code, reason, wasClean }));
    }

    _emit(event) {
      this.dispatchEvent(event);
      const handler = this[`on${event.type}`];
      if (typeof handler === 'function') {
        try {
          handler.call(this, event);
        } catch (err) {
          setTimeout(() => {
            throw err;
          }, 0);
        }
      }
    }
  }
  for (const [name, value] of Object.entries({ CONNECTING, OPEN, CLOSING, CLOSED })) {
    Object.defineProperty(DesktopWebSocket, name, { value, enumerable: true });
    Object.defineProperty(DesktopWebSocket.prototype, name, { value, enumerable: true });
  }
  DesktopWebSocket.__jutulDesktop = true;
  DesktopWebSocket.__jutulNative = Native;
  window.WebSocket = DesktopWebSocket;
}

// Must be in place before the page's first script runs: a live-plot page opens
// its socket from an inline script, and the native WebSocket rejects this scheme
// outright. Two things make that hold:
// - `executeInMainWorld` runs synchronously, here and now; `webFrame.executeJavaScript`
//   only queues the script and loses the race in an iframe (it is the fallback for
//   an Electron too old to have the former).
// - It is installed whatever the document's URL is right now, and handed its IPC
//   functions as an argument rather than through a named global. An iframe starts
//   as about:blank and keeps that window object when it navigates to the app's
//   origin, while this script runs again in a new context: a shim skipped for
//   the blank document would be missing for the real one, and a global exposed
//   for the blank document cannot be exposed a second time (that throws).
if (typeof contextBridge.executeInMainWorld === 'function') {
  contextBridge.executeInMainWorld({ func: installWebSocketShim, args: [socketApi] });
} else {
  try {
    contextBridge.exposeInMainWorld('__jutulDesktopIpc', socketApi);
  } catch {
    /* already exposed by an earlier run in this window */
  }
  webFrame.executeJavaScript(`(${installWebSocketShim.toString()})(window.__jutulDesktopIpc);`);
}

// ---- the status banner (top frame only) ------------------------------------

// The web UI reconnects a dropped session by itself; the banner only tells the
// user why the conversation paused, since the UI cannot know a helper crashed.
if (window.top === window) {
  let banner = null;
  let hideTimer = 0;
  const show = (text, tone) => {
    if (!document.body) return;
    if (!banner) {
      banner = document.createElement('div');
      banner.setAttribute('role', 'status');
      banner.style.cssText =
        'position:fixed;left:50%;bottom:18px;transform:translateX(-50%);z-index:2147483647;' +
        'padding:9px 16px;border-radius:8px;font:13px/1.4 system-ui,sans-serif;color:#fff;' +
        'box-shadow:0 4px 18px rgba(0,0,0,.3);max-width:80vw;pointer-events:none;';
      document.body.appendChild(banner);
    }
    banner.textContent = text;
    banner.style.background = tone === 'ok' ? '#1f7a4d' : '#9a3412';
    banner.style.display = 'block';
    clearTimeout(hideTimer);
  };
  const hide = (after) => {
    clearTimeout(hideTimer);
    hideTimer = setTimeout(() => {
      if (banner) banner.style.display = 'none';
    }, after);
  };
  let wasDown = false;
  statusListeners.add((status) => {
    if (location.pathname.startsWith('/__desktop__/')) return; // that page shows it itself
    if (status.state === 'restarting') {
      wasDown = true;
      show('The backend stopped unexpectedly. Restarting it…', 'warn');
    } else if (status.state === 'ready' && wasDown) {
      wasDown = false;
      show('The backend is running again. Reconnecting your session…', 'ok');
      hide(5000);
    }
  });
}
