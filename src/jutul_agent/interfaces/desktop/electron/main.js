// The jutul-agent desktop app: the web UI in a window, with no server anywhere.
//
// Process layout (every arrow is a pipe or Electron's own IPC, never a socket):
//
//   renderer (web UI, plot iframes)
//      │  fetch / iframes / downloads → the `jutul://` protocol handler below
//      │  WebSocket                   → preload.js shim → ipcMain below
//   main (this file)
//      │  frames over the child's stdin/stdout (backend.js, framing.js)
//   Python backend (jutul_agent.interfaces.desktop.backend)
//      │  anonymous pipes inherited at spawn
//   Julia kernel(s)
//
// The web UI is the unmodified build in ../../server/web_dist, served by the
// backend's own routes. This file adds what a browser tab gets from a server
// for free: an origin, and a way to reach the application behind it.
'use strict';

const { app, BrowserWindow, Menu, dialog, ipcMain, protocol, session, shell } = require('electron');
const fs = require('node:fs');
const path = require('node:path');
const { Backend } = require('./backend');

const SCHEME = 'jutul';
const HOST = 'localhost';
const ORIGIN = `${SCHEME}://${HOST}`;
const SHELL_PREFIX = '/__desktop__/';
// How long a request waits for a backend that is starting or restarting. The
// web UI's reconnect rides on this: its calls simply take longer.
const READY_WAIT_MS = 120_000;
const SIMULATORS = ['jutuldarcy', 'battmo', 'fimbul', 'mocca'];

// Must run before the app is ready. `standard` gives the scheme real origins
// (so storage and same-origin checks work), `secure` makes it a secure context,
// and `stream` lets responses stream.
protocol.registerSchemesAsPrivileged([
  {
    scheme: SCHEME,
    privileges: { standard: true, secure: true, supportFetchAPI: true, stream: true },
  },
]);

// ---- configuration -----------------------------------------------------------

function argValue(name) {
  const prefix = `--${name}=`;
  const hit = process.argv.find((a) => a.startsWith(prefix));
  return hit ? hit.slice(prefix.length) : null;
}

function settingsPath() {
  return path.join(app.getPath('userData'), 'desktop-settings.json');
}

function loadSettings() {
  try {
    return JSON.parse(fs.readFileSync(settingsPath(), 'utf8'));
  } catch {
    return {};
  }
}

function saveSettings(settings) {
  try {
    fs.mkdirSync(path.dirname(settingsPath()), { recursive: true });
    fs.writeFileSync(settingsPath(), JSON.stringify(settings, null, 2));
  } catch (err) {
    console.error('could not save settings:', err.message);
  }
}

let settings = {};
let workspace = null;
let simulator = null; // an explicit --sim for a folder that has none saved

function extraBackendArgs() {
  try {
    const parsed = JSON.parse(process.env.JUTUL_AGENT_BACKEND_ARGS || '[]');
    return Array.isArray(parsed) ? parsed.map(String) : [];
  } catch {
    return [];
  }
}

function launchSpec() {
  const python = process.env.JUTUL_AGENT_PYTHON || 'python3';
  const args = ['-m', 'jutul_agent.interfaces.desktop.backend', '--workspace', workspace];
  if (simulator) args.push('--sim', simulator);
  args.push(...extraBackendArgs());
  return { command: python, args, cwd: workspace };
}

// ---- backend -------------------------------------------------------------------

let backend = null;
let quitting = false;

function status() {
  return {
    state: backend ? backend.state : 'stopped',
    detail: backend ? backend.detail : '',
    log: backend ? backend.log.slice(-120) : [],
    workspace,
    simulators: SIMULATORS,
    version: backend?.info?.version || null,
  };
}

function broadcastStatus() {
  const current = status();
  for (const win of BrowserWindow.getAllWindows()) {
    if (!win.isDestroyed()) win.webContents.send('jutul:status', current);
  }
}

function onShellPage(win) {
  try {
    return new URL(win.webContents.getURL()).pathname.startsWith(SHELL_PREFIX);
  } catch {
    return true; // nothing loaded yet
  }
}

function createBackend() {
  backend = new Backend(launchSpec());
  backend.on('log', (line) => {
    process.stderr.write(`[backend] ${line}\n`);
    broadcastStatus();
  });
  backend.on('state', (state) => {
    broadcastStatus();
    const win = mainWindow;
    if (!win || win.isDestroyed() || quitting) return;
    if (state === 'ready' && onShellPage(win)) {
      win.loadURL(`${ORIGIN}/`);
    } else if (state === 'failed' && !onShellPage(win)) {
      // Past recovery. The web UI can do nothing without a backend, so show
      // why and offer the ways out.
      win.loadURL(`${ORIGIN}${SHELL_PREFIX}status`);
    }
    // 'restarting' leaves the web UI where it is: its requests wait for the new
    // backend, its session reconnects, and the preload shows a banner meanwhile.
  });
  backend.start();
}

// ---- the jutul:// protocol: HTTP without a server ----------------------------

const SHELL_FILES = { status: ['status.html', 'text/html; charset=utf-8'] };

function shellPage(name) {
  const entry = SHELL_FILES[name];
  if (!entry) return new Response('not found', { status: 404 });
  return new Response(fs.readFileSync(path.join(__dirname, entry[0])), {
    headers: { 'content-type': entry[1], 'cache-control': 'no-store' },
  });
}

function unavailable(request, reason) {
  // A page load gets the status page (which explains and offers a retry); an
  // API call gets an error it can handle.
  if (request.destination === 'document' && request.method === 'GET') {
    return Response.redirect(`${ORIGIN}${SHELL_PREFIX}status`, 302);
  }
  return new Response(JSON.stringify({ detail: reason }), {
    status: 503,
    headers: { 'content-type': 'application/json' },
  });
}

// Statuses whose responses may not carry a body.
const NULL_BODY = new Set([101, 204, 205, 304]);

async function handleRequest(request) {
  const url = new URL(request.url);
  if (url.host !== HOST) return new Response('not found', { status: 404 });
  if (url.pathname.startsWith(SHELL_PREFIX)) {
    return shellPage(url.pathname.slice(SHELL_PREFIX.length));
  }
  try {
    await backend.whenReady(READY_WAIT_MS);
  } catch (err) {
    return unavailable(request, err.message);
  }
  const headers = [];
  let hasOrigin = false;
  for (const [name, value] of request.headers) {
    if (name === 'host') continue;
    if (name === 'origin') hasOrigin = true;
    headers.push([name, value]);
  }
  // The app refuses state-changing requests from other origins. Chromium does
  // not always attach Origin for a custom scheme; the page that could reach
  // this handler is ours (see the navigation guards below), so say so.
  if (!hasOrigin && request.method !== 'GET' && request.method !== 'HEAD') {
    headers.push(['origin', ORIGIN]);
  }
  const hasBody = request.method !== 'GET' && request.method !== 'HEAD';
  const body = hasBody ? Buffer.from(await request.arrayBuffer()) : null;
  let answer;
  try {
    answer = await backend.request(
      { method: request.method, path: url.pathname, query: url.search.slice(1), headers },
      body,
      request.signal,
    );
  } catch (err) {
    return unavailable(request, err.message);
  }
  const responseHeaders = new Headers();
  for (const [name, value] of answer.headers) {
    try {
      responseHeaders.append(name, value);
    } catch {
      /* a header the Fetch API refuses to carry; the page never needed it */
    }
  }
  const status = answer.status >= 200 && answer.status <= 599 ? answer.status : 502;
  const noBody = NULL_BODY.has(status) || request.method === 'HEAD';
  if (noBody) answer.body.cancel().catch(() => {});
  return new Response(noBody ? null : answer.body, { status, headers: responseHeaders });
}

// ---- WebSockets over IPC -------------------------------------------------------

// key -> {frame, id, backendId}. The key is unique per renderer frame + its id.
const sockets = new Map();

// Whether a URL is one of the app's own pages. Compared by parts: Node reports
// the origin of a non-http scheme as "null", unlike the renderer.
function isOwnUrl(target) {
  try {
    const url = new URL(target);
    return url.protocol === `${SCHEME}:` && url.host === HOST;
  } catch {
    return false;
  }
}

function frameKey(frame, id) {
  return `${frame.processId}:${frame.routingId}:${id}`;
}

function ownPage(frame) {
  return !!frame && isOwnUrl(frame.url);
}

function toFrame(entry, kind, a, b) {
  try {
    if (entry.frame.isDestroyed?.()) throw new Error('frame destroyed');
    entry.frame.send('jutul:ws', entry.id, kind, a, b);
    return true;
  } catch {
    return false;
  }
}

function dropSocket(key, code) {
  const entry = sockets.get(key);
  if (!entry) return;
  sockets.delete(key);
  if (entry.backendId !== null) backend.closeSocket(entry.backendId, code);
}

function registerSocketIpc() {
  ipcMain.on('jutul:ws-open', async (event, id, target, protocols) => {
    const frame = event.senderFrame;
    // Only the app's own pages may reach the backend; this is the check the
    // server's origin guard makes for a browser.
    if (!ownPage(frame) || typeof id !== 'string' || typeof target !== 'string') return;
    const key = frameKey(frame, id);
    const entry = { frame, id, backendId: null };
    sockets.set(key, entry);
    try {
      await backend.whenReady(READY_WAIT_MS);
    } catch {
      /* answered as a refused connection below */
    }
    if (sockets.get(key) !== entry) return; // closed while waiting
    const url = new URL(target, `${ORIGIN}/`);
    const subprotocols = Array.isArray(protocols) ? protocols.map(String) : [];
    const headers = [['origin', ORIGIN]];
    if (subprotocols.length) headers.push(['sec-websocket-protocol', subprotocols.join(', ')]);
    entry.backendId = backend.openSocket(
      { path: url.pathname, query: url.search.slice(1), headers, subprotocols },
      {
        onAccept: (subprotocol) => {
          if (!toFrame(entry, 'open', subprotocol)) dropSocket(key, 1001);
        },
        onMessage: (data) => {
          if (!toFrame(entry, 'message', data)) dropSocket(key, 1001);
        },
        onClose: (code, reason) => {
          sockets.delete(key);
          toFrame(entry, 'close', code, reason);
        },
      },
    );
    if (entry.backendId === null) {
      sockets.delete(key);
      toFrame(entry, 'close', 1006, 'the backend is not running');
    }
  });

  ipcMain.on('jutul:ws-send', (event, id, data) => {
    const entry = sockets.get(frameKey(event.senderFrame, id));
    if (!entry || entry.backendId === null) return;
    backend.sendSocket(entry.backendId, typeof data === 'string' ? data : Buffer.from(data));
  });

  ipcMain.on('jutul:ws-close', (event, id, code) => {
    dropSocket(frameKey(event.senderFrame, id), typeof code === 'number' ? code : 1000);
  });

  // A frame that vanished without saying goodbye (a crashed renderer) must
  // not leave its sockets open in the backend: each one holds a stream or a
  // live plot view there.
  setInterval(() => {
    for (const [key, entry] of sockets) {
      let gone = false;
      try {
        gone = entry.frame.isDestroyed?.() || entry.frame.detached === true;
      } catch {
        gone = true;
      }
      if (gone) dropSocket(key, 1001);
    }
  }, 5000).unref();
}

// ---- shell commands (status page, menu) --------------------------------------

async function chooseWorkspace() {
  const result = await dialog.showOpenDialog(mainWindow, {
    title: 'Choose a workspace folder',
    properties: ['openDirectory', 'createDirectory'],
    defaultPath: workspace || app.getPath('home'),
  });
  if (result.canceled || !result.filePaths.length) return false;
  workspace = result.filePaths[0];
  simulator = null;
  settings.workspace = workspace;
  saveSettings(settings);
  await restartBackend();
  return true;
}

async function restartBackend(options = {}) {
  if (options.simulator && SIMULATORS.includes(options.simulator)) simulator = options.simulator;
  if (mainWindow && !mainWindow.isDestroyed() && !onShellPage(mainWindow)) {
    mainWindow.loadURL(`${ORIGIN}${SHELL_PREFIX}status`);
  }
  await backend.restart(launchSpec());
}

function registerShellIpc() {
  ipcMain.handle('jutul:get-status', () => status());
  ipcMain.handle('jutul:retry', (event, options) => {
    if (ownPage(event.senderFrame)) return restartBackend(options || {});
    return undefined;
  });
  ipcMain.handle('jutul:choose-workspace', (event) => {
    if (ownPage(event.senderFrame)) return chooseWorkspace();
    return false;
  });
}

// ---- windows ---------------------------------------------------------------------

let mainWindow = null;
let rendererCrashes = [];

const webPreferences = {
  preload: path.join(__dirname, 'preload.js'),
  contextIsolation: true,
  nodeIntegration: false,
  sandbox: true,
  // The preload must also run in the live-plot iframes: they open the
  // WebSockets that carry a figure's widget traffic.
  nodeIntegrationInSubFrames: true,
};


function guard(contents) {
  // The window shows the app and nothing else. A link elsewhere opens in the
  // user's browser; it never replaces the app or gains its access.
  contents.on('will-navigate', (event, target) => {
    if (isOwnUrl(target)) return;
    event.preventDefault();
    if (/^https?:/i.test(target)) shell.openExternal(target);
  });
  contents.setWindowOpenHandler(({ url }) => {
    if (isOwnUrl(url)) {
      // A plot popped out into its own window.
      return {
        action: 'allow',
        overrideBrowserWindowOptions: { webPreferences, autoHideMenuBar: true },
      };
    }
    if (/^https?:/i.test(url)) shell.openExternal(url);
    return { action: 'deny' };
  });
  contents.on('did-create-window', (child) => guard(child.webContents));
}

function createWindow() {
  const win = new BrowserWindow({
    width: 1440,
    height: 920,
    minWidth: 720,
    minHeight: 480,
    title: 'jutul-agent',
    backgroundColor: '#14161a',
    webPreferences,
  });
  mainWindow = win;
  guard(win.webContents);

  // A crashed renderer takes only the page with it: the backend still holds
  // the session and its Julia state, so reloading puts the user back in it.
  win.webContents.on('render-process-gone', (_event, details) => {
    if (details.reason === 'clean-exit' || quitting) return;
    console.error(`renderer gone (${details.reason}); reloading`);
    const now = Date.now();
    rendererCrashes = rendererCrashes.filter((t) => now - t < 60_000);
    rendererCrashes.push(now);
    if (rendererCrashes.length > 3) {
      dialog.showErrorBox(
        'jutul-agent',
        'The window keeps crashing. Use View → Reload to try again; your session is kept.',
      );
      return;
    }
    setTimeout(() => {
      if (!win.isDestroyed()) win.reload();
    }, 300);
  });
  win.on('unresponsive', async () => {
    const choice = await dialog.showMessageBox(win, {
      type: 'warning',
      message: 'The window is not responding.',
      detail: 'Reloading keeps your session and its Julia state.',
      buttons: ['Wait', 'Reload'],
      defaultId: 0,
    });
    if (choice.response === 1 && !win.isDestroyed()) win.webContents.forcefullyCrashRenderer();
  });
  win.on('closed', () => {
    if (mainWindow === win) mainWindow = null;
  });

  win.loadURL(backend.state === 'ready' ? `${ORIGIN}/` : `${ORIGIN}${SHELL_PREFIX}status`);
  return win;
}

function buildMenu() {
  const isMac = process.platform === 'darwin';
  const template = [
    ...(isMac ? [{ role: 'appMenu' }] : []),
    {
      label: 'File',
      submenu: [
        { label: 'Open Workspace…', accelerator: 'CmdOrCtrl+O', click: () => chooseWorkspace() },
        { label: 'Restart Backend', click: () => restartBackend() },
        { type: 'separator' },
        isMac ? { role: 'close' } : { role: 'quit' },
      ],
    },
    { role: 'editMenu' },
    {
      label: 'View',
      submenu: [
        { role: 'reload' },
        { role: 'toggleDevTools' },
        { type: 'separator' },
        { role: 'resetZoom' },
        { role: 'zoomIn' },
        { role: 'zoomOut' },
        { type: 'separator' },
        { role: 'togglefullscreen' },
      ],
    },
    { role: 'windowMenu' },
  ];
  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}

// ---- app lifecycle -----------------------------------------------------------------

// No single-instance lock: Electron implements it with a listening Unix socket
// (Chromium's "SingletonSocket"), and this app may not listen on anything. A
// second launch is simply a second, independent app.
{
  app.whenReady().then(async () => {
    settings = loadSettings();
    workspace = argValue('workspace') || process.env.JUTUL_AGENT_WORKSPACE || settings.workspace;
    if (!workspace || !fs.existsSync(workspace)) {
      const result = await dialog.showOpenDialog({
        title: 'Choose a workspace folder for jutul-agent',
        properties: ['openDirectory', 'createDirectory'],
      });
      if (result.canceled || !result.filePaths.length) {
        app.quit();
        return;
      }
      workspace = result.filePaths[0];
    }
    workspace = path.resolve(workspace);
    settings.workspace = workspace;
    saveSettings(settings);

    session.defaultSession.protocol.handle(SCHEME, handleRequest);
    // The app asks for nothing a web page needs permission for.
    session.defaultSession.setPermissionRequestHandler((_wc, _permission, callback) =>
      callback(false),
    );
    registerSocketIpc();
    registerShellIpc();
    buildMenu();
    createBackend();
    createWindow();

    app.on('activate', () => {
      if (BrowserWindow.getAllWindows().length === 0) createWindow();
    });
  });

  // A helper process of Chromium's own (GPU, network service) is restarted by
  // Electron; note it so a pattern shows up in the log.
  app.on('child-process-gone', (_event, details) => {
    console.error(`helper process gone: ${details.type} (${details.reason})`);
  });

  app.on('window-all-closed', () => {
    app.quit();
  });

  // Quit only once the backend has closed its sessions: each holds a Julia
  // process that should exit cleanly rather than be orphaned.
  app.on('before-quit', (event) => {
    if (quitting || !backend) return;
    quitting = true;
    event.preventDefault();
    backend.stop().finally(() => app.quit());
  });
}
