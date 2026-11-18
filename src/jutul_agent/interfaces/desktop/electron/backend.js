// Supervises the Python backend process and speaks the frame protocol with it.
//
// The backend is a child process; its stdin/stdout carry frames (framing.js)
// and its stderr is its log. Nothing here opens a network socket. The
// supervisor's job beyond plumbing is to keep the window usable when the
// backend dies: every request and WebSocket in flight is failed at once (the
// web UI already knows how to reconnect a dropped session), and the process is
// started again with a backoff. It deliberately has no Electron imports, so it
// runs under plain Node for the tests in test/.
'use strict';

const { spawn } = require('node:child_process');
const { EventEmitter } = require('node:events');
const { encode, Decoder } = require('./framing');

const LOG_LINES = 400;
// Restart delays after consecutive crashes; past the end the backend is given
// up on until the user asks again.
const BACKOFF_MS = [500, 1000, 2000, 4000, 8000];
// A backend that stayed up this long has recovered; the next crash starts the
// backoff over.
const STABLE_MS = 60_000;
// On quit: how long the backend may take to close its sessions (and their
// Julia kernels) before it is killed.
const STOP_GRACE_MS = 8000;

class BackendError extends Error {}

class Backend extends EventEmitter {
  /**
   * @param {{command: string, args: string[], cwd?: string, env?: object}} launch
   *   States: 'stopped' | 'starting' | 'ready' | 'restarting' | 'failed'.
   *   Events: 'state' (state, detail), 'log' (line).
   */
  constructor(launch) {
    super();
    this.launch = launch;
    this.state = 'stopped';
    this.detail = '';
    this.info = null; // the backend's `ready` frame
    this.log = [];
    this.proc = null;
    this.crashes = 0;
    this.everReady = false;
    this.readySince = 0;
    this.restartTimer = null;
    this.stopping = false;
    this.nextId = 1;
    this.http = new Map(); // id -> {resolve, reject, controller, started}
    this.sockets = new Map(); // id -> {onAccept, onMessage, onClose}
    this.pings = new Map();
    this.waiters = [];
  }

  // ---- lifecycle -------------------------------------------------------------

  start() {
    if (this.proc) return;
    clearTimeout(this.restartTimer);
    this.stopping = false;
    this.setState(this.everReady ? 'restarting' : 'starting');
    const { command, args, cwd, env } = this.launch;
    let proc;
    try {
      proc = spawn(command, args, {
        cwd,
        env: { ...process.env, ...env, PYTHONUNBUFFERED: '1' },
        stdio: ['pipe', 'pipe', 'pipe'],
        windowsHide: true,
      });
    } catch (err) {
      this.fail(`could not start the backend: ${err.message}`);
      return;
    }
    this.proc = proc;
    const decoder = new Decoder();
    proc.stdout.on('data', (chunk) => {
      let frames;
      try {
        frames = decoder.push(chunk);
      } catch (err) {
        // A corrupted stream cannot be resynchronized; start over.
        this.note(`backend sent a malformed frame (${err.message}); restarting it`);
        proc.kill();
        return;
      }
      for (const frame of frames) this.onFrame(frame.header, frame.body);
    });
    let partial = '';
    proc.stderr.setEncoding('utf8');
    proc.stderr.on('data', (text) => {
      const lines = (partial + text).split(/\r?\n/);
      partial = lines.pop();
      for (const line of lines) this.note(line);
    });
    // Writing to a pipe whose reader died raises EPIPE asynchronously; the exit
    // handler below is what reacts, so the error itself is only swallowed.
    proc.stdin.on('error', () => {});
    proc.on('error', (err) => {
      if (this.proc !== proc) return;
      this.proc = null;
      this.onExit(null, null, err);
    });
    proc.on('exit', (code, signal) => {
      if (partial) this.note(partial);
      if (this.proc !== proc) return;
      this.proc = null;
      this.onExit(code, signal, null);
    });
  }

  /** Ask the backend to shut down cleanly; kill it if it takes too long. */
  stop() {
    this.stopping = true;
    clearTimeout(this.restartTimer);
    const proc = this.proc;
    if (!proc) {
      this.setState('stopped');
      return Promise.resolve();
    }
    return new Promise((resolve) => {
      const timer = setTimeout(() => proc.kill('SIGKILL'), STOP_GRACE_MS);
      proc.once('exit', () => {
        clearTimeout(timer);
        resolve();
      });
      this.send({ t: 'shutdown' });
      proc.stdin.end(); // the backend also stops when its input closes
    });
  }

  /** Stop and start again, optionally with a new launch (another folder). */
  async restart(launch) {
    await this.stop();
    if (launch) this.launch = launch;
    this.crashes = 0;
    this.everReady = false;
    this.log = [];
    this.start();
  }

  onExit(code, signal, err) {
    const how = err ? err.message : signal ? `signal ${signal}` : `exit code ${code}`;
    this.failInFlight(`the backend stopped (${how})`);
    this.info = null;
    if (this.stopping) {
      this.setState('stopped');
      return;
    }
    this.note(`backend stopped unexpectedly (${how})`);
    if (!this.everReady) {
      // It never came up: a setup problem (no simulator for the folder, a bad
      // environment), which starting again would only repeat. Its log says why.
      this.fail(`the backend could not start (${how})`);
      return;
    }
    if (Date.now() - this.readySince > STABLE_MS) this.crashes = 0;
    const delay = BACKOFF_MS[this.crashes];
    this.crashes += 1;
    if (delay === undefined) {
      this.fail('the backend keeps stopping; giving up on restarting it');
      return;
    }
    this.setState('restarting', how);
    this.restartTimer = setTimeout(() => this.start(), delay);
  }

  fail(detail) {
    this.setState('failed', detail);
    const waiters = this.waiters;
    this.waiters = [];
    for (const w of waiters) w.reject(new BackendError(detail));
  }

  /** Fail everything that was waiting on the process that just went away. */
  failInFlight(reason) {
    const error = new BackendError(reason);
    for (const req of this.http.values()) {
      if (req.started) {
        try {
          req.controller.error(error);
        } catch {
          /* the page already stopped reading */
        }
      } else {
        req.reject(error);
      }
    }
    this.http.clear();
    const sockets = [...this.sockets.values()];
    this.sockets.clear();
    for (const ws of sockets) ws.onClose(1006, reason);
    for (const ping of this.pings.values()) ping.reject(error);
    this.pings.clear();
  }

  setState(state, detail = '') {
    this.state = state;
    this.detail = detail;
    this.emit('state', state, detail);
  }

  note(line) {
    this.log.push(line);
    if (this.log.length > LOG_LINES) this.log.splice(0, this.log.length - LOG_LINES);
    this.emit('log', line);
  }

  /** Resolves once the backend is ready; rejects if it failed or `ms` passed. */
  whenReady(ms = 120_000) {
    if (this.state === 'ready') return Promise.resolve();
    if (this.state === 'failed' || this.state === 'stopped') {
      return Promise.reject(new BackendError(this.detail || 'the backend is not running'));
    }
    return new Promise((resolve, reject) => {
      const waiter = {
        resolve: () => {
          clearTimeout(timer);
          resolve();
        },
        reject: (err) => {
          clearTimeout(timer);
          reject(err);
        },
      };
      const timer = setTimeout(() => {
        this.waiters = this.waiters.filter((w) => w !== waiter);
        reject(new BackendError('timed out waiting for the backend'));
      }, ms);
      this.waiters.push(waiter);
    });
  }

  // ---- frames ----------------------------------------------------------------

  send(header, body) {
    const proc = this.proc;
    if (!proc || !proc.stdin.writable) return false;
    proc.stdin.write(encode(header, body));
    return true;
  }

  onFrame(header, body) {
    switch (header.t) {
      case 'ready': {
        this.info = header;
        this.everReady = true;
        this.readySince = Date.now();
        this.setState('ready');
        const waiters = this.waiters;
        this.waiters = [];
        for (const w of waiters) w.resolve();
        break;
      }
      case 'http.start': {
        const req = this.http.get(header.id);
        if (!req) break;
        req.started = true;
        req.resolve({ status: header.status, headers: header.headers || [], body: req.stream });
        break;
      }
      case 'http.body': {
        const req = this.http.get(header.id);
        if (!req) break;
        try {
          if (body.length) req.controller.enqueue(new Uint8Array(body));
          if (!header.more) req.controller.close();
        } catch {
          /* the page cancelled the read */
        }
        if (!header.more) this.http.delete(header.id);
        break;
      }
      case 'ws.accept':
        this.sockets.get(header.id)?.onAccept(header.subprotocol || '');
        break;
      case 'ws.msg': {
        const ws = this.sockets.get(header.id);
        if (ws) ws.onMessage(header.binary ? body : body.toString('utf8'), !!header.binary);
        break;
      }
      case 'ws.close': {
        const ws = this.sockets.get(header.id);
        this.sockets.delete(header.id);
        if (ws) ws.onClose(header.code || 1000, header.reason || '');
        break;
      }
      case 'pong': {
        const ping = this.pings.get(header.id);
        this.pings.delete(header.id);
        if (ping) ping.resolve();
        break;
      }
      default:
        break;
    }
  }

  // ---- requests --------------------------------------------------------------

  /**
   * One HTTP exchange. Resolves with {status, headers, body: ReadableStream}
   * as soon as the response starts; the body streams as it arrives.
   */
  request({ method, path, query, headers }, body, signal) {
    return new Promise((resolve, reject) => {
      if (this.state !== 'ready') {
        reject(new BackendError('the backend is not running'));
        return;
      }
      const id = this.nextId++;
      const abort = () => {
        if (this.http.delete(id)) this.send({ t: 'http.abort', id });
      };
      const req = { resolve, reject, started: false, controller: null, stream: null };
      req.stream = new ReadableStream({
        start: (controller) => {
          req.controller = controller;
        },
        cancel: abort,
      });
      this.http.set(id, req);
      signal?.addEventListener('abort', () => {
        abort();
        if (!req.started) reject(new BackendError('aborted'));
      });
      this.send({ t: 'http', id, method, path, query, headers }, body);
    });
  }

  /** Open a WebSocket to the app. Returns its id, or null when not running. */
  openSocket({ path, query, headers, subprotocols }, handlers) {
    if (this.state !== 'ready') return null;
    const id = this.nextId++;
    this.sockets.set(id, handlers);
    this.send({ t: 'ws.open', id, path, query, headers, subprotocols });
    return id;
  }

  sendSocket(id, data) {
    if (!this.sockets.has(id)) return;
    const binary = typeof data !== 'string';
    this.send({ t: 'ws.msg', id, binary }, data);
  }

  closeSocket(id, code = 1000) {
    if (this.sockets.delete(id)) this.send({ t: 'ws.close', id, code });
  }

  /** Round-trip through the backend's event loop; rejects after `ms`. */
  ping(ms = 5000) {
    return new Promise((resolve, reject) => {
      if (this.state !== 'ready') {
        reject(new BackendError('the backend is not running'));
        return;
      }
      const id = this.nextId++;
      const timer = setTimeout(() => {
        this.pings.delete(id);
        reject(new BackendError('the backend did not answer'));
      }, ms);
      this.pings.set(id, {
        resolve: () => {
          clearTimeout(timer);
          resolve();
        },
        reject: (err) => {
          clearTimeout(timer);
          reject(err);
        },
      });
      this.send({ t: 'ping', id });
    });
  }
}

module.exports = { Backend, BackendError };
