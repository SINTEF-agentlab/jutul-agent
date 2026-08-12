// The application controller: the imperative glue between the store, the REST API,
// and the WebSocket transport. Components call these high-level commands; the
// controller decides what to send, what to fetch, and how to update the store.
// Kept out of React so the flow is one readable unit (and the store stays pure).

import type { StoreApi } from "zustand/vanilla";

import { ApiError, api } from "./api";
import { stageSize } from "./canvas/stageSize";
import { hostApiUrl, hostContext } from "./hostContext";
import { HISTORY_CHANGED, type ClientMessage, type ServerMessage } from "./protocol";
import type { CredentialPrompt, SessionStore } from "./store";
import { Transport } from "./transport";

export interface SlashCommand {
  name: string;
  hint?: string;
  desc: string;
}

export const SLASH_COMMANDS: SlashCommand[] = [
  { name: "/help", desc: "show available commands" },
  { name: "/clear", desc: "clear the visible conversation" },
  { name: "/new", desc: "start a new chat" },
  { name: "/copy", desc: "copy the last assistant message" },
  { name: "/context", desc: "show how much context the last turn used" },
  { name: "/model", hint: "[provider:model]", desc: "switch the model (keeps the session)" },
  { name: "/approval-mode", hint: "[ask|workspace|auto]", desc: "set the approval policy" },
  { name: "/transcript", hint: "[md]", desc: "download the conversation to share" },
  { name: "/memory", desc: "view the workspace memory" },
  { name: "/compact", desc: "summarize older turns to free context" },
  { name: "/add-dir", hint: "<path>", desc: "give the agent another folder" },
];

const APPROVAL_MODES = ["ask", "workspace", "auto"];

const REATTACHED_NOTE = "Reconnected to this session. Its Julia REPL and live state are intact.";
const RESTARTED_NOTE =
  "Resumed this session. The chat is restored, but the Julia REPL restarted. " +
  "Earlier files and artifacts are intact; re-run setup to rebuild in-memory state.";

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

/** The credential prompt carried by a `credential_required` API error, or null. */
function credentialRequiredOf(e: unknown): CredentialPrompt | null {
  if (!(e instanceof ApiError) || !e.detail || typeof e.detail !== "object") return null;
  const d = e.detail as Record<string, unknown>;
  if (d.error !== "credential_required") return null;
  return {
    provider: String(d.provider ?? ""),
    label: String(d.label ?? ""),
    env_var: String(d.env_var ?? ""),
  };
}

export class Controller {
  readonly transport: Transport;
  private notifyAsked = false;
  private promptHistory: string[] = [];
  private historyPos = -1;
  private historyDraft = "";
  // The composer registers a sink so an uploaded file's path can be appended to the
  // message box, whether the upload came from the attach button or a drag-drop.
  private composerSink: ((text: string) => void) | null = null;
  // A prompt the user sent while no socket was open (before the first session is
  // ready, or during a reconnect); delivered once a socket opens.
  private queuedPrompt: string | null = null;
  // True while a reconnect is in flight, so a second trigger does not start another.
  private recovering = false;
  // Bumped per reenter so a superseding one (e.g. a sidebar switch during a reconnect)
  // makes the older flow bail instead of the two clobbering each other's socket.
  private reenterSeq = 0;
  // What to do once a missing API key is saved: re-run the action the key blocked
  // (start the session, or apply the model switch). Cleared when the modal closes.
  private keyRetry: (() => void) | null = null;
  // Popup windows waiting for their `popout_ready` answer, per plot record (a
  // popup must be opened synchronously in the click to beat popup blockers, so
  // it exists before the server has replayed the figure).
  private pendingPopouts = new Map<string, Window[]>();

  constructor(private store: StoreApi<SessionStore>) {
    this.transport = new Transport(
      store,
      (msg) => this.effects(msg),
      () => void this.reconnect(),
      () => {
        this.queuedPrompt = null;
      },
    );
    // Keep the server's canvas-size hint fresh across window resizes, so a plot
    // made mid-turn is shaped for the panel as it is now, not as it was when
    // the turn started. Debounced: only the settled size matters.
    let resizeTimer: ReturnType<typeof setTimeout> | undefined;
    window.addEventListener("resize", () => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => this.sendCanvasHint(), 400);
    });
  }

  private get s() {
    return this.store.getState();
  }

  private meta(): string {
    const { model, sessionId } = this.s;
    return `${model || "no model"} · ${(sessionId || "").slice(0, 13)}`;
  }

  // --- startup --------------------------------------------------------------

  async init(): Promise<void> {
    const [sims, models, credentials] = await Promise.all([
      api.simulators(),
      api.models(),
      api.credentials(),
    ]);
    const names = sims.simulators || [];
    this.store.setState({
      sim: sims.default || names[0] || "jutuldarcy",
      simDetails: sims.details || {},
      model: models.default ?? null,
      models: models.models || [],
      credentials,
    });
    this.refreshHistory();
    this.refreshContextWindow();
    this.watchHostContext();
    await this.startSession();
  }

  /** Forward later changes to the host application's selection to the session.
   *
   *  The launch value rides in on create and resume; this covers the host
   *  changing it while a conversation is open. A send that finds no socket is
   *  dropped on purpose: every path that opens one (create, resume, reconnect)
   *  carries the current selection in its request, so the session cannot be left
   *  behind by a change it missed.
   *
   *  Never torn down, because it lives as long as the page does: `init` runs
   *  once, and a reload replaces the whole controller along with the listener. */
  private watchHostContext(): void {
    hostContext.listen();
    hostContext.subscribe((context) => {
      this.transport.send({ type: "host_context", context });
    });
  }

  async refreshCredentials(): Promise<void> {
    this.s.setCredentials(await api.credentials());
  }

  async refreshContextWindow(): Promise<void> {
    const { model } = this.s;
    const data = await api.modelWindow(model || "");
    this.s.setContextWindow(data.window ?? null);
  }

  async startSession(): Promise<void> {
    // Supersede any in-flight reenter: starting fresh (init or /new) must win over a
    // reconnect that is still trying to get back to the old session.
    this.reenterSeq++;
    const { sim } = this.s;
    this.store.setState({
      meta: `starting ${sim}… (first run builds its environment, this can take a few minutes)`,
    });
    try {
      const { session_id } = await api.createSession({
        sim: sim || undefined,
        host_context: hostContext.current() ?? undefined,
        host_api: hostApiUrl ?? undefined,
      });
      this.s.setSession(session_id, "");
      this.store.setState({ meta: this.meta() });
      this.transport.open(session_id);
      this.flushQueuedPrompt();
    } catch (e) {
      const required = credentialRequiredOf(e);
      if (required) {
        // The model has no API key yet: prompt for it, then start the session. Any
        // queued prompt is kept so it sends once the key lands and the socket opens.
        this.store.setState({ meta: "waiting for an API key", busy: false, working: false });
        this.keyRetry = () => void this.startSession();
        this.s.openApiKeys(required);
        return;
      }
      this.store.setState({ meta: "could not start a session" });
      if (this.queuedPrompt) {
        this.queuedPrompt = null;
        this.store.setState({ busy: false, working: false });
        this.s.addSysNote("Could not start a session.", "warn");
      }
    }
  }

  // Deliver a prompt held while no socket was open (see `queuedPrompt`). The socket
  // buffers it if it is still connecting. The canvas hint rides ahead of it: a
  // fresh chat's first prompt takes this path, before any socket existed to
  // carry the hint `deliver` normally sends — without this, the first plot of a
  // new session is the one plot not shaped for the panel.
  private flushQueuedPrompt(): void {
    if (!this.queuedPrompt) return;
    this.sendCanvasHint();
    this.transport.send({ type: "prompt", text: this.queuedPrompt });
    this.queuedPrompt = null;
  }

  private effects(msg: ServerMessage): void {
    if (msg.type === "turn_end") {
      this.refreshHistory();
      if (!msg.cancelled) this.notifyDone("The agent finished your turn.");
    } else if (msg.type === "ui" && msg.action === HISTORY_CHANGED) {
      this.refreshHistory();
    } else if (msg.type === "popout_ready") {
      this.resolvePopout(msg.record, msg.url, msg.error ?? null);
    } else if (msg.type === "refit_done") {
      // Stage what the figure actually became — the squash floor may have held
      // the width, so the echo is the truth, never the size we asked for.
      const view = Object.values(this.s.views).find((v) => v.record === msg.record);
      if (view) this.s.setViewSize(view.id, msg.width, msg.height);
    } else if (msg.type === "credential_required") {
      // The server refused a model switch for want of a key (the UI usually catches
      // this before sending; this covers any path that slips through). Prompt for it.
      this.s.openApiKeys({ provider: msg.provider, label: msg.label, env_var: msg.env_var });
    }
  }

  // --- canvas: popout + regenerate -------------------------------------------

  /** Pop a view into its own window. A live plot with recorded code gets an
   *  independent replayed figure (a live figure supports exactly one frame, so
   *  the popup must never load the inline view's own route); anything else
   *  opens its URL directly. */
  popout(viewId: string): void {
    const view = this.s.views[viewId];
    if (!view) return;
    if (!(view.kind === "plot" && view.live && view.record) || !this.transport.isOpen()) {
      window.open(view.url, "_blank", "noopener");
      return;
    }
    // Size the window for the figure's own shape, capped to the screen.
    const w = Math.min(view.width || 1200, (window.screen?.availWidth ?? 1600) - 60);
    const h = Math.min((view.height || 800) + 30, (window.screen?.availHeight ?? 1000) - 60);
    const popup = window.open("about:blank", "_blank", `width=${w},height=${h}`);
    if (!popup) {
      this.s.addSysNote("The popout window was blocked; allow popups for this site.", "warn");
      return;
    }
    popup.document.write(
      `<title>${view.title}</title><body style="margin:0;display:grid;place-items:center;` +
        `height:100vh;font:14px system-ui;color:#556">Preparing an independent view…</body>`,
    );
    const queue = this.pendingPopouts.get(view.record) ?? [];
    queue.push(popup);
    this.pendingPopouts.set(view.record, queue);
    // The replay re-fits the figure to the window the browser actually gave us
    // (the requested size is a wish; the measured viewport is the fact).
    const inner = popup.innerWidth >= 100 && popup.innerHeight >= 100;
    this.transport.send({
      type: "replot",
      record: view.record,
      target: "popout",
      width: inner ? popup.innerWidth : w,
      height: inner ? popup.innerHeight : h,
    });
  }

  /** The server's answer to a popout replay: navigate the waiting popup to its
   *  independent live view (and release the figure when the window closes), or
   *  say why there is none. */
  private resolvePopout(record: string, url: string | null, error: string | null): void {
    const queue = this.pendingPopouts.get(record);
    const popup = queue?.shift();
    if (queue && queue.length === 0) this.pendingPopouts.delete(record);
    if (!popup || popup.closed) return;
    if (!url) {
      popup.document.body.textContent = error || "The plot could not be replayed.";
      return;
    }
    popup.location.href = url;
    // The popup carries no script of ours (it hosts the served figure page), so
    // the opener polls for its close and tells the server to release the route.
    const timer = setInterval(() => {
      if (!popup.closed) return;
      clearInterval(timer);
      this.transport.send({ type: "popout_closed", url });
    }, 1000);
  }

  /** Re-run a dead view's recorded code so it comes back live, in place — and
   *  re-fitted: a fresh panel-size hint rides ahead on the same socket, and the
   *  server fits the replay to it exactly as it fits a new plot (full-size text
   *  at the panel's size, width floored against squashing a wide layout). */
  regenerate(viewId: string): void {
    const view = this.s.views[viewId];
    if (!view?.record) return;
    if (this.s.busy) {
      this.s.addSysNote("Finish the current turn before regenerating a plot.", "warn");
      return;
    }
    if (!this.transport.isOpen()) {
      this.s.addSysNote("Not connected; reload the page and try again.", "warn");
      return;
    }
    this.sendCanvasHint();
    this.transport.send({ type: "replot", record: view.record });
    this.s.beginWorking(); // the refreshed viz + turn_end (or an error) clears it
  }

  // The last stage size each view asked a refit for, so a request whose answer
  // could not match it exactly (the squash floor held the width) is not re-sent
  // forever for the same stage. A new stage size makes a new key.
  private sentRefits = new Map<string, string>();

  /** A live view's stage and figure disagree on size — the panel grew, shrank,
   *  or changed shape: have the kernel re-fit the figure, in place. The
   *  reliable direction — a kernel-side resize pushes the new layout through
   *  Bonito to the frame, while the served page noticing its own frame change
   *  is not dependable. The answer (`refit_done`) carries the real size to
   *  stage. Sent even mid-turn: the server queues the resize behind the
   *  kernel's current eval and it lands the moment the kernel frees. */
  refitView(viewId: string, width: number, height: number): void {
    const view = this.s.views[viewId];
    if (!view?.record || !view.live) return;
    const key = `${width}x${height}`;
    if (this.sentRefits.get(viewId) === key) return;
    if (!this.transport.send({ type: "refit", record: view.record, width, height })) return;
    this.sentRefits.set(viewId, key);
  }

  /** Tell the server how big the plot stage is (or will be), so a new figure
   *  is shaped for the panel it lands in. Sent with every prompt and after a
   *  window resize settles — cheap, and only ever a hint. */
  sendCanvasHint(): void {
    const { width, height } = stageSize();
    if (width < 100 || height < 100) return;
    // `send` buffers while the socket is still connecting and answers false
    // with no socket at all — either way a missed hint is only a hint.
    this.transport.send({
      type: "ui_event",
      payload: { kind: "canvas_size", width, height },
    });
  }

  // --- composing / sending --------------------------------------------------

  send(text: string): void {
    const trimmed = text.trim();
    if (!trimmed) return;
    if (trimmed.startsWith("/")) {
      this.runSlash(trimmed);
      return;
    }
    if (this.s.busy) return;
    if (this.s.pending) {
      this.s.addUser(trimmed);
      this.s.pinBottom();
      this.sendDecision("respond", trimmed);
      return;
    }
    this.s.startTurn(trimmed);
    this.s.pinBottom();
    this.pushPrompt(trimmed);
    this.requestNotifyPermission();
    this.deliver({ type: "prompt", text: trimmed });
  }

  retry(text: string): void {
    this.s.startTurn(text);
    this.s.pinBottom();
    this.deliver({ type: "prompt", text });
  }

  // Send a prompt, or hold it and reconnect if the socket is gone. `send` buffers
  // while connecting; a false means no socket, so keep the text and re-establish the
  // session (a fresh start flushes it, a reconnect replays then flushes it).
  private deliver(msg: { type: "prompt"; text: string }): void {
    // The hint rides ahead of the prompt on the same ordered socket, so any
    // plot this turn makes is shaped for the panel as it is right now.
    this.sendCanvasHint();
    if (this.transport.send(msg)) {
      // Buffered into a still-connecting socket: keep a durable copy too, since a
      // reconnect would clear the transport's buffer. The socket's onOpen drops this
      // copy once the buffer is actually flushed, so it is never sent twice.
      if (!this.transport.isOpen()) this.queuedPrompt = msg.text;
      return;
    }
    this.queuedPrompt = msg.text;
    if (this.s.sessionId) void this.reconnect();
  }

  stop(): void {
    this.transport.send({ type: "cancel" });
  }

  sendDecision(decision: string, message?: string): void {
    if (!this.s.pending) return;
    this.transport.send({ type: "decision", decision, ...(message ? { message } : {}) });
    this.s.clearInterrupt();
    this.s.beginWorking();
  }

  // --- slash commands -------------------------------------------------------

  runSlash(text: string): void {
    const name = text.split(/\s+/)[0];
    const arg = text.slice(name.length).trim();
    switch (name) {
      case "/help":
        return this.s.addHelp();
      case "/clear":
        return this.store.setState({ items: [] });
      case "/new":
        void this.newChat();
        return;
      case "/copy":
        return this.copyLast();
      case "/context":
        void this.showContext();
        return;
      case "/model":
        return this.setModel(arg); // empty arg opens the picker via the composer
      case "/approval-mode":
        return this.setApproval(arg);
      case "/transcript":
        return this.transcript(arg === "md" || arg === "markdown" ? "md" : "html");
      case "/memory":
        return this.memory();
      case "/compact":
        return this.compact();
      case "/add-dir":
        return this.addDir(arg);
      default:
        return this.s.addSysNote(`Unknown command ${name}. Type /help for the list.`, "warn");
    }
  }

  setModel(id: string): void {
    if (!id) return; // the composer opens the picker for a bare /model
    if (this.s.busy) {
      this.s.addSysNote("Finish the current turn before changing the model.", "warn");
      return;
    }
    // Prompt for a missing key before switching, so the switch doesn't bounce off
    // the server's guard. After the key is saved, retry this same switch.
    const required = this.credentialMissingFor(id);
    if (required) {
      this.keyRetry = () => this.setModel(id);
      this.s.openApiKeys(required);
      return;
    }
    // Apply to the UI only if the command actually went out, so a send that fails
    // (socket down) doesn't leave the UI showing a model the server never switched to.
    if (!this.sendCommand("set_model", id)) return;
    this.s.setModel(id);
    this.store.setState({ meta: this.meta() });
    this.s.addSysNote(`Switched the model to ${id}.`);
    this.refreshContextWindow();
  }

  // --- API keys -------------------------------------------------------------

  /** The provider whose key is missing for `id`, or null if the model needs none
   *  (a local model, or an unknown provider the server guard will handle). */
  private credentialMissingFor(id: string): CredentialPrompt | null {
    const provider = this.s.models.find((m) => m.id === id)?.provider ?? id.split(":")[0];
    const cred = this.s.credentials.find((c) => c.provider === provider);
    if (!cred || cred.is_set) return null;
    return { provider: cred.provider, label: cred.label, env_var: cred.env_var };
  }

  /** Open the keys modal in "manage" mode (change a key at any time). */
  openKeys(): void {
    this.s.openApiKeys(null);
  }

  /** Save a provider's key. Returns an error message on failure, else null. */
  async submitCredential(provider: string, value: string): Promise<string | null> {
    try {
      await api.setCredential(provider, value);
    } catch (e) {
      return e instanceof Error ? e.message : String(e);
    }
    await this.refreshCredentials();
    return null;
  }

  /** Close the keys modal. With `retry`, re-run the action the key was blocking. */
  closeKeys(retry: boolean): void {
    const resume = this.keyRetry;
    this.keyRetry = null;
    this.s.closeApiKeys();
    if (retry) resume?.();
  }

  setApproval(mode: string): void {
    if (!APPROVAL_MODES.includes(mode)) {
      this.s.addSysNote(`Usage: /approval-mode ${APPROVAL_MODES.join(" | ")}.`);
      return;
    }
    if (this.sendCommand("set_approval", mode)) this.s.addSysNote(`Approval policy set to ${mode}.`);
  }

  private sendCommand(command: "set_model" | "set_approval" | "add_dir" | "compact", arg: string): boolean {
    if (!this.transport.isOpen()) return false;
    // The server applies these between turns and rejects them mid-turn, so don't
    // send (or print an optimistic note) while a turn is running.
    if (this.s.busy) {
      this.s.addSysNote("Finish the current turn before changing settings.", "warn");
      return false;
    }
    return this.transport.send({ type: "command", command, arg });
  }

  compact(): void {
    if (this.s.busy) {
      this.s.addSysNote("Finish the current turn before compacting.", "warn");
      return;
    }
    // Begin the working state only once the command is actually sent: doing it first
    // would trip sendCommand's own busy guard and strand the spinner with no turn to
    // end it. The server's notice (or an error) clears it.
    if (this.sendCommand("compact", "")) {
      this.s.addSysNote("Compacting the conversation…");
      this.s.beginWorking();
    }
  }

  addDir(path: string): void {
    if (!path) {
      this.s.addSysNote("Usage: /add-dir <path>.");
      return;
    }
    this.sendCommand("add_dir", path);
  }

  transcript(fmt: "html" | "md"): void {
    const { sessionId } = this.s;
    if (!sessionId) return this.s.addSysNote("No active session.");
    const a = document.createElement("a");
    a.href = api.transcriptUrl(sessionId, fmt);
    a.download = `transcript.${fmt}`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    this.s.addSysNote(`Downloading the transcript (${fmt}) to share.`);
  }

  memory(): void {
    const { sessionId } = this.s;
    if (!sessionId) return this.s.addSysNote("No active session.");
    this.s.pinDoc(api.memoryUrl(sessionId), "Workspace memory", "memory");
  }

  private async showContext(): Promise<void> {
    const { sessionId } = this.s;
    if (!sessionId) return this.s.addSysNote("No active session.");
    const data = await api.context(sessionId);
    if (!data.markdown) return this.s.addSysNote("Could not read context usage.", "warn");
    this.s.addContext(data.markdown);
  }

  private copyLast(): void {
    const last = [...this.s.items].reverse().find((it) => it.kind === "assistant");
    if (!last || last.kind !== "assistant") return this.s.addSysNote("No assistant message to copy yet.");
    navigator.clipboard.writeText(last.text).then(
      () => this.s.addSysNote("Copied the last reply to the clipboard."),
      () => this.s.addSysNote("Could not access the clipboard.", "warn"),
    );
  }

  // --- files ----------------------------------------------------------------

  setComposerSink(sink: ((text: string) => void) | null): void {
    this.composerSink = sink;
  }

  async upload(file: File): Promise<string | null> {
    const { sessionId } = this.s;
    if (!sessionId) {
      this.s.addSysNote("Start a session before uploading.", "warn");
      return null;
    }
    try {
      const { path } = await api.uploadFile(sessionId, file);
      this.s.addSysNote(`Uploaded ${path}. It is referenced below; ask the agent to use it.`);
      this.composerSink?.(path);
      return path;
    } catch (e) {
      this.s.addSysNote(`Upload failed: ${e}`, "warn");
      return null;
    }
  }

  // --- session lifecycle ----------------------------------------------------

  async resume(id: string, sim?: string): Promise<void> {
    if (id === this.s.sessionId) return;
    if (sim) this.store.setState({ sim });
    try {
      await this.reenter(id);
    } catch {
      this.store.setState({ meta: "could not resume" });
      this.s.addSysNote("Could not resume that session.", "warn");
    }
  }

  // Reopen a session: reattach to it if the server kept it live (its Julia REPL
  // survives), otherwise resume it from disk (a fresh kernel). Throws on failure;
  // callers decide how to report it.
  private async reenter(id: string): Promise<void> {
    const seq = ++this.reenterSeq;
    const stale = () => seq !== this.reenterSeq; // a newer reenter superseded this one
    // Reconnecting to the current session (vs switching to a different one): if the
    // server kept it live, the thread, canvas, and any pending approval are already
    // correct, so reattach without wiping and re-fetching the conversation.
    const reconnecting = id === this.s.sessionId;
    this.transport.close();
    this.store.setState({ meta: `resuming ${this.s.sim ?? ""}…` });
    const body = await api.resumeSession(id, {
      sim: this.s.sim || undefined,
      model: this.s.model || undefined,
      // An older session reopened from the sidebar is brought up to date with
      // what the application has selected now, and pointed at where it listens
      // now, rather than what it had when the conversation was last touched.
      host_context: hostContext.current() ?? undefined,
      host_api: hostApiUrl ?? undefined,
    });
    if (stale()) return;
    if (!(reconnecting && body.kernel_restarted === false)) {
      // A from-disk resume or a switch to another session: rebuild from the record.
      // Lay the history down before opening the socket so a live frame can't
      // interleave with the replay.
      this.s.reset();
      this.s.setSession(body.session_id, "");
      this.s.replay(await api.messages(id));
      if (stale()) return;
    }
    this.store.setState({ meta: this.meta() });
    this.transport.open(body.session_id);
    this.flushQueuedPrompt();
    this.s.addSysNote(body.kernel_restarted === false ? REATTACHED_NOTE : RESTARTED_NOTE);
    this.refreshHistory();
  }

  // Recover after the socket dropped (a network blip, sleep/wake, or a proxy idle
  // timeout). Reattaches to the still-live session (keeping its REPL), or resumes it
  // from disk if the server let it go. Retried with backoff against the brief window
  // where the server has not yet released the old connection.
  private async reconnect(): Promise<void> {
    const id = this.s.sessionId;
    if (!id || this.recovering) return;
    this.recovering = true;
    this.store.setState({ reconnecting: true });
    try {
      for (let attempt = 0; ; attempt++) {
        try {
          await this.reenter(id); // the new socket clears `reconnecting` on open
          return;
        } catch {
          if (attempt >= 3) throw new Error("unreachable");
          await sleep(Math.min(500 * 2 ** attempt, 4000));
        }
      }
    } catch {
      this.queuedPrompt = null;
      this.store.setState({ reconnecting: false, busy: false, working: false });
      this.s.addSysNote("Connection lost. Reload the page to continue.", "warn");
    } finally {
      this.recovering = false;
    }
  }

  async newChat(): Promise<void> {
    // Detach only — the previous session's host (its kernel, its live figures)
    // stays alive on the server, so switching back through the sidebar finds
    // the session as it was left, live plots included. Sessions are released
    // when the server shuts down.
    this.transport.close();
    this.s.reset();
    this.store.setState({ sessionId: null });
    await this.startSession();
  }

  async refreshHistory(): Promise<void> {
    this.s.setHistory(await api.history());
  }

  // --- prompt history (↑/↓) -------------------------------------------------

  private pushPrompt(text: string): void {
    if (this.promptHistory.at(-1) !== text) this.promptHistory.push(text);
    this.historyPos = -1;
  }

  recall(dir: -1 | 1, draft: string): string | null {
    if (!this.promptHistory.length) return null;
    if (this.historyPos === -1) {
      if (dir > 0) return null;
      this.historyDraft = draft;
      this.historyPos = this.promptHistory.length - 1;
    } else {
      this.historyPos += dir;
    }
    if (this.historyPos < 0) this.historyPos = 0;
    if (this.historyPos >= this.promptHistory.length) {
      this.historyPos = -1;
      return this.historyDraft;
    }
    return this.promptHistory[this.historyPos];
  }

  // --- notifications --------------------------------------------------------

  private requestNotifyPermission(): void {
    if (this.notifyAsked || !("Notification" in window) || Notification.permission !== "default") return;
    this.notifyAsked = true;
    Notification.requestPermission().catch(() => {});
  }

  private notifyDone(body: string): void {
    if (document.hidden && "Notification" in window && Notification.permission === "granted") {
      try {
        new Notification("jutul-agent", { body });
      } catch {
        /* best-effort */
      }
    }
  }
}
