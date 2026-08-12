// The application embedding this UI, as the browser sees it.
//
// A host app (a desktop tool that opens this UI in a frame) launches us with two
// base64 query parameters: `?data=`, its current selection as a JSON object of
// its own identifiers, and `?apiurl=`, where its local HTTP API listens.
//
// The selection belongs to the *page*, not to any one session: every session
// started or resumed in this frame is told the current one, so what the agent
// believes is selected always matches what the user sees in the application. It
// can change while the page is open: the host reloads the frame with a new
// `?data=` (a fresh page, so nothing to do here) or posts a message to us, which
// is the only way to update without losing the conversation.
//
// The API URL is different: it is only good for this launch, so it is read once
// and never stored anywhere.

/** The host's payload: opaque to us, meaningful to the host's own tools. */
export type HostContext = Record<string, unknown>;

/** The query parameter the host encodes its selection into. */
export const HOST_CONTEXT_PARAM = "data";

/** The query parameter the host encodes its API's base URL into. */
export const HOST_API_PARAM = "apiurl";

/** The `postMessage` envelope a host uses to update the selection in place. */
export const HOST_CONTEXT_MESSAGE = "jutul-agent:host-context";

/** Decode a launch parameter: base64 (standard or url-safe) of UTF-8 text.
 *
 *  Tolerant on purpose, because the encoder is someone else's code: url-safe and
 *  standard alphabets both decode, padding is optional, and a space is read as
 *  the "+" it almost certainly was. `URLSearchParams` percent-decodes for us but
 *  also applies form decoding, where "+" means a space, and "+" is a perfectly
 *  ordinary base64 character.
 */
function decodeBase64(raw: string): string | null {
  try {
    let b64 = raw.trim().replace(/ /g, "+").replace(/-/g, "+").replace(/_/g, "/");
    b64 += "=".repeat((4 - (b64.length % 4)) % 4);
    const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
    return new TextDecoder().decode(bytes);
  } catch {
    return null;
  }
}

/** The selection from its encoded form, or null if it is not a JSON object. A
 *  malformed parameter costs the selection, never the app starting. */
export function decodeHostContext(raw: string): HostContext | null {
  const text = decodeBase64(raw);
  if (text === null) return null;
  try {
    const parsed = JSON.parse(text);
    return parsed && typeof parsed === "object" && !Array.isArray(parsed)
      ? (parsed as HostContext)
      : null;
  } catch {
    return null;
  }
}

/** The host API's base URL from its encoded form, without a trailing slash.
 *
 *  Returns the parsed URL's *own* serialization, never the text as it arrived:
 *  the value is later interpolated into code and into request URLs, and `new URL`
 *  happily accepts a quote or a backtick in a path, which it percent-encodes only
 *  when it serializes. Only http(s) is accepted, and a query or fragment is
 *  refused, because a base URL has no use for either, so one is a sign it is
 *  not what it claims to be.
 */
export function decodeHostApiUrl(raw: string): string | null {
  const text = decodeBase64(raw)?.trim();
  if (!text) return null;
  try {
    const url = new URL(text);
    if (url.protocol !== "http:" && url.protocol !== "https:") return null;
    if (url.search || url.hash || url.username || url.password) return null;
    return `${url.origin}${url.pathname}`.replace(/\/+$/, "");
  } catch {
    return null;
  }
}

/** The selection encoded in a query string, or null if it carries none. */
export function readHostContext(search: string): HostContext | null {
  const raw = new URLSearchParams(search).get(HOST_CONTEXT_PARAM);
  return raw ? decodeHostContext(raw) : null;
}

/** The host API's base URL encoded in a query string, or null if it carries none. */
export function readHostApiUrl(search: string): string | null {
  const raw = new URLSearchParams(search).get(HOST_API_PARAM);
  return raw ? decodeHostApiUrl(raw) : null;
}

/** The selection carried by a `postMessage`, or null if it is not one of ours.
 *
 *  A host may send the decoded object or the same base64 string it would have put
 *  in the URL, so one encoder on their side serves both paths.
 */
export function hostContextFromMessage(data: unknown): HostContext | null {
  if (!data || typeof data !== "object") return null;
  const msg = data as Record<string, unknown>;
  if (msg.type !== HOST_CONTEXT_MESSAGE) return null;
  const payload = msg.context ?? msg.data;
  if (typeof payload === "string") return decodeHostContext(payload);
  if (payload && typeof payload === "object" && !Array.isArray(payload)) {
    return payload as HostContext;
  }
  return null;
}

export interface HostContextSource {
  /** The selection as it stands, or null when this UI was not opened from a host. */
  current(): HostContext | null;
  /** Observe changes; returns an unsubscribe. Only fires when the value differs. */
  subscribe(fn: (context: HostContext | null) => void): () => void;
  /** Listen for host updates. Returns a teardown; safe to call once per page. */
  listen(): () => void;
}

export function createHostContextSource(win: Window): HostContextSource {
  let value = readHostContext(win.location.search);
  const listeners = new Set<(context: HostContext | null) => void>();

  const set = (next: HostContext | null) => {
    // Compared by serialization: the host re-sends its whole selection on every
    // change, so an unchanged one arriving again must not rebuild the agent.
    if (JSON.stringify(next) === JSON.stringify(value)) return;
    value = next;
    for (const fn of listeners) fn(value);
  };

  return {
    current: () => value,
    subscribe(fn) {
      listeners.add(fn);
      return () => listeners.delete(fn);
    },
    listen() {
      const onMessage = (event: MessageEvent) => {
        // Only the frame that embedded us may drive the selection. The origin is
        // deliberately not pinned: the host is a local desktop application whose
        // origin is not known at build time, and it could equally have put this
        // payload in our URL. The value is treated as data throughout: it is
        // JSON, it is never executed, and the agent is told these are the host's
        // identifiers.
        if (win.parent && event.source !== win.parent) return;
        const next = hostContextFromMessage(event.data);
        if (next) set(next);
      };
      win.addEventListener("message", onMessage);
      return () => win.removeEventListener("message", onMessage);
    },
  };
}

/** The page's one source. Reads `?data=` at import, before any session starts. */
export const hostContext: HostContextSource =
  typeof window === "undefined"
    ? {
        current: () => null,
        subscribe: () => () => {},
        listen: () => () => {},
      }
    : createHostContextSource(window);

/** Where the host application's API listens, for this launch only.
 *
 *  Deliberately a constant rather than part of the source above: the value only
 *  describes this launch, and a reload re-reads it here anyway. Nothing persists
 *  it. */
export const hostApiUrl: string | null =
  typeof window === "undefined" ? null : readHostApiUrl(window.location.search);
