// The REST surface of the server (everything except the per-turn WebSocket),
// typed. Mirrors the FastAPI routes in interfaces/server/app.py.

import type { HostContext } from "./hostContext";
import type { ReplayMessage } from "./protocol";

export interface ModelInfo {
  id: string;
  label: string;
  provider: string;
  note?: string | null;
}

export interface SimDetails {
  display_name?: string;
  examples?: string[];
}

export interface HistoryEntry {
  id: string;
  title: string;
  started: string;
  last_active: string;
  sim: string;
}

export interface SimulatorsResponse {
  simulators: string[];
  default: string | null;
  details: Record<string, SimDetails>;
}

export interface ModelsResponse {
  default: string | null;
  providers: string[];
  models: ModelInfo[];
}

/** One provider's API-key state, from `GET /credentials`. */
export interface CredentialInfo {
  provider: string;
  label: string;
  env_var: string;
  is_set: boolean;
  masked: string | null;
  source: "file" | "environment" | "none";
  shadowed: boolean;
}

/** An HTTP error that carries the server's parsed `detail`, so callers can act on a
 *  structured error (e.g. a `credential_required` create-session refusal). */
export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly detail: unknown,
  ) {
    super(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
}

async function getJSON<T>(url: string, fallback: T): Promise<T> {
  try {
    const resp = await fetch(url);
    if (!resp.ok) return fallback;
    return (await resp.json()) as T;
  } catch {
    return fallback;
  }
}

async function postJSON<T>(url: string, body: unknown): Promise<T> {
  const resp = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    // FastAPI wraps an error in {detail: ...}; surface the detail so a structured
    // error (a dict) stays usable instead of collapsing to a string.
    const text = await resp.text().catch(() => resp.statusText);
    let detail: unknown = text;
    try {
      const parsed = JSON.parse(text);
      detail = "detail" in parsed ? parsed.detail : parsed;
    } catch {
      /* not JSON; keep the text */
    }
    throw new ApiError(resp.status, detail);
  }
  return (await resp.json()) as T;
}

export const api = {
  simulators: () =>
    getJSON<SimulatorsResponse>("/simulators", { simulators: [], default: null, details: {} }),

  models: () =>
    getJSON<ModelsResponse>("/models", { default: null, providers: [], models: [] }),

  credentials: async (): Promise<CredentialInfo[]> => {
    const data = await getJSON<{ path: string; providers: CredentialInfo[] }>("/credentials", {
      path: "",
      providers: [],
    });
    return data.providers;
  },

  setCredential: (provider: string, value: string) =>
    postJSON<{ provider: string; env_var: string; path: string }>("/credentials", {
      provider,
      value,
    }),

  modelWindow: (model: string) =>
    getJSON<{ model: string; window: number | null }>(
      `/models/window?model=${encodeURIComponent(model)}`,
      { model, window: null },
    ),

  history: async (limit = 40): Promise<HistoryEntry[]> => {
    const data = await getJSON<{ sessions: HistoryEntry[] }>(
      `/sessions/history?limit=${limit}`,
      { sessions: [] },
    );
    return data.sessions;
  },

  messages: async (id: string): Promise<ReplayMessage[]> => {
    const data = await getJSON<{ messages: ReplayMessage[] }>(
      `/sessions/${id}/messages`,
      { messages: [] },
    );
    return data.messages;
  },

  // `host_context` is the embedding application's current selection and
  // `host_api` is where its own API listens. Both are sent on resume as well as
  // create because they describe the application now, not the stored session:
  // omitting the selection (a UI opened outside the host) leaves the session's
  // stored one alone, while omitting the URL drops the host's tools, which is
  // right, because the port it was on is gone.
  createSession: (body: {
    sim?: string;
    model?: string;
    host_context?: HostContext;
    host_api?: string;
  }) => postJSON<{ session_id: string }>("/sessions", body),

  resumeSession: (
    id: string,
    body: { sim?: string; model?: string; host_context?: HostContext; host_api?: string },
  ) => postJSON<{ session_id: string; kernel_restarted: boolean }>(`/sessions/${id}/resume`, body),

  deleteSession: (id: string) =>
    fetch(`/sessions/${id}`, { method: "DELETE" }).catch(() => undefined),

  context: (id: string) =>
    getJSON<{ markdown: string }>(`/sessions/${id}/context`, { markdown: "" }),

  uploadFile: async (id: string, file: File): Promise<{ path: string }> => {
    const fd = new FormData();
    fd.append("file", file);
    const resp = await fetch(`/sessions/${id}/upload`, { method: "POST", body: fd });
    if (!resp.ok) throw new Error(await resp.text().catch(() => resp.statusText));
    return (await resp.json()) as { path: string };
  },

  transcriptUrl: (id: string, fmt: "html" | "md") =>
    `/sessions/${id}/transcript?format=${fmt}`,

  memoryUrl: (id: string) => `/sessions/${id}/memory`,
};
