// The REST surface of the server (everything except the per-turn WebSocket),
// typed. Mirrors the FastAPI routes in interfaces/server/app.py.

import type { HostContext } from "./hostContext";
import type { ReplayMessage } from "./protocol";
import { withBase } from "./basePath";

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

/** What an installed capability calls this UI, when it takes the simulator's place
 *  on the welcome screen. Each field falls back to the simulator's own on its own. */
export interface Branding {
  display_name?: string;
  tagline?: string;
  examples?: string[];
}

export interface HistoryEntry {
  id: string;
  title: string;
  started: string;
  last_active: string;
  sim: string;
  /** The session's host is still in memory, so a resume keeps its Julia REPL state.
   * Optional: an older server does not send it, and absent must not read as live. */
  live?: boolean;
}

export interface SimulatorsResponse {
  simulators: string[];
  default: string | null;
  details: Record<string, SimDetails>;
  branding?: Branding | null;
}

/** What the server runs, from `GET /version`. Each part is `null` when unknown. */
export interface VersionInfo {
  jutul_agent: string | null;
  /** A newer jutul-agent release, when the update check knows of one. */
  update: string | null;
  simulator: { package: string; version: string | null } | null;
  /** This folder's Julia environment is older than this version supports. */
  env_update?: boolean;
  /** Identifies the server process; it changes when the server restarts. */
  boot?: string | null;
}

/** A simulator package in the folder's Julia environment that needs updating. */
export interface OutdatedPackage {
  package: string;
  version: string;
  required: string;
}

/** An update started from the browser, as it runs. */
export interface UpdateJob {
  state: "running" | "restarting" | "done" | "failed";
  install: boolean;
  error: string | null;
  /** Output lines after the `since` the request asked for. */
  lines: string[];
  /** Pass as `since` next time to get only newer lines. */
  next: number;
}

/** What an update would do here, from `GET /update`. */
export interface UpdateStatus {
  current: string | null;
  latest: string | null;
  /** Served on this computer, so the browser may update it. */
  local: boolean;
  /** jutul-agent can upgrade itself from here; otherwise `install_note` says how. */
  can_install: boolean;
  install_note: string | null;
  /** Upgrading jutul-agent continues in a separate window, which it then runs in. */
  separate_window: boolean;
  env: { outdated: OutdatedPackage[]; sysimage: boolean } | null;
  job: UpdateJob | null;
}

export type SysimageChoice = "rebuild" | "off";

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

// Endpoints use unprefixed paths; apply the deployment prefix only at this boundary.
function request(path: string, init?: RequestInit): Promise<Response> {
  return fetch(withBase(path), init);
}

async function getJSON<T>(url: string, fallback: T): Promise<T> {
  try {
    const resp = await request(url);
    if (!resp.ok) return fallback;
    return (await resp.json()) as T;
  } catch {
    return fallback;
  }
}

async function postJSON<T>(url: string, body: unknown): Promise<T> {
  const resp = await request(url, {
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
    getJSON<SimulatorsResponse>("/simulators", {
      simulators: [],
      default: null,
      details: {},
    }),

  version: () =>
    getJSON<VersionInfo>("/version", { jutul_agent: null, update: null, simulator: null }),

  /** `null` when the server can't be reached, as while it restarts. */
  updateStatus: async (since = 0): Promise<UpdateStatus | null> => {
    try {
      const resp = await request(`/update?since=${since}`);
      return resp.ok ? ((await resp.json()) as UpdateStatus) : null;
    } catch {
      return null;
    }
  },

  startUpdate: (body: { install: boolean; sysimage: SysimageChoice | null }) =>
    postJSON<UpdateStatus>("/update", body),

  models: () =>
    getJSON<ModelsResponse>("/models", { default: null, providers: [], models: [] }),

  liveModels: () => getJSON<{ models: ModelInfo[] }>("/models/live", { models: [] }),

  credentials: async (): Promise<CredentialInfo[]> => {
    const data = await getJSON<{ path: string; providers: CredentialInfo[] }>(
      "/credentials",
      {
        path: "",
        providers: [],
      },
    );
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
  ) =>
    postJSON<{ session_id: string; kernel_restarted: boolean }>(
      `/sessions/${id}/resume`,
      body,
    ),

  deleteSession: (id: string) =>
    request(`/sessions/${id}`, { method: "DELETE" }).catch(() => undefined),

  context: (id: string) =>
    getJSON<{ markdown: string }>(`/sessions/${id}/context`, { markdown: "" }),

  uploadFile: async (id: string, file: File): Promise<{ path: string }> => {
    const fd = new FormData();
    fd.append("file", file);
    const resp = await request(`/sessions/${id}/upload`, { method: "POST", body: fd });
    if (!resp.ok) throw new Error(await resp.text().catch(() => resp.statusText));
    return (await resp.json()) as { path: string };
  },

  transcriptUrl: (id: string, fmt: "html" | "md") =>
    withBase(`/sessions/${id}/transcript?format=${fmt}`),

  memoryUrl: (id: string) => withBase(`/sessions/${id}/memory`),
};
