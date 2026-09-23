// Public URL prefix the server injected into index.html (e.g. "/restricted"
// when an SSO wrapper reverse-proxies the UI). Empty means the server is at "/".
// API and WebSocket calls are origin-absolute, so they must include this prefix
// when the page is not served from the site root.

declare global {
  interface Window {
    __JUTUL_BASE_PATH__?: string;
  }
}

/** Normalize to "" or a leading-slash prefix with no trailing slash. */
export function normalizeBasePath(raw: string | null | undefined): string {
  if (!raw) return "";
  let p = raw.trim();
  if (!p || p === "/") return "";
  if (!p.startsWith("/")) p = `/${p}`;
  return p.replace(/\/+$/, "");
}

/** The base path the server injected, or "" when absent / in tests. */
export function basePath(): string {
  if (typeof window === "undefined") return "";
  return normalizeBasePath(window.__JUTUL_BASE_PATH__);
}

/** Prefix a site-relative path with the public base path. */
export function withBase(path: string): string {
  if (!path.startsWith("/")) return path;
  const prefix = basePath();
  return prefix ? `${prefix}${path}` : path;
}
