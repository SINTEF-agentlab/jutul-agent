// The collapsible left sidebar: brand, "New chat", and the resumable session
// history (most recently used first). Clicking a session resumes it.

import { useController, useSel } from "../context";
import { timeAgo } from "../format";
import { PlusIcon } from "../icons";

const LIVE_LABEL = "Live: resumes with its Julia session intact";

export function Sidebar({ collapsed }: { collapsed: boolean }) {
  const history = useSel((s) => s.history);
  const sessionId = useSel((s) => s.sessionId);
  const controller = useController();

  // A session earns a title from its first prompt; ones without are empty/abandoned
  // new-chats, so leave them out to keep the list to real conversations.
  const sessions = history.filter((s) => s.title);

  return (
    <aside className={`sidebar${collapsed ? " collapsed" : ""}`}>
      <div className="sidebar-head">
        <span className="brand">
          <span className="brand-mark" /> jutul-agent
        </span>
      </div>
      <button className="sidebar-new" title="Start a new chat" onClick={() => controller.newChat()}>
        <PlusIcon /> New chat
      </button>
      <div className="sidebar-section">Chats</div>
      <div className="sidebar-list">
        {sessions.length === 0 ? (
          <div className="history-empty">No past sessions yet.</div>
        ) : (
          sessions.map((s) => (
            <button
              key={s.id}
              className={`history-item${s.id === sessionId ? " current" : ""}`}
              title={s.title}
              onClick={() => controller.resume(s.id, s.sim)}
            >
              <div className="h-title">
                {/* Its Julia is still running, so resuming picks up where it left
                    off. Sessions without the dot replay against a fresh Julia. */}
                {s.live && (
                  <span
                    className="h-live"
                    title={LIVE_LABEL}
                    aria-label={LIVE_LABEL}
                    role="img"
                  />
                )}
                <span className="h-title-text">{s.title || "Untitled session"}</span>
              </div>
              {/* Just the time. The simulator is a property of the folder, not of
                  the chat, and the top bar already names it. */}
              <div className="h-meta">{timeAgo(s.last_active || s.started)}</div>
            </button>
          ))
        )}
      </div>
    </aside>
  );
}
