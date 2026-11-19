// The approval card shown while a turn is paused for the user's decision. The
// allowed decisions and the "always allow" affordance come straight from the
// server's interrupt message.

import { useEffect, useRef } from "react";

import { useController, useSel } from "../context";
import { Markdown } from "../markdown";
import { argPreview } from "../toolPolicy";

export function Approval() {
  const pending = useSel((s) => s.pending);
  const controller = useController();
  const ref = useRef<HTMLDivElement>(null);

  // An approval needs attention, so bring it into view even if the user had
  // scrolled up — otherwise it can land below the fold.
  useEffect(() => {
    // Optional-call: not every environment implements scrollIntoView (jsdom).
    ref.current?.scrollIntoView?.({ behavior: "smooth", block: "center" });
  }, []);

  if (!pending) return null;
  const names = pending.actions.map((a) => a.label || a.name).join(", ");
  const allowEdits =
    pending.allowlist.length === 1 && pending.allowlist[0] === "file_edits";
  const canAlwaysAllow = pending.allowed.includes("approve") && pending.allowlist.length > 0;

  return (
    <div className="approval" ref={ref}>
      <div className="title">Approve {names}?</div>
      {pending.actions.map((a, i) => {
        // The server carries the full card body (command, content preview, or
        // on-disk diff) as markdown; the one-line arg preview is the fallback
        // for older servers. Approving must never show less than the TUI does.
        if (a.body) {
          return <Markdown key={i} text={a.body} className="approval-detail markdown" />;
        }
        const detail = argPreview(a.args) || a.description;
        return detail ? <pre key={i} className="approval-detail">{detail}</pre> : null;
      })}
      <div className="buttons">
        {pending.allowed
          .filter((d) => d !== "respond")
          .map((d) => (
            <button
              key={d}
              className={d === "approve" ? "btn primary" : "btn"}
              onClick={() => controller.sendDecision(d)}
            >
              {d}
            </button>
          ))}
        {canAlwaysAllow ? (
          <button
            className="btn"
            title="Approve, and don't ask again for this kind of action this session"
            onClick={() => controller.sendDecision("always_allow")}
          >
            {allowEdits ? "always allow edits" : "always allow"}
          </button>
        ) : null}
      </div>
      {pending.allowed.includes("respond") ? (
        <div className="approval-hint">…or type a reply below to send feedback.</div>
      ) : null}
    </div>
  );
}
