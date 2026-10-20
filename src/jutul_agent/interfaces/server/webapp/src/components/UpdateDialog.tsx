// The update dialog, opened from the foot of the sidebar. It says what an update
// does here, runs it on the server (POST /update), shows the output as it arrives,
// and follows the server through its restart into the new version.

import { useEffect, useLayoutEffect, useRef, useState } from "react";

import { terminalSegments } from "../ansi";
import type { SysimageChoice, UpdateStatus } from "../api";
import { useController, useSel } from "../context";
import { CloseIcon } from "../icons";

export function UpdateDialog() {
  const update = useSel((s) => s.update);
  const controller = useController();
  const [sysimage, setSysimage] = useState<SysimageChoice>("rebuild");

  if (!update.open) return null;
  const { status, phase } = update;
  const confirming = phase === "idle" || phase === "starting";
  const busy = !confirming && phase !== "done" && phase !== "failed";
  const plan = status ? describe(status) : null;

  return (
    <div className="modal-backdrop" onClick={() => controller.closeUpdate()}>
      <div
        className="modal update-dialog"
        role="dialog"
        aria-modal="true"
        aria-label="Update"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="modal-head">
          <h2>{plan?.title ?? "Update"}</h2>
          {!busy && phase !== "starting" && (
            <button
              className="icon-btn"
              aria-label="Close"
              onClick={() => controller.closeUpdate()}
            >
              <CloseIcon />
            </button>
          )}
        </div>

        {confirming ? (
          !status ? (
            <p className="update-text">Checking…</p>
          ) : (
            <>
              {plan!.lines.map((line) => (
                <p className="update-text" key={line}>
                  {line}
                </p>
              ))}
              {plan!.note && <p className="update-note">{plan!.note}</p>}
              {plan!.action && status.env?.sysimage && (
                <fieldset className="update-sysimage">
                  <legend>
                    This folder starts Julia from a system image. A new environment needs a new
                    image:
                  </legend>
                  <label>
                    <input
                      type="radio"
                      checked={sysimage === "rebuild"}
                      onChange={() => setSysimage("rebuild")}
                    />
                    Rebuild it (about 20 minutes more)
                  </label>
                  <label>
                    <input
                      type="radio"
                      checked={sysimage === "off"}
                      onChange={() => setSysimage("off")}
                    />
                    Stop using one (build it again later with{" "}
                    <code>jutul-agent sysimage build</code>)
                  </label>
                </fieldset>
              )}
              {update.error && <p className="update-error">{update.error}</p>}
              {plan!.action && (
                <p className="update-sub">
                  Open chats are closed while it runs; their conversations are kept. This takes a
                  few minutes.
                  {plan!.action.install && " jutul-agent then restarts and this page reloads."}
                </p>
              )}
            </>
          )
        ) : (
          <Progress lines={update.lines} phase={phase} error={update.error} />
        )}

        <div className="modal-foot">
          {confirming && plan?.action ? (
            <>
              <button
                className="btn"
                disabled={phase === "starting"}
                onClick={() => controller.closeUpdate()}
              >
                Cancel
              </button>
              <button
                className="btn primary"
                disabled={phase === "starting"}
                onClick={() =>
                  void controller.startUpdate(
                    plan.action!.install,
                    status?.env?.sysimage ? sysimage : null,
                  )
                }
              >
                {phase === "starting" ? "Starting…" : plan.action.label}
              </button>
            </>
          ) : (
            !busy && (
              <button className="btn primary" onClick={() => controller.closeUpdate()}>
                Close
              </button>
            )
          )}
        </div>
      </div>
    </div>
  );
}

interface Plan {
  title: string;
  lines: string[];
  note: string | null;
  action: { install: boolean; label: string } | null;
}

/** What the dialog offers, given what the server can do. */
export function describe(status: UpdateStatus): Plan {
  const outdated = status.env?.outdated ?? [];
  const found = outdated.map((p) => `${p.package} ${p.version}`).join(", ");
  const needed = outdated.map((p) => `${p.package} ${p.required}`).join(", ");
  const envLine = outdated.length
    ? `This folder's Julia environment has ${found}; this version of jutul-agent needs ${needed}.`
    : null;

  if (status.latest) {
    const lines = [`jutul-agent ${status.latest} is available. This is ${status.current}.`];
    if (status.can_install) {
      lines.push(
        "Updating installs it, then brings this folder's Julia environment up to what it " +
          "supports, keeping packages added to it.",
      );
      return {
        title: "Update jutul-agent",
        lines,
        note: null,
        action: { install: true, label: "Update" },
      };
    }
    if (envLine) lines.push(envLine);
    return {
      title: "Update jutul-agent",
      lines,
      note: status.install_note,
      // The environment can still be brought up to this version.
      action: envLine && status.local ? { install: false, label: "Update environment" } : null,
    };
  }
  if (envLine) {
    return {
      title: "Update the Julia environment",
      lines: [envLine, "Updating keeps packages added to it."],
      note: status.local ? null : status.install_note,
      action: status.local ? { install: false, label: "Update environment" } : null,
    };
  }
  return {
    title: "Up to date",
    lines: [
      `jutul-agent ${status.current ?? ""} and this folder's Julia environment are up to date.`,
    ],
    note: null,
    action: null,
  };
}

function Progress({
  lines,
  phase,
  error,
}: {
  lines: string[];
  phase: string;
  error: string | null;
}) {
  const ref = useRef<HTMLPreElement>(null);
  const [follow, setFollow] = useState(true);
  // Keep the newest output in view unless the user scrolled up to read.
  useLayoutEffect(() => {
    const el = ref.current;
    if (el && follow) el.scrollTop = el.scrollHeight;
  }, [lines, follow]);
  useEffect(() => setFollow(true), [phase]);

  return (
    <>
      <p className={`update-state ${phase}`} role="status">
        {phase === "running" && (
          <>
            <span className="spinner" /> Updating…
          </>
        )}
        {phase === "restarting" && (
          <>
            <span className="spinner" /> Restarting jutul-agent. This page reloads when it's back.
          </>
        )}
        {phase === "done" && "Done. Close this to carry on."}
        {phase === "failed" && "The update didn't finish."}
      </p>
      {error && <p className="update-error">{error}</p>}
      {lines.length > 0 && (
        <pre
          ref={ref}
          className="tool-output update-log"
          onScroll={(e) => {
            const el = e.currentTarget;
            setFollow(el.scrollHeight - el.scrollTop - el.clientHeight < 24);
          }}
        >
          {terminalSegments(lines.join("\n")).map((seg, i) => (
            <span key={i} style={{ color: seg.color, fontWeight: seg.bold ? 600 : undefined }}>
              {seg.text}
            </span>
          ))}
        </pre>
      )}
    </>
  );
}
