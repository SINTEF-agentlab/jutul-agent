// The API-keys modal. It opens on its own when the chosen model needs a provider
// key the server doesn't have (blocking the session or a model switch), and from
// the top bar's "Keys" button to change a key at any time. Keys are saved to the
// server's global .env via POST /credentials; only masked previews are shown.

import { useEffect, useRef, useState } from "react";

import type { CredentialInfo } from "../api";
import { useController, useSel } from "../context";
import { CloseIcon } from "../icons";

export function ApiKeys() {
  const open = useSel((s) => s.apiKeys.open);
  const required = useSel((s) => s.apiKeys.required);
  const credentials = useSel((s) => s.credentials);
  const controller = useController();

  const [values, setValues] = useState<Record<string, string>>({});
  const [saving, setSaving] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const requiredInput = useRef<HTMLInputElement>(null);

  // Focus the provider that is being asked for, so the user can type straight away.
  useEffect(() => {
    if (open && required) requiredInput.current?.focus();
  }, [open, required]);

  if (!open) return null;

  // Always have a row for the required provider, even if the status list is empty.
  const rows: CredentialInfo[] =
    credentials.length || !required
      ? credentials
      : [
          {
            provider: required.provider,
            label: required.label,
            env_var: required.env_var,
            is_set: false,
            masked: null,
            source: "none",
            shadowed: false,
          },
        ];

  const requiredSatisfied =
    !required || rows.some((c) => c.provider === required.provider && c.is_set);

  const save = async (provider: string) => {
    const value = (values[provider] || "").trim();
    if (!value) return;
    setSaving(provider);
    setError(null);
    const err = await controller.submitCredential(provider, value);
    setSaving(null);
    if (err) setError(err);
    else setValues((v) => ({ ...v, [provider]: "" }));
  };

  return (
    <div className="modal-backdrop" onClick={() => controller.closeKeys(false)}>
      <div
        className="modal api-keys"
        role="dialog"
        aria-modal="true"
        aria-label="API keys"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="modal-head">
          <h2>API keys</h2>
          <button className="icon-btn" aria-label="Close" onClick={() => controller.closeKeys(false)}>
            <CloseIcon />
          </button>
        </div>

        {required ? (
          <p className="api-keys-banner">
            {required.label} needs an API key before this can run. Paste it below to continue.
          </p>
        ) : (
          <p className="api-keys-sub">
            Add or replace a provider key. Keys are saved on this machine and used right away.
          </p>
        )}

        <div className="api-keys-rows">
          {rows.map((c) => (
            <div className="api-keys-row" key={c.provider}>
              <div className="api-keys-meta">
                <span className="api-keys-name">{c.label}</span>
                <code className="api-keys-var">{c.env_var}</code>
                <span className={`api-keys-status ${statusClass(c)}`}>{statusText(c)}</span>
              </div>
              <div className="api-keys-entry">
                <input
                  ref={required?.provider === c.provider ? requiredInput : undefined}
                  type="password"
                  autoComplete="off"
                  placeholder={c.is_set ? "Replace the saved key" : `Paste ${c.env_var}`}
                  value={values[c.provider] || ""}
                  onChange={(e) => setValues((v) => ({ ...v, [c.provider]: e.target.value }))}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") void save(c.provider);
                  }}
                />
                <button
                  className="btn"
                  disabled={saving === c.provider || !(values[c.provider] || "").trim()}
                  onClick={() => void save(c.provider)}
                >
                  {saving === c.provider ? "Saving…" : "Save"}
                </button>
              </div>
            </div>
          ))}
        </div>

        {error ? <p className="api-keys-error">{error}</p> : null}

        <div className="modal-foot">
          {required ? (
            <button
              className="btn primary"
              disabled={!requiredSatisfied}
              onClick={() => controller.closeKeys(true)}
            >
              Continue
            </button>
          ) : (
            <button className="btn primary" onClick={() => controller.closeKeys(false)}>
              Done
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

function statusText(c: CredentialInfo): string {
  if (!c.is_set) return "not set";
  if (c.source === "environment") return `from your environment (${c.masked})`;
  if (c.shadowed) return `saved, overridden by your environment (${c.masked})`;
  return `saved (${c.masked})`;
}

function statusClass(c: CredentialInfo): string {
  if (!c.is_set) return "missing";
  if (c.shadowed) return "warn";
  return "ok";
}
