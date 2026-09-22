// The composer: the message box plus its keyboard-driven slash-command menu, model
// picker, prompt history (↑/↓), and file attach. The send button doubles as a stop
// button while a turn runs.

import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";

import { SLASH_COMMANDS, type SlashCommand } from "../controller";
import { useController, useSel } from "../context";
import { AttachIcon, SendIcon } from "../icons";

export function Composer() {
  const controller = useController();
  const busy = useSel((s) => s.busy);
  const pending = useSel((s) => s.pending);
  const models = useSel((s) => s.models);
  const model = useSel((s) => s.model);

  const [value, setValue] = useState("");
  const [slashIndex, setSlashIndex] = useState(0);
  const [modelOpen, setModelOpen] = useState(false);
  const [modelIndex, setModelIndex] = useState(0);
  const [refreshingModels, setRefreshingModels] = useState(false);

  const textarea = useRef<HTMLTextAreaElement>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const activeModelRef = useRef<HTMLDivElement>(null);

  const slashItems = useMemo<SlashCommand[]>(
    () =>
      value.startsWith("/") && !/\s/.test(value)
        ? SLASH_COMMANDS.filter((c) => c.name.startsWith(value))
        : [],
    [value],
  );
  const slashOpen = slashItems.length > 0;

  const placeholder = pending?.allowed.includes("respond")
    ? "Reply to the agent…"
    : "Message jutul-agent…";

  // Auto-grow the textarea up to a cap.
  useLayoutEffect(() => {
    const el = textarea.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 200) + "px";
  }, [value]);

  // An uploaded file's path (from the attach button or a drag-drop) is appended to
  // the message box through this sink, so the user can reference it in their prompt.
  useEffect(() => {
    controller.setComposerSink((t) => {
      setValue((v) => v + (v && !/\s$/.test(v) ? " " : "") + t + " ");
      textarea.current?.focus();
    });
    return () => controller.setComposerSink(null);
  }, [controller]);

  // Keep the highlighted model in view as the arrow keys move through the list.
  useEffect(() => {
    if (modelOpen) activeModelRef.current?.scrollIntoView?.({ block: "nearest" });
  }, [modelIndex, modelOpen]);

  useEffect(() => {
    setModelIndex((i) => Math.min(i, Math.max(0, models.length - 1)));
  }, [models.length]);

  // Dismiss the model picker on an outside click.
  useEffect(() => {
    if (!modelOpen) return;
    const onClick = (e: MouseEvent) => {
      if (!(e.target as HTMLElement)?.closest(".model-menu")) setModelOpen(false);
    };
    document.addEventListener("click", onClick);
    return () => document.removeEventListener("click", onClick);
  }, [modelOpen]);

  const focus = () => textarea.current?.focus();

  const refreshModels = async () => {
    setRefreshingModels(true);
    try {
      await controller.refreshAvailableModels();
    } finally {
      setRefreshingModels(false);
    }
  };

  const openModelMenu = () => {
    setModelIndex(Math.max(0, models.findIndex((m) => m.id === model)));
    setModelOpen(true);
    void refreshModels();
    focus();
  };

  const chooseModel = (id: string) => {
    setModelOpen(false);
    controller.setModel(id);
    setValue("");
  };

  const submit = () => {
    const text = value.trim();
    if (!text) return;
    if (text === "/model") {
      setValue("");
      openModelMenu();
      return;
    }
    controller.send(text);
    setValue("");
  };

  const completeSlash = (cmd: SlashCommand) => {
    setValue(cmd.name + (cmd.hint ? " " : ""));
    focus();
  };

  const recall = (dir: -1 | 1) => {
    const next = controller.recall(dir, value);
    if (next == null) return false;
    setValue(next);
    requestAnimationFrame(() => {
      const el = textarea.current;
      if (el) el.setSelectionRange(el.value.length, el.value.length);
    });
    return true;
  };

  const atStart = () =>
    textarea.current?.selectionStart === 0 && textarea.current?.selectionEnd === 0;

  const atEnd = () => {
    const el = textarea.current;
    return !!el && el.selectionStart === el.value.length && el.selectionEnd === el.value.length;
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Escape") {
      if (modelOpen) return setModelOpen(false);
      if (busy) return controller.stop();
      return;
    }
    if (modelOpen && models.length) {
      if (e.key === "ArrowDown") {
        e.preventDefault();
        return setModelIndex((i) => (i + 1) % models.length);
      }
      if (e.key === "ArrowUp") {
        e.preventDefault();
        return setModelIndex((i) => (i - 1 + models.length) % models.length);
      }
      if (e.key === "Enter" || e.key === "Tab") {
        e.preventDefault();
        return chooseModel(models[Math.min(modelIndex, models.length - 1)].id);
      }
      return;
    }
    if (slashOpen) {
      if (e.key === "ArrowDown") {
        e.preventDefault();
        return setSlashIndex((i) => (i + 1) % slashItems.length);
      }
      if (e.key === "ArrowUp") {
        e.preventDefault();
        return setSlashIndex((i) => (i - 1 + slashItems.length) % slashItems.length);
      }
      const highlighted = slashItems[Math.min(slashIndex, slashItems.length - 1)];
      if (e.key === "Tab") {
        e.preventDefault();
        return completeSlash(highlighted);
      }
      if (e.key === "Enter") {
        e.preventDefault();
        // A fully-typed command runs; a partial one completes to the highlighted item.
        if (SLASH_COMMANDS.some((c) => c.name === value.trim())) return submit();
        return completeSlash(highlighted);
      }
    } else if (e.key === "ArrowUp" && atStart()) {
      if (recall(-1)) e.preventDefault();
      return;
    } else if (e.key === "ArrowDown" && atEnd()) {
      // Only recall forward from the end of the draft, mirroring ArrowUp-at-start,
      // so pressing Down to move the cursor within a draft never clobbers it.
      if (recall(1)) e.preventDefault();
      return;
    }
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  const onAttach = (files: FileList | null) => {
    if (!files) return;
    for (const f of Array.from(files)) void controller.upload(f); // sink appends the path
  };

  const sendOrStop = () => {
    if (busy && !value.trim().startsWith("/")) controller.stop();
    else submit();
  };

  return (
    <footer className="composer-wrap">
      {modelOpen ? (
        <div className="slash-menu model-menu">
          <div className="slash-head model-head">
            <span>Switch model</span>
            <button type="button" onClick={() => void refreshModels()} disabled={refreshingModels}>
              {refreshingModels ? "Refreshing…" : "Refresh list"}
            </button>
          </div>
          {!models.length ? <div className="model-provider">No selectable models found.</div> : null}
          {models.map((m, i) => {
            const firstOfProvider = i === 0 || models[i - 1].provider !== m.provider;
            return (
              <div key={m.id}>
                {firstOfProvider ? <div className="model-provider">{m.provider}</div> : null}
                <div
                  ref={i === modelIndex ? activeModelRef : undefined}
                  className={`slash-item${i === modelIndex ? " active" : ""}`}
                  onClick={() => chooseModel(m.id)}
                >
                  <span className="slash-name">{m.label}</span>
                  <span className="slash-desc">{m.id}{m.note ? ` · ${m.note}` : ""}</span>
                </div>
              </div>
            );
          })}
        </div>
      ) : null}
      {slashOpen ? (
        <div className="slash-menu">
          {slashItems.map((c, i) => (
            <div
              key={c.name}
              className={`slash-item${i === Math.min(slashIndex, slashItems.length - 1) ? " active" : ""}`}
              onClick={() => completeSlash(c)}
            >
              <span className="slash-name">{c.name + (c.hint ? " " + c.hint : "")}</span>
              <span className="slash-desc">{c.desc}</span>
            </div>
          ))}
        </div>
      ) : null}
      <div className="composer">
        <button
          className="attach"
          title="Attach a file"
          aria-label="Attach a file"
          onClick={() => fileInput.current?.click()}
        >
          <AttachIcon />
        </button>
        <input
          ref={fileInput}
          type="file"
          hidden
          multiple
          onChange={(e) => {
            onAttach(e.target.files);
            e.target.value = "";
          }}
        />
        <textarea
          ref={textarea}
          rows={1}
          placeholder={placeholder}
          value={value}
          onChange={(e) => {
            setValue(e.target.value);
            setSlashIndex(0);
          }}
          onKeyDown={onKeyDown}
        />
        <button
          className={`send${busy ? " stop" : ""}`}
          title={busy ? "Stop" : "Send"}
          aria-label={busy ? "Stop" : "Send"}
          onClick={sendOrStop}
        >
          <SendIcon />
        </button>
      </div>
      <div className="hint">Enter to send · Shift+Enter for a newline · / for commands</div>
    </footer>
  );
}
