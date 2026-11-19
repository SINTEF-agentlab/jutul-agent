// The app shell: creates the store + controller, runs startup once, and lays out
// the three panes (sidebar · conversation · canvas). Drag-a-file anywhere over the
// conversation uploads it into the session.

import { useEffect, useMemo, useRef, useState } from "react";

import { ApiKeys } from "./components/ApiKeys";
import { Canvas } from "./components/Canvas";
import { Composer } from "./components/Composer";
import { Sidebar } from "./components/Sidebar";
import { Thread } from "./components/Thread";
import { Topbar } from "./components/Topbar";
import { UpdateDialog } from "./components/UpdateDialog";
import { Controller } from "./controller";
import { SessionProvider, useSel } from "./context";
import { installDebug } from "./debug";
import { createSessionStore } from "./store";

function ReconnectingBar() {
  const reconnecting = useSel((s) => s.reconnecting);
  if (!reconnecting) return null;
  return (
    <div className="reconnecting" role="status">
      <span className="spinner" /> Reconnecting…
    </div>
  );
}

export function App() {
  const ctx = useMemo(() => {
    const store = createSessionStore();
    return { store, controller: new Controller(store) };
  }, []);

  const started = useRef(false);
  useEffect(() => {
    if (started.current) return; // guard against a double-invoke in dev
    started.current = true;
    installDebug(ctx.store, ctx.controller);
    void ctx.controller.init();
    return () => ctx.controller.transport.close();
  }, [ctx]);

  const [collapsed, setCollapsed] = useState(() => {
    const v = localStorage.getItem("ja_sidebar");
    return v === "collapsed" || (v === null && window.innerWidth <= 760);
  });
  const toggleSidebar = () =>
    setCollapsed((c) => {
      const next = !c;
      localStorage.setItem("ja_sidebar", next ? "collapsed" : "open");
      return next;
    });

  const [dropping, setDropping] = useState(false);
  const dragDepth = useRef(0);

  return (
    <SessionProvider value={ctx}>
      <div className="workspace">
        <Sidebar collapsed={collapsed} />
        <div
          className={`app${dropping ? " dropping" : ""}`}
          onDragEnter={(e) => {
            if (e.dataTransfer && Array.from(e.dataTransfer.types).includes("Files")) {
              dragDepth.current++;
              setDropping(true);
            }
          }}
          onDragOver={(e) => {
            if (dropping) e.preventDefault();
          }}
          onDragLeave={() => {
            if (--dragDepth.current <= 0) {
              dragDepth.current = 0;
              setDropping(false);
            }
          }}
          onDrop={(e) => {
            e.preventDefault();
            dragDepth.current = 0;
            setDropping(false);
            for (const f of Array.from(e.dataTransfer.files)) void ctx.controller.upload(f);
          }}
        >
          <Topbar onToggleSidebar={toggleSidebar} />
          <ReconnectingBar />
          <Thread />
          <Composer />
        </div>
        <Canvas />
        <ApiKeys />
        <UpdateDialog />
      </div>
    </SessionProvider>
  );
}
