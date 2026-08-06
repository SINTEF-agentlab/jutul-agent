// The conversation pane: the welcome screen when empty, otherwise the ordered
// thread items, the "thinking" indicator, and any pending approval. Keeps the view
// pinned to the bottom while streaming unless the user has scrolled up to read.

import { memo, useLayoutEffect, useRef } from "react";

import { useController, useSel } from "../context";
import type { ThreadItem } from "../store";
import { Approval } from "./Approval";
import {
  ArtifactFileCard,
  ArtifactImageCard,
  AssistantMessage,
  ContextCard,
  ErrorCard,
  HelpCard,
  ReasoningBlock,
  SysNote,
  ToolCard,
  UiNote,
  UserBubble,
  VizChip,
} from "./messages";

const DEFAULT_EXAMPLES = [
  "Set up a small simulation and show me the interactive result.",
  "Plot the results from the run.",
  "Give me a quick tour of what this simulator can do.",
];

function Welcome() {
  const sim = useSel((s) => s.sim);
  const details = useSel((s) => s.simDetails);
  const busy = useSel((s) => s.busy);
  const controller = useController();
  const d = sim ? details[sim] : undefined;
  const display = d?.display_name || sim;
  const prompts = d?.examples && d.examples.length ? d.examples : DEFAULT_EXAMPLES;
  return (
    <div className="welcome">
      <h1>{display ? `What would you like to explore with ${display}?` : "What would you like to explore?"}</h1>
      <p>
        Ask a question or describe a task. The agent runs the simulator, writes and runs Julia, and
        shows results here.
      </p>
      <div className="examples">
        {prompts.map((t) => (
          <button key={t} className="example" disabled={busy} onClick={() => controller.send(t)}>
            {t}
          </button>
        ))}
      </div>
    </div>
  );
}

const ThreadItemView = memo(function ThreadItemView({ item }: { item: ThreadItem }) {
  switch (item.kind) {
    case "user":
      return <UserBubble text={item.text} />;
    case "assistant":
      return <AssistantMessage text={item.text} />;
    case "reasoning":
      return <ReasoningBlock text={item.text} live={item.live} />;
    case "tool":
      return <ToolCard item={item} />;
    case "viz-chip":
      return <VizChip viewId={item.viewId} title={item.title} viewKind={item.viewKind} />;
    case "artifact-image":
      return <ArtifactImageCard url={item.url} title={item.title} />;
    case "artifact-file":
      return <ArtifactFileCard url={item.url} caption={item.caption} />;
    case "sys-note":
      return <SysNote text={item.text} level={item.level} />;
    case "ui-note":
      return <UiNote action={item.action} payload={item.payload} />;
    case "error":
      return <ErrorCard message={item.message} canRetry={item.canRetry} />;
    case "help":
      return <HelpCard />;
    case "context":
      return <ContextCard markdown={item.markdown} />;
  }
});

function Working() {
  return (
    <div className="working">
      <span />
      <span />
      <span />
    </div>
  );
}

export function Thread() {
  const items = useSel((s) => s.items);
  const working = useSel((s) => s.working);
  const pending = useSel((s) => s.pending);
  const bottomPin = useSel((s) => s.bottomPin);

  const scroller = useRef<HTMLDivElement>(null);
  const stick = useRef(true);

  const onScroll = () => {
    const el = scroller.current;
    if (el) stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 120;
  };
  // After any content change, keep the view pinned to the bottom if it already was.
  useLayoutEffect(() => {
    const el = scroller.current;
    if (el && stick.current) el.scrollTop = el.scrollHeight;
  }, [items, working, pending]);
  // Force the view to the bottom when the user sends a message, even if they had
  // scrolled up, so their prompt and the incoming reply are visible.
  useLayoutEffect(() => {
    const el = scroller.current;
    if (!el) return;
    stick.current = true;
    el.scrollTop = el.scrollHeight;
  }, [bottomPin]);

  return (
    <main className="conversation" ref={scroller} onScroll={onScroll}>
      <div className="thread">
        {items.length === 0 ? (
          <Welcome />
        ) : (
          items.map((item) => <ThreadItemView key={item.id} item={item} />)
        )}
        {working ? <Working /> : null}
        {pending ? <Approval /> : null}
      </div>
    </main>
  );
}
