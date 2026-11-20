// Assistant prose and server-rendered panels, as Markdown via react-markdown. It
// never renders raw HTML, and rehype-sanitize is layered on as defence in depth, so
// model/tool output cannot inject markup. GFM adds tables and strikethrough.

import { useRef, type ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import rehypeSanitize from "rehype-sanitize";
import remarkGfm from "remark-gfm";

function CopyButton({ targetRef }: { targetRef: React.RefObject<HTMLElement | null> }) {
  const btn = useRef<HTMLButtonElement>(null);
  return (
    <button
      ref={btn}
      className="copy-btn"
      type="button"
      onClick={(e) => {
        e.stopPropagation();
        const code = targetRef.current?.querySelector("code") ?? targetRef.current;
        navigator.clipboard.writeText(code?.textContent ?? "").then(() => {
          const el = btn.current;
          if (!el) return;
          el.textContent = "Copied";
          setTimeout(() => (el.textContent = "Copy"), 1200);
        });
      }}
    >
      Copy
    </button>
  );
}

function Pre({ children }: { children?: ReactNode }) {
  const ref = useRef<HTMLPreElement>(null);
  return (
    <pre ref={ref} style={{ position: "relative" }}>
      {children}
      <CopyButton targetRef={ref} />
    </pre>
  );
}

export function Markdown({ text, className = "markdown" }: { text: string; className?: string }) {
  return (
    <div className={className}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeSanitize]}
        components={{
          a: ({ href, children }) => (
            <a href={href} target="_blank" rel="noopener noreferrer">
              {children}
            </a>
          ),
          pre: ({ children }) => <Pre>{children}</Pre>,
        }}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
}
