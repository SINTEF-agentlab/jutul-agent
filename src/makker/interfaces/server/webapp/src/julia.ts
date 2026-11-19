// A small, dependency-free Julia tokenizer for code shown in tool cards. It splits
// comments/strings/macros/numbers/words so nothing is highlighted inside a string
// or comment; good enough for display (not a real parser). Returns tokens the React
// layer renders as styled spans — no HTML, so no escaping concerns.

const JULIA_KEYWORDS = new Set([
  "function", "end", "if", "else", "elseif", "for", "while", "do", "return", "break",
  "continue", "using", "import", "export", "struct", "mutable", "abstract", "primitive",
  "const", "global", "local", "let", "begin", "module", "macro", "quote", "try", "catch",
  "finally", "where", "in", "isa", "true", "false", "nothing", "missing",
]);

/** A token class maps to a `.jl-*` CSS class; `undefined` means plain text. */
export type JuliaClass = "jl-com" | "jl-str" | "jl-mac" | "jl-num" | "jl-kw" | "jl-type";

export interface JuliaToken {
  text: string;
  cls?: JuliaClass;
}

const TOKEN =
  /(#=[\s\S]*?=#|#[^\n]*)|("""[\s\S]*?"""|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])')|(@[A-Za-z_]\w*)|(\b\d+\.?\d*(?:[eE][+-]?\d+)?\b)|([A-Za-z_]\w*!?)/g;

export function tokenizeJulia(code: string): JuliaToken[] {
  const tokens: JuliaToken[] = [];
  let last = 0;
  let m: RegExpExecArray | null;
  TOKEN.lastIndex = 0;
  const plain = (text: string) => {
    if (text) tokens.push({ text });
  };
  while ((m = TOKEN.exec(code))) {
    plain(code.slice(last, m.index));
    last = TOKEN.lastIndex;
    if (m[1]) tokens.push({ text: m[1], cls: "jl-com" });
    else if (m[2]) tokens.push({ text: m[2], cls: "jl-str" });
    else if (m[3]) tokens.push({ text: m[3], cls: "jl-mac" });
    else if (m[4]) tokens.push({ text: m[4], cls: "jl-num" });
    else if (JULIA_KEYWORDS.has(m[5])) tokens.push({ text: m[5], cls: "jl-kw" });
    else if (/^[A-Z]/.test(m[5])) tokens.push({ text: m[5], cls: "jl-type" });
    else plain(m[5]);
  }
  plain(code.slice(last));
  return tokens;
}
