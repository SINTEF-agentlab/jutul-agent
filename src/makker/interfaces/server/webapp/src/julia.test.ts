import { describe, expect, it } from "vitest";

import { tokenizeJulia } from "./julia";

const classOf = (code: string, text: string) =>
  tokenizeJulia(code).find((t) => t.text === text)?.cls;

describe("tokenizeJulia", () => {
  it("classifies keywords, types, macros, numbers, strings, comments", () => {
    expect(classOf("function f()", "function")).toBe("jl-kw");
    expect(classOf("x = Foo()", "Foo")).toBe("jl-type");
    expect(classOf("@time run()", "@time")).toBe("jl-mac");
    expect(classOf("n = 42", "42")).toBe("jl-num");
    expect(classOf('s = "hello"', '"hello"')).toBe("jl-str");
    expect(classOf("# a note", "# a note")).toBe("jl-com");
  });

  it("does not highlight keywords inside strings or comments", () => {
    const inString = tokenizeJulia('"function end"');
    expect(inString.some((t) => t.cls === "jl-kw")).toBe(false);
    expect(inString).toEqual([{ text: '"function end"', cls: "jl-str" }]);
  });

  it("round-trips the source exactly", () => {
    const code = 'using Foo\n@time x = bar(1, "y") # go';
    expect(tokenizeJulia(code).map((t) => t.text).join("")).toBe(code);
  });
});
