import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { Markdown } from "./markdown";

describe("Markdown", () => {
  it("renders basic markdown", () => {
    const { container, getByText } = render(<Markdown text="**bold** and `code`" />);
    expect(container.querySelector("strong")?.textContent).toBe("bold");
    expect(container.querySelector("code")?.textContent).toBe("code");
    expect(getByText("bold")).toBeInTheDocument();
  });

  it("renders GFM tables", () => {
    const md = "| a | b |\n| - | - |\n| 1 | 2 |";
    const { container } = render(<Markdown text={md} />);
    expect(container.querySelector("table")).toBeInTheDocument();
    expect(container.querySelectorAll("td")).toHaveLength(2);
  });

  it("does not render raw HTML from model/tool output (no XSS)", () => {
    const evil = 'hi <script>alert(1)</script> <img src=x onerror="alert(1)">';
    const { container } = render(<Markdown text={evil} />);
    expect(container.querySelector("script")).toBeNull();
    expect(container.querySelector("img")).toBeNull();
    expect(container.textContent).toContain("hi");
  });

  it("opens links in a new tab safely", () => {
    const { container } = render(<Markdown text="[site](https://example.com)" />);
    const a = container.querySelector("a");
    expect(a?.getAttribute("href")).toBe("https://example.com");
    expect(a?.getAttribute("target")).toBe("_blank");
    expect(a?.getAttribute("rel")).toContain("noopener");
  });
});
