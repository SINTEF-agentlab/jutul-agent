import { describe, expect, it, vi } from "vitest";

import {
  HOST_CONTEXT_MESSAGE,
  createHostContextSource,
  decodeHostContext,
  hostContextFromMessage,
  readHostApiUrl,
  readHostContext,
} from "./hostContext";

// The payload is opaque to jutul-agent: whatever the embedding application calls
// its own objects. These stand in for that without borrowing any one
// application's vocabulary.
const SELECTION = {
  primaryId: "11111111-1111-4111-8111-111111111111",
  items: ["22222220-2222-4222-8222-222222222222"],
  groups: ["33333330-3333-4333-8333-333333333333"],
};

/** Base64 of UTF-8 JSON: what a host app's encoder produces. (Plain `btoa` would
 *  encode a non-ASCII string as Latin-1 and not exercise the UTF-8 path at all.) */
const encode = (value: unknown) => {
  const bytes = new TextEncoder().encode(JSON.stringify(value));
  return btoa(String.fromCharCode(...bytes));
};

/** Base64 of plain text, for the API-URL parameter. */
const encode2 = (text: string) => btoa(text);

// Payloads chosen so their base64 actually contains the two characters that make
// this encoding awkward in a URL: "+" (which form decoding turns into a space)
// and "/" (which the url-safe alphabet replaces).
const PLUS_PAYLOAD = { id: "~~~" };
const SLASH_PAYLOAD = { id: "q?" };

describe("decodeHostContext", () => {
  it("decodes the host's base64 JSON", () => {
    expect(decodeHostContext(encode(SELECTION))).toEqual(SELECTION);
  });

  it("reads a space as the '+' form decoding turned it into", () => {
    // "+" is a standard base64 character, and URLSearchParams (form decoding)
    // hands it back as a space. Without this the payload fails to decode at all.
    const raw = encode(PLUS_PAYLOAD);
    expect(raw).toContain("+");
    expect(decodeHostContext(raw.replace(/\+/g, " "))).toEqual(PLUS_PAYLOAD);
  });

  it("accepts the url-safe alphabet and missing padding", () => {
    const urlSafe = (value: unknown) =>
      encode(value).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
    expect(urlSafe(SLASH_PAYLOAD)).toContain("_");
    expect(urlSafe(PLUS_PAYLOAD)).toContain("-");
    expect(decodeHostContext(urlSafe(SLASH_PAYLOAD))).toEqual(SLASH_PAYLOAD);
    expect(decodeHostContext(urlSafe(PLUS_PAYLOAD))).toEqual(PLUS_PAYLOAD);
    expect(decodeHostContext(urlSafe(SELECTION))).toEqual(SELECTION);
  });

  it("decodes non-ASCII as UTF-8 rather than mangling it", () => {
    expect(decodeHostContext(encode({ name: "Bjørn" }))).toEqual({ name: "Bjørn" });
  });

  it("returns null for anything that is not a JSON object", () => {
    expect(decodeHostContext("not base64 at all!")).toBeNull();
    expect(decodeHostContext(btoa("[1,2,3]"))).toBeNull();
    expect(decodeHostContext(btoa("plain text"))).toBeNull();
    expect(decodeHostContext("")).toBeNull();
  });
});

describe("readHostContext", () => {
  it("decodes a full launch URL, percent-encoded padding and all", () => {
    // The shape a host application actually sends: several identifiers, and
    // base64 whose "=" padding arrives percent-encoded from the frame's src.
    const search =
      "?data=eyJwcmltYXJ5SWQiOiIxMTExMTExMS0xMTExLTQxMTEtODExMS0xMTExMTExMTExMTEiLCJpdGVtcyI6WyIyMjIyMj" +
      "IyMC0yMjIyLTQyMjItODIyMi0yMjIyMjIyMjIyMjIiLCIyMjIyMjIyMS0yMjIyLTQyMjItODIyMi0yMjIyMjIyMjIyMjIiLC" +
      "IyMjIyMjIyMi0yMjIyLTQyMjItODIyMi0yMjIyMjIyMjIyMjIiLCIyMjIyMjIyMy0yMjIyLTQyMjItODIyMi0yMjIyMjIyMj" +
      "IyMjIiLCIyMjIyMjIyNC0yMjIyLTQyMjItODIyMi0yMjIyMjIyMjIyMjIiXSwiZ3JvdXBzIjpbIjMzMzMzMzMwLTMzMzMtND" +
      "MzMy04MzMzLTMzMzMzMzMzMzMzMyIsIjMzMzMzMzMxLTMzMzMtNDMzMy04MzMzLTMzMzMzMzMzMzMzMyIsIjMzMzMzMzMyLT" +
      "MzMzMtNDMzMy04MzMzLTMzMzMzMzMzMzMzMyJdfQ%3D%3D";
    const context = readHostContext(search);

    expect(context?.primaryId).toBe("11111111-1111-4111-8111-111111111111");
    expect(context?.items).toHaveLength(5);
    expect(context?.groups).toHaveLength(3);
  });

  it("reads the data parameter, percent-encoding and all", () => {
    const search = `?data=${encodeURIComponent(encode(SELECTION))}&other=1`;
    expect(readHostContext(search)).toEqual(SELECTION);
  });

  it("is null when the UI was opened without one", () => {
    expect(readHostContext("")).toBeNull();
    expect(readHostContext("?sim=jutuldarcy")).toBeNull();
  });
});

describe("readHostApiUrl", () => {
  it("decodes the host API's base URL", () => {
    // The two launch URLs from the host application, whose ports differ per run.
    expect(readHostApiUrl("?apiurl=aHR0cDovLzEyNy4wLjAuMTo1NTkyNA==")).toBe(
      "http://127.0.0.1:55924",
    );
    expect(readHostApiUrl("?apiurl=aHR0cDovLzEyNy4wLjAuMTo0MjU0&data=x")).toBe(
      "http://127.0.0.1:4254",
    );
  });

  it("drops a trailing slash so paths join cleanly", () => {
    expect(readHostApiUrl(`?apiurl=${encode2("http://127.0.0.1:4254/")}`)).toBe(
      "http://127.0.0.1:4254",
    );
  });

  it("refuses anything that is not an http(s) URL", () => {
    expect(readHostApiUrl(`?apiurl=${encode2("file:///etc/passwd")}`)).toBeNull();
    expect(readHostApiUrl(`?apiurl=${encode2("not a url")}`)).toBeNull();
    expect(readHostApiUrl("?apiurl=")).toBeNull();
    expect(readHostApiUrl("")).toBeNull();
  });

  it("never hands back characters that could escape a string literal", () => {
    // `new URL` accepts a quote or a backtick in a path and only percent-encodes
    // it when it serializes, so returning the text as it arrived would let a
    // launch parameter carry code into whatever the address is interpolated into.
    const evil = 'http://127.0.0.1:5000/";run(`id`);x=raw"';
    const decoded = readHostApiUrl(`?apiurl=${encode2(evil)}`);

    expect(decoded).not.toBeNull();
    expect(decoded).not.toContain('"');
    expect(decoded).not.toContain("`");
    expect(decoded).toBe("http://127.0.0.1:5000/%22;run(%60id%60);x=raw%22");
  });

  it("refuses credentials, a query, or a fragment", () => {
    expect(readHostApiUrl(`?apiurl=${encode2("http://user:pw@127.0.0.1:4254")}`)).toBeNull();
    expect(readHostApiUrl(`?apiurl=${encode2("http://127.0.0.1:4254?a=1")}`)).toBeNull();
    expect(readHostApiUrl(`?apiurl=${encode2("http://127.0.0.1:4254#x")}`)).toBeNull();
  });

  it("keeps a path prefix, normalised", () => {
    expect(readHostApiUrl(`?apiurl=${encode2("http://127.0.0.1:4254/app/")}`)).toBe(
      "http://127.0.0.1:4254/app",
    );
  });
});

describe("hostContextFromMessage", () => {
  it("takes a decoded object or the base64 string", () => {
    expect(hostContextFromMessage({ type: HOST_CONTEXT_MESSAGE, context: SELECTION })).toEqual(
      SELECTION,
    );
    expect(hostContextFromMessage({ type: HOST_CONTEXT_MESSAGE, data: encode(SELECTION) })).toEqual(
      SELECTION,
    );
  });

  it("ignores messages that are not ours", () => {
    expect(hostContextFromMessage({ type: "something-else", context: SELECTION })).toBeNull();
    expect(hostContextFromMessage("hello")).toBeNull();
    expect(hostContextFromMessage(null)).toBeNull();
  });
});

/** A window stand-in: a query string, a parent, and message listeners. */
function fakeWindow(search: string) {
  const listeners = new Set<(event: MessageEvent) => void>();
  const win = {
    location: { search },
    parent: { name: "parent" },
    addEventListener: (_type: string, fn: (event: MessageEvent) => void) => void listeners.add(fn),
    removeEventListener: (_type: string, fn: (event: MessageEvent) => void) =>
      void listeners.delete(fn),
  };
  const post = (data: unknown, source: unknown = win.parent) => {
    for (const fn of [...listeners]) fn({ data, source } as MessageEvent);
  };
  return { win: win as unknown as Window, post, listenerCount: () => listeners.size };
}

describe("host context source", () => {
  it("starts from the launch parameter", () => {
    const { win } = fakeWindow(`?data=${encodeURIComponent(encode(SELECTION))}`);
    expect(createHostContextSource(win).current()).toEqual(SELECTION);
  });

  it("adopts an update from the embedding frame and notifies subscribers", () => {
    const { win, post } = fakeWindow("");
    const source = createHostContextSource(win);
    const seen = vi.fn();
    source.subscribe(seen);
    source.listen();

    post({ type: HOST_CONTEXT_MESSAGE, context: SELECTION });

    expect(source.current()).toEqual(SELECTION);
    expect(seen).toHaveBeenCalledWith(SELECTION);
  });

  it("does not notify when the host re-sends the same selection", () => {
    // The host may post its whole selection on every change; an unchanged one
    // must not reach the server, where it would rebuild the agent for nothing.
    const { win, post } = fakeWindow(`?data=${encodeURIComponent(encode(SELECTION))}`);
    const source = createHostContextSource(win);
    const seen = vi.fn();
    source.subscribe(seen);
    source.listen();

    post({ type: HOST_CONTEXT_MESSAGE, context: { ...SELECTION } });

    expect(seen).not.toHaveBeenCalled();
  });

  it("ignores messages from anywhere but the embedding frame", () => {
    const { win, post } = fakeWindow("");
    const source = createHostContextSource(win);
    source.listen();

    post({ type: HOST_CONTEXT_MESSAGE, context: SELECTION }, { name: "someone-else" });

    expect(source.current()).toBeNull();
  });

  it("stops listening when torn down", () => {
    const { win, post, listenerCount } = fakeWindow("");
    const source = createHostContextSource(win);
    const stop = source.listen();
    stop();
    post({ type: HOST_CONTEXT_MESSAGE, context: SELECTION });

    expect(listenerCount()).toBe(0);
    expect(source.current()).toBeNull();
  });
});
