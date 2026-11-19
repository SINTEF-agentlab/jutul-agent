import { act, fireEvent, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { CredentialInfo } from "./api";
import { ApiKeys } from "./components/ApiKeys";
import { renderWithStore } from "./test/util";

const creds = (openaiSet: boolean): CredentialInfo[] => [
  {
    provider: "openai",
    label: "OpenAI",
    env_var: "OPENAI_API_KEY",
    is_set: openaiSet,
    masked: openaiSet ? "sk-***abcd" : null,
    source: openaiSet ? "file" : "none",
    shadowed: false,
  },
  {
    provider: "anthropic",
    label: "Anthropic",
    env_var: "ANTHROPIC_API_KEY",
    is_set: false,
    masked: null,
    source: "none",
    shadowed: false,
  },
];

describe("ApiKeys modal", () => {
  it("is hidden until opened", () => {
    const { container } = renderWithStore(<ApiKeys />);
    expect(container.querySelector(".modal.api-keys")).toBeNull();
  });

  it("shows the required banner and gates Continue until the key is saved", () => {
    const { store } = renderWithStore(<ApiKeys />);
    act(() => {
      store.setState({ credentials: creds(false) });
      store.getState().openApiKeys({
        provider: "openai",
        label: "OpenAI",
        env_var: "OPENAI_API_KEY",
      });
    });
    expect(screen.getByText(/OpenAI needs an API key/)).toBeInTheDocument();
    // A row per provider, with the env var shown.
    expect(screen.getByText("OPENAI_API_KEY")).toBeInTheDocument();
    expect(screen.getByText("ANTHROPIC_API_KEY")).toBeInTheDocument();

    const cont = screen.getByRole("button", { name: "Continue" }) as HTMLButtonElement;
    expect(cont.disabled).toBe(true);

    // Once the required provider's key is present, Continue unlocks.
    act(() => store.setState({ credentials: creds(true) }));
    expect((screen.getByRole("button", { name: "Continue" }) as HTMLButtonElement).disabled).toBe(
      false,
    );
  });

  it("manage mode closes on Done", () => {
    const { store } = renderWithStore(<ApiKeys />);
    act(() => {
      store.setState({ credentials: creds(true) });
      store.getState().openApiKeys(null);
    });
    expect(screen.getByText(/Add or replace a provider key/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Done" }));
    expect(store.getState().apiKeys.open).toBe(false);
  });
});
