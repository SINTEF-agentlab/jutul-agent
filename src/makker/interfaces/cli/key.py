"""``makker key`` subcommand: view and set provider API keys.

A pip-installed user has no checkout to edit, so this is the supported way to
add, change, or inspect the keys makker saves in the global ``.env``. It
needs no workspace and no Julia. The model selector (TUI and web) prompts for a
missing key on its own; this command is the explicit, scriptable path and the
only way to *replace* a key that is already set.
"""

from __future__ import annotations

import argparse
import getpass
import sys

from makker.credentials import (
    key_status,
    provider_by_name,
    store_credential_for_provider,
    user_env_path,
)


def build_parser(prog: str = "makker key") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="View, add, or replace the provider API keys makker saves.",
    )
    parser.add_argument(
        "provider",
        nargs="?",
        help="Provider to set a key for (e.g. openai, anthropic, google). "
        "Omit to list the current status.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Only print the current key status; never prompt.",
    )
    return parser


def _print_status() -> None:
    """List each provider, whether its key is set, and where it comes from."""
    print(f"Keys are saved in {user_env_path()}\n")
    for st in key_status():
        if not st.is_set:
            mark, detail = "✗", "not set"
        elif st.source == "environment":
            mark, detail = "•", f"set in your shell or a project .env ({st.masked})"
        elif st.shadowed:
            mark, detail = "!", f"saved key is overridden by your shell/.env ({st.masked})"
        else:
            mark, detail = "✓", f"saved ({st.masked})"
        print(f"  {mark} {st.label:<10} {st.env_var:<20} {detail}")
        if st.shadowed:
            print(
                f"      note: {st.env_var} is also set in your environment, which wins. "
                "Clear it there for the saved key to take effect."
            )
    print("\nSet or replace a key with: makker key <provider>   (e.g. makker key openai)")


def _set_key(provider: str) -> int:
    """Prompt for and save a provider's key. Returns a process exit code."""
    info = provider_by_name(provider)
    if info is None or info.key_env_var is None:
        from makker.credentials import key_providers

        names = ", ".join(p.name for p in key_providers())
        print(f"error: unknown provider {provider!r}. Known: {names}.", file=sys.stderr)
        return 2
    if not sys.stdin.isatty():
        print(
            f"error: setting a key needs an interactive terminal. Set {info.key_env_var} "
            "in your shell or a .env instead.",
            file=sys.stderr,
        )
        return 2
    print(f"Enter the {info.label} API key ({info.key_env_var}). Input is hidden.")
    try:
        value = getpass.getpass(f"{info.key_env_var}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return 1
    if not value:
        print("No key entered; nothing changed.")
        return 0
    _, path = store_credential_for_provider(info.name, value)
    print(f"Saved {info.key_env_var} to {path}.")
    return 0


def run(args: argparse.Namespace) -> int:
    from makker.credentials import load_user_credentials

    # Reflect the saved global .env in the environment, so status matches what a
    # real session would resolve (the top-level CLI only loads a project .env).
    load_user_credentials()

    if args.provider:
        return _set_key(args.provider)

    _print_status()
    if args.show or not sys.stdin.isatty():
        return 0

    # Interactive convenience: offer to set a key right away so a user who ran a
    # bare `makker key` to check status can fix a missing one in place.
    try:
        choice = input("\nProvider to set a key for (blank to skip): ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return 0
    if not choice:
        return 0
    return _set_key(choice)
