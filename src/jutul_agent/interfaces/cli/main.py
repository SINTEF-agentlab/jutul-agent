"""Top-level dispatcher for ``jutul-agent``.

Dispatched on the first argument. With no command, ``jutul-agent`` starts the
web UI, the normal way to work:

- ``jutul-agent [--sim <name>]``             web UI (HTTP + WebSocket server);
  ``jutul-agent web`` is the same command.
- ``jutul-agent tui [--sim <name>]``         terminal UI.
- ``jutul-agent run "<prompt>" [--sim …]``   one headless turn, then print the result.

Setup and utilities:

- ``jutul-agent init|setup [--sim <name>]``  bootstrap the current workspace.
- ``jutul-agent key [<provider>]``           view or set provider API keys.
- ``jutul-agent sysimage [build|status]``    manage the folder's Julia system image.
- ``jutul-agent doctor``                     diagnose the workspace setup.
- ``jutul-agent upgrade``                    update the install to the latest.
- ``jutul-agent transcript [<id>]``         render a session trace.
- ``jutul-agent sessions``                  list resumable sessions.
- ``jutul-agent eval [<suite>...]``          run bench suites through Inspect.
- ``jutul-agent review [<id>]``              list review findings, or review a session.

The active *workspace* is ``--workspace`` if given, else the current working
directory. State (sessions, traces) lives under ``--state-home`` or its default
(``$XDG_DATA_HOME/jutul-agent``).
"""

from __future__ import annotations

import contextlib
import difflib
import sys

from dotenv import load_dotenv

from jutul_agent.interfaces.cli import doctor as doctor_cmd
from jutul_agent.interfaces.cli import init as init_cmd
from jutul_agent.interfaces.cli import run as run_cmd
from jutul_agent.interfaces.cli import transcript as transcript_cmd
from jutul_agent.interfaces.cli._helpers import apply_workspace_flags


def _force_utf8_streams() -> None:
    """Make stdout/stderr UTF-8 so non-ASCII output can't crash the CLI.

    Windows consoles default to a legacy code page (e.g. cp1252); printing a
    character it can't encode raises ``UnicodeEncodeError`` and takes down the
    whole command. Reconfiguring to UTF-8 (with replacement) keeps output
    legible and the process alive regardless of the console's code page.
    """

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            with contextlib.suppress(ValueError, OSError):
                reconfigure(encoding="utf-8", errors="replace")


HELP = """usage: jutul-agent [<command>] [options]

With no command, jutul-agent starts the web UI for the current folder: chat
with interactive plots and reports, in the browser at http://127.0.0.1:8742.

  jutul-agent                  Web UI (`jutul-agent web` is the same).
  jutul-agent tui              Terminal UI.
  jutul-agent run "<prompt>"   Run a single turn headlessly and print the result.

The web UI's options (--sim, --port, --model, ...) work with or without `web`;
`jutul-agent web -h` lists them.

Set up a folder first:  jutul-agent init --sim <name>
Set an API key:         jutul-agent key <provider>
More commands:          doctor, sysimage, upgrade, transcript, sessions, review, eval
                        (add -h to any of them for options)
"""

COMMANDS = (
    "web",
    "tui",
    "run",
    *sorted(init_cmd.INIT_COMMANDS),
    "key",
    "keys",
    "auth",
    "sysimage",
    "doctor",
    "upgrade",
    "transcript",
    "sessions",
    "eval",
    "review",
)


def main(argv: list[str] | None = None) -> int:
    _force_utf8_streams()
    load_dotenv()
    argv = sys.argv[1:] if argv is None else argv

    if argv and argv[0] in ("-h", "--help"):
        print(HELP)
        return 0

    if argv and argv[0] in ("--version", "-V"):
        from jutul_agent import __version__

        print(f"jutul-agent {__version__}")
        raise SystemExit(0)

    # No command starts the web UI, and so does a leading option: `jutul-agent
    # --port 8800` is `jutul-agent web --port 8800`. Its parser rejects anything it
    # doesn't know, so a stray flag fails instead of starting a server.
    if not argv or argv[0] == "web" or argv[0].startswith("-"):
        from jutul_agent.interfaces.cli import web as web_cmd

        named = argv[:1] == ["web"]
        parser = web_cmd.build_parser(prog="jutul-agent web" if named else "jutul-agent")
        args = parser.parse_args(argv[1:] if named else argv)
        return web_cmd.run(args)  # applies the workspace flags itself

    if argv[0] == "tui":
        args = run_cmd.build_parser(prog="jutul-agent tui").parse_args(argv[1:])
        if args.prompt is not None:
            print(
                "error: `tui` is interactive and takes no prompt. For one turn, use "
                '`jutul-agent run "<prompt>"`.',
                file=sys.stderr,
            )
            return 2
        apply_workspace_flags(args)
        return run_cmd.dispatch(args)

    if argv[0] == "run":
        args = run_cmd.build_parser(prog="jutul-agent run").parse_args(argv[1:])
        if not args.prompt:
            print(
                'error: `run` needs a prompt, e.g. `jutul-agent run "compute mean([1,2,3])"`. '
                "For an interactive session, run `jutul-agent` (web UI) or `jutul-agent tui`.",
                file=sys.stderr,
            )
            return 2
        apply_workspace_flags(args)
        return run_cmd.dispatch(args)

    if argv and argv[0] in init_cmd.INIT_COMMANDS:
        sub = argv[0]
        args = init_cmd.build_parser(prog=f"jutul-agent {sub}").parse_args(argv[1:])
        apply_workspace_flags(args)
        return init_cmd.run(args)

    if argv and argv[0] in ("key", "keys", "auth"):
        from jutul_agent.interfaces.cli import key as key_cmd

        args = key_cmd.build_parser(prog=f"jutul-agent {argv[0]}").parse_args(argv[1:])
        return key_cmd.run(args)

    if argv and argv[0] == "sysimage":
        from jutul_agent.interfaces.cli import sysimage as sysimage_cmd

        args = sysimage_cmd.build_parser().parse_args(argv[1:])
        apply_workspace_flags(args)
        return sysimage_cmd.run(args)

    if argv and argv[0] == "doctor":
        args = doctor_cmd.build_parser().parse_args(argv[1:])
        apply_workspace_flags(args)
        return doctor_cmd.run(args)

    if argv and argv[0] == "upgrade":
        from jutul_agent.interfaces.cli import upgrade as upgrade_cmd

        args = upgrade_cmd.build_parser().parse_args(argv[1:])
        apply_workspace_flags(args)
        return upgrade_cmd.run(args)

    if argv and argv[0] == "transcript":
        args = transcript_cmd.build_parser().parse_args(argv[1:])
        apply_workspace_flags(args)
        return transcript_cmd.run(args)

    if argv and argv[0] == "sessions":
        from jutul_agent.interfaces.cli import sessions as sessions_cmd

        args = sessions_cmd.build_parser().parse_args(argv[1:])
        apply_workspace_flags(args)
        return sessions_cmd.run(args)

    if argv and argv[0] == "eval":
        from jutul_agent.interfaces.cli import eval as eval_cmd

        args = eval_cmd.build_parser().parse_args(argv[1:])
        return eval_cmd.run(args)

    if argv and argv[0] == "review":
        from jutul_agent.interfaces.cli import review as review_cmd

        args = review_cmd.build_parser().parse_args(argv[1:])
        apply_workspace_flags(args)
        return review_cmd.run(args)

    # No known command matched. Don't start the web UI on a typo or a bare prompt;
    # name the choices instead.
    close = difflib.get_close_matches(argv[0], COMMANDS, n=1)
    hint = f" Did you mean `jutul-agent {close[0]}`?" if close else ""
    print(f"error: unknown command: {argv[0]!r}.{hint}\n", file=sys.stderr)
    print(HELP, file=sys.stderr)
    return 2
