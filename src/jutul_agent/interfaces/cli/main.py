"""Top-level dispatcher for ``jutul-agent``.

Three modes, dispatched on the first argument:

- ``jutul-agent init|setup [--sim <name>]``  bootstrap the current workspace.
- ``jutul-agent doctor``                     diagnose the workspace setup.
- ``jutul-agent upgrade``                    update the install to the latest.
- ``jutul-agent transcript [<id>]``         render a session trace.
- ``jutul-agent sessions``                  list resumable sessions.
- ``jutul-agent eval [<suite>...]``          run bench suites through Inspect.
- ``jutul-agent review [<id>]``              list review findings, or review a session.
- ``jutul-agent skill <add|list>``           manage user-defined skills.
- ``jutul-agent simulator <add|list>``       manage user-defined simulators.
- ``jutul-agent [--sim <name>] [prompt]``   launch the TUI, or run one turn.

The active *workspace* is ``--workspace`` if given, else the current
working directory. State (sessions, traces) lives under ``--state-home``
or its default (``$XDG_DATA_HOME/jutul-agent``).
"""

from __future__ import annotations

import contextlib
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


def main(argv: list[str] | None = None) -> int:
    _force_utf8_streams()
    load_dotenv()
    argv = sys.argv[1:] if argv is None else argv

    if argv and argv[0] in init_cmd.INIT_COMMANDS:
        sub = argv[0]
        args = init_cmd.build_parser(prog=f"jutul-agent {sub}").parse_args(argv[1:])
        apply_workspace_flags(args)
        return init_cmd.run(args)

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

    if argv and argv[0] == "skill":
        from jutul_agent.interfaces.cli import skill as skill_cmd

        return skill_cmd.run(argv[1:])

    if argv and argv[0] == "simulator":
        from jutul_agent.interfaces.cli import simulator as simulator_cmd

        return simulator_cmd.run(argv[1:])

    args = run_cmd.build_parser().parse_args(argv)
    apply_workspace_flags(args)
    return run_cmd.dispatch(args)
