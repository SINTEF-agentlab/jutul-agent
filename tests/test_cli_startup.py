"""Guard the CLI cold-start cost: the entry point must stay light.

The agent stack (deepagents, the Anthropic SDK) is ~0.9s to import. Commands that
never build an agent (``--version``, ``--help``, ``doctor``, ``sessions``,
``transcript``) must not pay it. This catches a stray top-level import that would
quietly pull the stack back in. Run in a fresh interpreter so prior tests'
imports don't mask a regression.
"""

from __future__ import annotations

import subprocess
import sys


def test_cli_entry_does_not_import_the_agent_stack():
    code = (
        "import sys, jutul_agent.interfaces.cli.main as m;"
        "heavy=[x for x in ('deepagents', 'anthropic', 'langchain_anthropic') "
        "if x in sys.modules];"
        "print(heavy);"
        "sys.exit(1 if heavy else 0)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, (
        f"the CLI entry now imports the agent stack ({result.stdout.strip()}); "
        "a top-level import is pulling it in. Defer it into the command that needs it."
    )


def test_cli_hides_the_pydantic_v1_warning():
    # The LangChain packages import pydantic.v1, which warns at the start of every
    # command on Python 3.14+. Issuing the warning again after the entry point loads
    # checks the filter on any Python, not only 3.14.
    code = (
        "import warnings, jutul_agent.__main__;"
        "warnings.warn(\"Core Pydantic V1 functionality isn't compatible with "
        'Python 3.14 or greater.", UserWarning)'
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "Pydantic V1" not in result.stderr


def test_cli_hides_the_experimental_stream_warning():
    # The turn loop uses langgraph's v3 event stream on purpose; its "experimental"
    # warning would otherwise print at every turn.
    code = (
        "import warnings, jutul_agent.__main__;"
        'warnings.warn("The v3 streaming protocol on Pregel is experimental.", UserWarning)'
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "experimental" not in result.stderr
