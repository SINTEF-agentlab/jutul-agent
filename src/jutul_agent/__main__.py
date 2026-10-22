"""Entry point for the jutul-agent CLI.

Delegates to ``interfaces.cli.main`` so the CLI plumbing has one home.
"""

from __future__ import annotations

import sys
import warnings

# The LangChain packages import pydantic.v1 so they can accept V1 models, and on
# Python 3.14+ that import alone warns at the start of every command. jutul-agent
# never hands them a V1 model, so the warning says nothing about this program.
warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14",
    category=UserWarning,
)
# The turn loop uses langgraph's v3 event stream on purpose (agent/turns.py), which
# warns that it is experimental at every turn. Nothing a user can act on.
warnings.filterwarnings("ignore", message="The v3 streaming protocol on Pregel is experimental")

from jutul_agent.interfaces.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
