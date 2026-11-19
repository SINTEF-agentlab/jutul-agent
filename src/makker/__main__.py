"""Entry point for the makker CLI.

Delegates to ``interfaces.cli.main`` so the CLI plumbing has one home.
"""

from __future__ import annotations

import sys
import warnings

# The LangChain packages import pydantic.v1 so they can accept V1 models, and on
# Python 3.14+ that import alone warns at the start of every command. makker
# never hands them a V1 model, so the warning says nothing about this program.
warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14",
    category=UserWarning,
)

from makker.interfaces.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
