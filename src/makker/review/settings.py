"""When session review runs and which model it uses.

A developer tool, toggled by environment so it never touches the user-facing
config schema. Off by default; when on, a review runs automatically after every
completed turn. The reviewer model is deliberately separate from (and usually more
capable than) the agent model, since judging plausibility is the harder task.
"""

from __future__ import annotations

import os

REVIEW_ENABLED_ENV = "MAKKER_REVIEW"
REVIEW_MODEL_ENV = "MAKKER_REVIEW_MODEL"

# A capable default; override with MAKKER_REVIEW_MODEL (e.g. a mini model for
# quick local iteration).
DEFAULT_REVIEW_MODEL = "openai:gpt-5.4"

_TRUE = {"1", "true", "yes", "on"}


def review_enabled() -> bool:
    return os.environ.get(REVIEW_ENABLED_ENV, "").strip().lower() in _TRUE


def review_model() -> str:
    return os.environ.get(REVIEW_MODEL_ENV, "").strip() or DEFAULT_REVIEW_MODEL
