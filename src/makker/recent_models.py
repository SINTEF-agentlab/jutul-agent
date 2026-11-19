"""Recently-used models (MRU) for the model selector.

A short, most-recent-first list of ``provider:model`` ids, persisted at
``state_home()/recent_models.json``. The selector shows it as a "Recent"
section on top of the discovery-built catalog so the models you actually use
stay within reach without any hand-maintained list.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from makker.paths import state_home

logger = logging.getLogger(__name__)

RECENT_LIMIT = 5


def recent_models_path() -> Path:
    return state_home() / "recent_models.json"


def load_recent_models() -> list[str]:
    """The MRU model ids, most-recent first; empty when absent or unreadable."""
    try:
        data = json.loads(recent_models_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    return [m for m in data if isinstance(m, str) and m][:RECENT_LIMIT]


def record_recent_model(model_id: str) -> None:
    """Move ``model_id`` to the front of the MRU list, capped at ``RECENT_LIMIT``."""
    if not model_id:
        return
    recents = [model_id, *(m for m in load_recent_models() if m != model_id)][:RECENT_LIMIT]
    path = recent_models_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(recents), encoding="utf-8")
    except OSError:
        logger.debug("Could not write recent models to %s", path, exc_info=True)
