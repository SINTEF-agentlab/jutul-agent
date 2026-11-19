"""Workaround for Gemini thought signatures through Inspect's agent bridge.

Gemini 3 requires the ``thought_signature`` it returned with a function call
to be replayed on subsequent requests. Through the agent bridge the bytes
make a round trip: the ``google.genai`` SDK on the agent side serializes
them as urlsafe base64, the bridge forwards the string as-is, and
Inspect's google provider decodes it with standard ``base64.b64decode``,
which silently drops ``-``/``_`` characters and then fails with "Incorrect
padding". Native (non-bridge) Inspect runs never see urlsafe input, which
is why this only bites bridged agents.

Until that is fixed upstream (UKGovernmentBEIS/inspect_ai), :func:`apply`
swaps the provider module's ``base64`` binding for a shim whose
``b64decode`` accepts both alphabets and missing padding. Decoding strictly
gains tolerance; valid standard base64 decodes exactly as before.

Not fixed here: Gemini's implicit prompt cache never fires through the
bridge (cache reads are zero on every turn, hence ~14x cost on
gemini-3.5-flash). It is **not** the system-prompt serialization — coercing
``system_instruction`` to a canonical string does not help once the system
content is preserved. The bridge dispatches each model turn as two identical
requests, which defeats prefix caching; the real fix is in the bridge/provider
(or explicit Gemini context caching, which Inspect's google provider does not
support). See ``docs/agent-friction-followups.md`` item 12.
"""

from __future__ import annotations

import base64 as _base64
from types import SimpleNamespace
from typing import Any

_URLSAFE_TO_STANDARD = bytes.maketrans(b"-_", b"+/")


def tolerant_b64decode(data: Any, *args: Any, **kwargs: Any) -> bytes:
    """``base64.b64decode`` accepting urlsafe alphabet and unpadded input."""
    if isinstance(data, str):
        data = data.encode("ascii")
    data = bytes(data).translate(_URLSAFE_TO_STANDARD)
    data += b"=" * (-len(data) % 4)
    return _base64.b64decode(data)


def apply() -> None:
    """Patch Inspect's google provider to decode signatures tolerantly.

    Also covers ``inspect_ai._util.images``: a data URI that crossed the
    bridge with urlsafe characters reaches ``file_as_data``, whose standard
    ``b64decode`` silently discards ``-``/``_`` and then fails on the
    resulting length ("data characters cannot be 1 more than a multiple
    of 4"). Same bug, second decode site.
    """
    from inspect_ai._util import images as util_images
    from inspect_ai.model._providers import google as provider

    shim = SimpleNamespace(
        **{name: getattr(_base64, name) for name in dir(_base64) if not name.startswith("_")}
    )
    shim.b64decode = tolerant_b64decode
    for module in (provider, util_images):
        if getattr(module, "_jutul_tolerant_b64", False):
            continue
        module.base64 = shim
        module._jutul_tolerant_b64 = True
