"""Capabilities: the layers an agent is composed from.

The agent for a session is built from layers: the base tools and skills, the
active simulator, the front end (surface), and whatever a host application adds.
Each layer is a ``Capability`` that can contribute tools, skills, subagents, and
a fragment of the system prompt. ``build_agent`` collects the contributions of
every capability that applies to the current surface.

A capability reaches the agent three ways: passed in directly, discovered from
an installed package's entry points (``discover_extensions``), or built from a
host application's declarative tool specs (``http_tool_capability``). The first
two carry in-process Python tools; the third lets an application in any language
expose its own routines over HTTP.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

    from jutul_agent.session import Session

# A tool factory takes the live session (some tools need it, some ignore it) and
# returns the tool to register.
ToolFactory = Callable[["Session"], "BaseTool"]
SubagentFactory = Callable[["Session"], dict[str, Any]]


@dataclass(frozen=True)
class Capability:
    """One layer's contributions to a session's agent.

    ``surfaces`` restricts the capability to certain front ends (``"tui"``,
    ``"web"``, ``"cli"``); empty means it applies to every surface. ``skill_dirs``
    are ``(path, label)`` pairs in the form the skills middleware expects.
    ``dependencies`` are paths to local Julia package roots or their
    ``Project.toml`` files; the launcher reads the package metadata from that
    path and adds it to the active Julia env before launch.
    """

    name: str
    tools: tuple[ToolFactory, ...] = ()
    skill_dirs: tuple[tuple[str, str], ...] = ()
    dependencies: tuple[Path, ...] = ()
    subagents: tuple[SubagentFactory, ...] = ()
    prompt_fragment: str = ""
    ui_actions: tuple[str, ...] = ()
    surfaces: tuple[str, ...] = ()


def select_for_surface(capabilities: Sequence[Capability], surface: str) -> list[Capability]:
    """The capabilities that apply to ``surface``: unrestricted ones plus those that name it."""
    return [cap for cap in capabilities if not cap.surfaces or surface in cap.surfaces]


def collect_tools(capabilities: Sequence[Capability], session: Session) -> list[BaseTool]:
    return [factory(session) for cap in capabilities for factory in cap.tools]


def collect_skill_dirs(capabilities: Sequence[Capability]) -> list[tuple[str, str]]:
    return [pair for cap in capabilities for pair in cap.skill_dirs]


def collect_dependency_paths(capabilities: Sequence[Capability]) -> list[Path]:
    return [Path(path) for cap in capabilities for path in cap.dependencies]


def collect_subagents(capabilities: Sequence[Capability], session: Session) -> list[dict[str, Any]]:
    return [factory(session) for cap in capabilities for factory in cap.subagents]


def collect_prompt_fragments(capabilities: Sequence[Capability]) -> list[str]:
    return [cap.prompt_fragment for cap in capabilities if cap.prompt_fragment.strip()]


# Entry-point group an installed package publishes a Capability under.
EXTENSION_ENTRY_POINT_GROUP = "jutul_agent.extensions"


def discover_extensions() -> list[Capability]:
    """Capabilities published by installed packages under the extension entry point.

    Each entry point resolves to a ``Capability`` or a zero-argument callable
    that returns one. A broken entry point is warned about and skipped rather
    than failing the whole session: one bad extension must not cost the user
    every other tool. The warning matters as much as the skip, because a
    capability that fails to load is otherwise indistinguishable from one that
    is not installed, and the usual cause is a setup step the user can fix
    (an unset environment variable, a missing data folder).
    """
    import importlib.metadata as importlib_metadata

    capabilities: list[Capability] = []
    try:
        entry_points = importlib_metadata.entry_points(group=EXTENSION_ENTRY_POINT_GROUP)
    except Exception as exc:
        print(f"warning: could not read installed extensions: {exc}", file=sys.stderr)
        return capabilities
    for entry_point in entry_points:
        try:
            loaded = entry_point.load()
            capability = (
                loaded() if not isinstance(loaded, Capability) and callable(loaded) else loaded
            )
        except Exception as exc:
            print(
                f"warning: skipping extension {entry_point.name!r}: {exc}",
                file=sys.stderr,
            )
            continue
        if not isinstance(capability, Capability):
            print(
                f"warning: skipping extension {entry_point.name!r}: it resolved to "
                f"{type(capability).__name__}, not a Capability",
                file=sys.stderr,
            )
            continue
        capabilities.append(capability)
    return capabilities


# ---------------------------------------------------------------------------
# Declarative HTTP tools: turn an application's tool specs into agent tools.


@dataclass(frozen=True)
class HttpToolSpec:
    """A host application's operation, exposed to the agent as an HTTP-backed tool."""

    name: str
    description: str
    endpoint: str
    # JSON-schema-like parameter map: ``{name: {"type": "...", "description": "...",
    # "required": bool, "default": ...}}``.
    parameters: dict[str, dict[str, Any]] = field(default_factory=dict)


_PY_TYPES: dict[str, type] = {
    "string": str,
    "number": float,
    "integer": int,
    "boolean": bool,
    "object": dict,
    "array": list,
}


def http_tool_capability(
    name: str,
    specs: Sequence[HttpToolSpec],
    *,
    client: Any | None = None,
    surfaces: tuple[str, ...] = (),
) -> Capability:
    """Build a capability whose tools call a host application's HTTP endpoints.

    ``client`` is an optional ``httpx.AsyncClient`` (or anything with a matching
    async ``post``); when omitted, each call opens and closes its own client.
    This is how an application written in any language gives the agent its
    routines: it sends the specs, and the agent gets tools that POST to them.
    """
    tools = tuple(_http_tool_factory(spec, client) for spec in specs)
    return Capability(name=name, tools=tools, surfaces=surfaces)


def _http_tool_factory(spec: HttpToolSpec, client: Any | None) -> ToolFactory:
    tool = _build_http_tool(spec, client)
    return lambda _session: tool


def _build_http_tool(spec: HttpToolSpec, client: Any | None) -> BaseTool:
    from langchain_core.tools import StructuredTool

    args_model = _args_model(spec.name, spec.parameters)

    async def _call(**kwargs: Any) -> str:
        import httpx

        owned = client is None
        http = client or httpx.AsyncClient()
        try:
            response = await http.post(spec.endpoint, json=kwargs)
            response.raise_for_status()
            return response.text
        except httpx.HTTPStatusError as exc:
            # Return the failure as the tool result so the model can recover (retry,
            # adjust, or explain) instead of aborting the whole turn.
            body = exc.response.text[:500]
            return f"The '{spec.name}' endpoint returned HTTP {exc.response.status_code}: {body}"
        except httpx.HTTPError as exc:
            return f"The '{spec.name}' endpoint could not be reached: {exc}"
        finally:
            if owned:
                await http.aclose()

    return StructuredTool.from_function(
        coroutine=_call,
        name=spec.name,
        description=spec.description,
        args_schema=args_model,
    )


def _args_model(tool_name: str, parameters: dict[str, dict[str, Any]]) -> type:
    from pydantic import Field, create_model

    fields: dict[str, Any] = {}
    for param_name, param in parameters.items():
        param = param or {}
        py_type = _PY_TYPES.get(str(param.get("type")), str)
        description = str(param.get("description") or "")
        if param.get("required", True):
            fields[param_name] = (py_type, Field(..., description=description))
        else:
            fields[param_name] = (
                py_type | None,
                Field(param.get("default"), description=description),
            )
    return create_model(f"{tool_name}_args", **fields)
