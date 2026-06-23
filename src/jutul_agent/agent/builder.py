"""Construct the Deep Agents runtime for a Session.

This module wires everything ``create_deep_agent`` needs in one place:

- ``build_backend``: the real-path CompositeBackend over the workspace
  (skills, memory, package source, and added folders all at their real paths).
- ``register_provider_profiles``: ``HarnessProfile`` registration per
  provider (disables the default general-purpose subagent; all prompt text
  lives in ``agent.prompts``).
- ``build_agent``: the entry point used by the CLI/TUI. Returns the agent
  together with its live backend, so callers can add extra folders
  mid-session (``/add-dir``).
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deepagents import (
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    create_deep_agent,
    register_harness_profile,
)
from deepagents.backends import CompositeBackend

from jutul_agent.agent.added_dirs import add_dir
from jutul_agent.agent.approval import ApprovalMode, interrupt_on_for_mode, parse_approval_mode
from jutul_agent.agent.backend import RecursiveGrepBackend, WorkspaceShellBackend
from jutul_agent.agent.context_editing import build_context_editing_middleware
from jutul_agent.agent.memory import (
    build_memory_middleware,
    ensure_memory_dir,
    make_remember_tool,
)
from jutul_agent.agent.plot_julia import (
    make_close_plots_tool,
    make_plot_julia_tool,
    make_recapture_tool,
)
from jutul_agent.agent.prompts import assemble_session_prompt
from jutul_agent.agent.recovery import InvalidToolCallRecoveryMiddleware
from jutul_agent.agent.tools import (
    make_record_attempt_tool,
    make_reset_julia_tool,
    make_run_julia_tool,
    make_write_report_tool,
)
from jutul_agent.agent.windows_paths import enable_windows_real_paths
from jutul_agent.models import PROVIDERS, provider_of
from jutul_agent.paths import (
    SHARED_SKILLS_DIR,
    user_simulators_dir,
    user_skills_dir,
    workspace_memory_dir,
    workspace_root,
)
from jutul_agent.session import Session
from jutul_agent.simulators.base import SimulatorAdapter
from jutul_agent.trace import TraceRecorder

__all__ = ["PackageSource", "build_agent", "build_backend", "resolve_model"]

# Re-exported from the light models module so importing it for these constants does
# not pull the agent stack; kept here for the existing import sites.
from jutul_agent.models import DEFAULT_MODEL, MODEL_ENV_VAR


@dataclass(frozen=True)
class PackageSource:
    """An installed Julia package's real source dir and write policy.

    Records where a package the active environment resolves lives on disk
    (found with ``pkgdir``) and whether the agent may write there. ``writable``
    is set only for a ``Pkg.develop`` checkout; a registry install in the shared
    depot stays read-only, and ``build_backend`` uses the read-only ones to
    guard depot writes.
    """

    name: str
    path: Path
    writable: bool = False


_provider_profiles_registered = False


def resolve_model(
    explicit: Any | None = None,
    *,
    workspace_model: str | None = None,
    user_model: str | None = None,
) -> Any:
    """Resolve the model id by precedence, highest first.

    ``--model`` (``explicit``) > workspace config > user-global config >
    ``$JUTUL_AGENT_MODEL`` (a dev/CI override) > ``DEFAULT_MODEL``.
    """
    return (
        explicit or workspace_model or user_model or os.environ.get(MODEL_ENV_VAR) or DEFAULT_MODEL
    )


def register_provider_profiles() -> None:
    """Register the base HarnessProfile for every provider, once per process.

    deepagents has no wildcard profile key, so the same profile is registered
    under each provider in ``models.PROVIDERS``. The profile only disables the
    stock general-purpose subagent; every prompt rule lives in
    ``agent.prompts`` so each is stated exactly once.

    Parallel tool calls are a first-class capability (the tool node runs them
    concurrently) and we do not suppress them per provider: local models emit
    clean parallel calls the vast majority of the time, and the rare malformed
    one is caught generically by ``InvalidToolCallRecoveryMiddleware`` rather
    than by crippling the capability through the prompt.
    """

    global _provider_profiles_registered
    if _provider_profiles_registered:
        return

    profile = HarnessProfile(
        general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
    )
    for provider in PROVIDERS:
        register_harness_profile(provider, profile)
    _provider_profiles_registered = True


# Reasoning visibility: models that can reason get it requested explicitly.
# OpenAI's recent reasoning models default to effort "none" (no reasoning at
# all) and stream nothing during the thinking phase unless a summary is
# requested; Anthropic's extended thinking is off unless a budget is given;
# Gemini thinks by default but keeps the thoughts hidden.
_OPENAI_REASONING = {"effort": "medium", "summary": "auto"}
_ANTHROPIC_THINKING_BUDGET_TOKENS = 10_000
# Thinking spends from the same output budget as the answer, so the cap must
# clear the thinking budget with room for a long reply.
_ANTHROPIC_MAX_TOKENS = 24_000


def _model_profile(model_id: str) -> dict[str, Any]:
    """The provider package's bundled profile for the model (``{}`` unknown).

    Profiles are the maintained capability source. Keying decisions on them
    means new models are covered by upgrading the provider package, never by
    editing a list here. Reading one builds the model, which needs the
    provider key; callers treat a raised error as "no settings".
    """
    from langchain.chat_models import init_chat_model

    return init_chat_model(model_id).profile or {}


def _ollama_settings(model_id: str) -> dict[str, Any]:
    from jutul_agent import ollama_client

    name = ollama_client.model_name(model_id)
    settings: dict[str, Any] = {"num_ctx": ollama_client.num_ctx(name)}
    # Thinking-capable local models must have think mode requested
    # explicitly: left at the daemon default, the thinking segment is
    # dropped on the client side, so a turn the model spends entirely on
    # thinking surfaces as an empty reply with no tool calls, and the agent
    # falls silent. Requested, the thinking is separated, tool calls parse
    # reliably, and the reasoning becomes visible like the cloud providers'.
    if ollama_client.thinks(name):
        settings["reasoning"] = True
    return settings


def _openai_settings(model_id: str) -> dict[str, Any] | None:
    # The profile must also mark the temperature parameter unsupported: that
    # separates true reasoning models (which accept reasoning.effort) from
    # the -chat hybrids (which reject it).
    profile = _model_profile(model_id)
    if profile.get("reasoning_output") and profile.get("temperature") is False:
        return {"reasoning": dict(_OPENAI_REASONING)}
    return None


def _anthropic_settings(model_id: str) -> dict[str, Any] | None:
    if _model_profile(model_id).get("reasoning_output"):
        return {
            "thinking": {
                "type": "enabled",
                "budget_tokens": _ANTHROPIC_THINKING_BUDGET_TOKENS,
            },
            "max_tokens": _ANTHROPIC_MAX_TOKENS,
        }
    return None


def _google_settings(model_id: str) -> dict[str, Any] | None:
    # Gemini thinks by default (level "high" on Gemini 3+); include_thoughts
    # only makes the thinking visible. An empty profile means the model is
    # newer than the package's data, and those all think; the data marks the
    # legacy non-thinking models explicitly.
    profile = _model_profile(model_id)
    if profile.get("reasoning_output") or not profile:
        return {"include_thoughts": True}
    return None


# Construction-time keyword arguments per provider, applied when a model spec
# string is resolved for the agent. The resolution loop is provider-agnostic:
# adding a provider is one entry here, providers without an entry (and any
# model whose resolver declines or fails) pass through as plain spec strings.
# Neither the agent framework nor langchain offers a cross-provider request
# shape for reasoning, so this registry is where the per-provider shapes live.
_MODEL_SETTINGS: dict[str, Callable[[str], dict[str, Any] | None]] = {
    "ollama": _ollama_settings,
    "openai": _openai_settings,
    "anthropic": _anthropic_settings,
    "google_genai": _google_settings,
}


def _resolve_model_for_agent(model: Any) -> Any:
    """A pre-built model instance when the model needs extra construction
    arguments, the spec string otherwise.

    Two reasons to pass an instance instead of the spec string. deepagents
    skips profile resolution for specs with more than one colon, so
    `ollama:<model>:<tag>` ids would otherwise get neither our harness
    profile nor a context setting (an instance resolves by provider). And
    models that can reason get it requested and made visible at construction
    (``_MODEL_SETTINGS``): without that the model reasons silently or, on
    recent OpenAI models, not at all. Everything else (unknown providers,
    models whose resolver declines, failures while probing) keeps the spec
    string so deepagents builds it exactly as before; an already-built model
    is passed through untouched.
    """
    if not isinstance(model, str):
        return model
    settings = _MODEL_SETTINGS.get(provider_of(model))
    if settings is None:
        return model
    try:
        kwargs = settings(model)
        if kwargs:
            from langchain.chat_models import init_chat_model

            instance = init_chat_model(model, **kwargs)
            _set_profile_window(instance, model)
            return instance
    except Exception:
        pass  # no key yet, unknown model, or offline: run with the plain spec
    return model


def _set_profile_window(model: Any, model_id: str) -> None:
    """Feed the real loaded window into the model's profile.

    deepagents' stock summarizer sizes its trigger from
    ``model.profile['max_input_tokens']``. Ollama ships no profile, so the
    trigger would otherwise fall back to a fixed default far larger than the
    loaded num_ctx (the local-overflow bug). Feed in the window we actually use.
    Best-effort: a model whose profile can't be set keeps its native window.
    """
    from jutul_agent.models import context_window

    window = context_window(model_id)
    if not window:
        return
    try:
        current = getattr(model, "profile", None)
        profile = dict(current) if isinstance(current, dict) else {}
        profile["max_input_tokens"] = window
        model.profile = profile
    except Exception:
        pass


def _depot_readonly_roots(
    package_sources: Sequence[PackageSource] | None,
) -> tuple[Path, ...]:
    """Real directories the agent must not write into: the shared Julia depot.

    Registry package source (``writable=False``) lives under a depot
    ``.../packages/`` directory that is shared across projects; editing it would
    corrupt other projects, so writes there are refused. We guard the whole
    ``packages`` ancestor (so a package the agent ``Pkg.add``s mid-session is
    covered too), and never a ``Pkg.develop`` checkout, which lives outside the
    depot and stays writable. Reads/greps are unaffected.
    """
    roots: set[Path] = set()
    for src in package_sources or ():
        if src.writable:
            continue
        path = Path(src.path).resolve()
        depot_packages = next((p for p in (path, *path.parents) if p.name == "packages"), path)
        roots.add(depot_packages)
    return tuple(sorted(roots))


def build_backend(
    *,
    workspace: Path | None = None,
    package_sources: Sequence[PackageSource] | None = None,
    added_dirs: Sequence[str | Path] | None = None,
    artifacts_root: str | Path | None = None,
) -> CompositeBackend:
    """The agent's filesystem: one real-path backend over the workspace.

    The workspace is rooted at ``workspace`` (defaults to ``workspace_root()``)
    and runs in real-path mode, so a relative path resolves against it and an
    absolute path as itself, the same file the shell and the Julia REPL see.
    Package source, skills, memory, and added folders are all read and written
    at their real paths through this backend.
    Writes into the shared Julia depot (registry ``package_sources``) are refused
    so the agent can study installed source but not corrupt it. ``added_dirs``
    are validated and recorded (see ``agent.added_dirs``), and the agent uses their
    real paths. The composite wrapper carries the recursive-grep fix and an (empty)
    route table the live session can still extend.
    """

    ws = workspace or workspace_root()
    backend = RecursiveGrepBackend(
        default=WorkspaceShellBackend(
            root_dir=ws,
            virtual_mode=False,
            inherit_env=True,
            readonly_roots=_depot_readonly_roots(package_sources),
        ),
        routes={},
        # Where the framework writes offloaded artifacts (evicted tool results,
        # summarized conversation history). On a real-path backend the default
        # "/" would resolve to the filesystem root, so these writes need a real
        # writable directory; build_agent points it at the session state dir.
        artifacts_root=str(artifacts_root) if artifacts_root is not None else "/",
    )
    for raw in added_dirs or ():
        add_dir(backend, raw, workspace=ws)
    return backend


def skill_sources(adapter: SimulatorAdapter) -> list[str | tuple[str, str]]:
    """Skill sources for ``SkillsMiddleware``: the real skill directories.

    Bundled skills ship with the package (so they resolve at a real path even
    from a pip install, no repo checkout needed). The middleware reads their
    ``SKILL.md`` through the real-path backend. Labels are explicit.

    Precedence (each layer can override the previous):
    built-in shared → built-in simulator → user global → user per-simulator → workspace-local
    """

    sources: list[str | tuple[str, str]] = []
    if SHARED_SKILLS_DIR.exists():
        sources.append((str(SHARED_SKILLS_DIR), "Built-in"))
    if adapter.skills_dir.exists():
        sources.append((str(adapter.skills_dir), adapter.display_name))
    global_user = user_skills_dir()
    if global_user.exists():
        sources.append((str(global_user), "User"))
    sim_user = user_simulators_dir() / adapter.name / "skills"
    if sim_user.exists():
        sources.append((str(sim_user), f"User ({adapter.display_name})"))
    ws_skills = workspace_root() / ".jutul-agent" / "skills"
    if ws_skills.exists():
        sources.append((str(ws_skills), "Workspace"))
    return sources


def build_agent(
    session: Session,
    *,
    model: Any | None = None,
    checkpointer: Any | None = None,
    approval_mode: ApprovalMode | str | None = None,
    package_sources: Sequence[PackageSource] | None = None,
    added_dirs: Sequence[str | Path] | None = None,
) -> tuple[Any, CompositeBackend]:
    """Build the session agent and return it with its live ``CompositeBackend``.

    The backend is returned alongside the agent because it's the same object the
    filesystem middleware uses: callers keep it to add more folders mid-session
    (the TUI ``/add-dir`` command), and a route added to it is visible to the
    agent's next tool call. Callers that don't need it just ignore the second
    element.
    """

    register_provider_profiles()
    # The file tools run on real paths; on Windows, let deepagents' path validation
    # accept real ``C:\\...`` paths instead of rejecting them as "not virtual".
    enable_windows_real_paths()

    memory_dir = ensure_memory_dir(session.memory_dir(workspace_memory=workspace_memory_dir()))
    backend = build_backend(
        package_sources=package_sources,
        added_dirs=added_dirs,
        artifacts_root=session.state_dir,
    )

    tools = [
        make_run_julia_tool(session),
        make_reset_julia_tool(session),
        make_plot_julia_tool(session),
        make_recapture_tool(session),
        make_close_plots_tool(session),
        make_record_attempt_tool(session),
        make_write_report_tool(session),
        make_remember_tool(memory_dir),
    ]
    mode = (
        approval_mode
        if isinstance(approval_mode, ApprovalMode)
        else parse_approval_mode(approval_mode)
    )
    subagents = [factory(session) for factory in session.simulator.subagent_factories]
    model_spec = resolve_model(model)
    resolved_model = _resolve_model_for_agent(model_spec)
    agent = create_deep_agent(
        model=resolved_model,
        backend=backend,
        tools=tools,
        system_prompt=assemble_session_prompt(
            session.simulator,
            open_windows=session.open_windows,
            resumed=session.resumed,
            workspace=workspace_root(),
        ),
        skills=skill_sources(session.simulator),
        subagents=subagents,
        interrupt_on=interrupt_on_for_mode(mode),
        # Auto-compaction is deepagents' stock SummarizationMiddleware (installed
        # by create_deep_agent); its trigger is sized from the model profile,
        # which _set_profile_window points at the real loaded window. The
        # TraceRecorder records each compaction. Tool-result clearing is our one
        # addition: it sheds old results below the summary trigger so the
        # expensive summary call is reserved for when clearing isn't enough.
        middleware=[
            m
            for m in (
                build_memory_middleware(backend, memory_dir),
                build_context_editing_middleware(
                    model_id=model_spec if isinstance(model_spec, str) else None,
                ),
                # Before the recorder in the list so its after_model hook runs
                # after the recorder's (after_model composes in reverse): the
                # model call is traced before a recovery jump re-enters the model.
                InvalidToolCallRecoveryMiddleware(),
                TraceRecorder(session.trace),
            )
            if m is not None
        ],
        checkpointer=checkpointer,
    )
    return agent, backend
