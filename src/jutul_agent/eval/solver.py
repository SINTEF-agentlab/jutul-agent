"""Run one full jutul-agent session per Inspect sample.

The solver builds the same stack a real session uses (workspace, Julia
kernel, :class:`Session`, ``build_agent``, ``TurnRunner``) inside Inspect's
``agent_bridge``. The bridge intercepts the model client, so the eval's
``--model`` decides which provider actually answers while the agent's tools,
skills, prompt, and trace run unchanged. Scorers grade through the sample
store, which records where the session's workspace and trace database ended
up (see :data:`STORE_WORKSPACE`, :data:`STORE_TRACE_DB`).

Sample metadata understood by the solver:

- ``fixtures``: mapping of workspace-relative path -> file content, written
  before the turn starts.
- ``needs_env``: instantiate the simulator's Julia environment in the
  workspace first (slow on a cold depot; off by default).
- ``needs_display``: start a managed virtual display so GLMakie plotting
  works headless.
"""

from __future__ import annotations

import asyncio
import re
import tempfile
import uuid
from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from inspect_ai.agent import AgentState, agent_bridge
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.solver import Generate, Solver, TaskState, solver

STORE_WORKSPACE = "jutul/workspace"
STORE_TRACE_DB = "jutul/trace_db"
STORE_SESSION_ID = "jutul/session_id"
STORE_OUTPUT_DIR = "jutul/output_dir"
STORE_RUNCONFIG = "jutul/runconfig"

# One session at a time: the workspace root is process-global state
# (``paths.set_workspace_root``), so concurrent samples would race on it.
_SESSION_LOCK = asyncio.Lock()


def load_eval_credentials() -> None:
    """Load provider keys the way the app does (global + cwd ``.env``).

    Call at task-module import time: Inspect resolves the eval's model (and
    checks its key) before any solver runs.
    """
    from dotenv import load_dotenv

    from jutul_agent.credentials import load_user_credentials

    load_user_credentials()
    load_dotenv()


def _bridge_model() -> Any:
    """The model instance the agent under test calls.

    The literal model name ``inspect`` is what the agent bridge intercepts;
    requests never reach the network, so the API key only has to satisfy the
    client constructor. The client family has to match the eval's target
    provider in two cases:

    - Gemini targets use the google client: Gemini 3 requires its
      ``thought_signature`` to be replayed with function calls, and only the
      google-format round-trip preserves it.
    - Everything else uses the Anthropic client, because langchain-anthropic
      issues the plain ``messages.create`` calls the bridge understands
      (langchain-openai wraps every call in ``with_raw_response``, which the
      bridge's patched client does not produce).

    Streaming is disabled because the bridge rejects streaming requests; the
    agent's event stream still works, the HTTP call just resolves in one piece.
    """
    from langchain.chat_models import init_chat_model

    provider = str(get_model()).partition("/")[0]
    if provider in ("google", "vertex"):
        from jutul_agent.eval import _gemini_compat

        _gemini_compat.apply()
        spec = "google_genai:inspect"
    else:
        spec = "anthropic:inspect"
    return init_chat_model(spec, disable_streaming=True, api_key="inspect-agent-bridge")


# The bridge annotates assistant text with internal capsules that must not
# reach scorers: a <content-internal> metadata tag, and (for Gemini) a redacted
# <think> reasoning signature. Both carry base64 a number-extracting check would
# happily mine digits out of, so they are stripped before the answer is graded.
_INTERNAL_RE = re.compile(
    r"<content-internal>.*?</content-internal>|<think\b[^>]*>.*?</think>",
    re.DOTALL,
)


def _final_text(messages: list[Any]) -> str:
    """Text of the last assistant message, flattened from content blocks."""
    for message in reversed(messages):
        if getattr(message, "type", None) != "ai":
            continue
        content = message.content
        if isinstance(content, str):
            return _INTERNAL_RE.sub("", content).strip()
        if isinstance(content, list):
            parts = [
                part.get("text", "") if isinstance(part, dict) else str(part) for part in content
            ]
            return _INTERNAL_RE.sub("", "\n".join(p for p in parts if p)).strip()
        return _INTERNAL_RE.sub("", str(content)).strip()
    return ""


async def _eval_package_sources(julia_project: Path | None) -> list[Any] | None:
    """The installed package sources, resolved as a real session does.

    A real CLI session enumerates the env's package sources and passes them to
    ``build_agent`` (see ``run.py``) so the read-only guard protects the depot
    while the agent reads source at its real ``pkgdir`` path. The eval must do
    the same, so its filesystem matches a live session. Resolved from the env's
    manifest (no compile); ``None`` for env-less samples, which have no packages
    to browse.
    """
    if julia_project is None:
        return None
    from jutul_agent.agent.builder import PackageSource
    from jutul_agent.simulators.env_setup import resolve_env_package_sources

    env = await asyncio.to_thread(resolve_env_package_sources, julia_project)
    return [
        PackageSource(name=name, path=path, writable=is_dev)
        for name, (path, is_dev) in sorted(env.items())
    ]


# Simulators whose golden env has been built or re-resolved by this process;
# the alignment check runs once per run, not once per sample.
_ALIGNED_ENVS: set[str] = set()


def _golden_env(adapter: Any, simulator: str) -> Path:
    """A prepared workspace env for this simulator, built once and kept aligned.

    Instantiating a simulator env from scratch takes minutes; one prepared
    copy lives under the jutul-agent home (``eval-envs/<sim>``) and each
    sample starts from a copy of it. Copying preserves mtimes, so the
    precompile marker stays current and the per-sample env prep is a cheap
    reconcile instead of a full instantiate.

    Envs carry no version pins, so a cached copy freezes whatever upstream
    served when it was built and would silently grade against old versions
    after a new release. On its first use in a run, a cached env is therefore
    re-resolved against the registry (``update_env``); if that fails, the run
    fails rather than grade a misaligned environment.

    The golden is then brought fully current once with ``prepare_workspace_env``
    (the normal launch reconcile): the in-env JutulAgent runtime is refreshed and
    marked, and the env precompiled. Without this, a golden whose warm-source
    marker is stale (e.g. cached before the runtime changed) makes *every*
    per-sample copy re-copy the runtime and re-bake the env — turning a one-time
    cost into a per-sample one. Aligning the golden lets each copy's reconcile be
    the intended cheap no-op.
    """
    from jutul_agent.paths import state_home
    from jutul_agent.simulators.env_setup import (
        bootstrap_workspace,
        prepare_workspace_env,
        update_env,
    )

    golden = state_home() / "eval-envs" / simulator
    env = golden / ".jutul-agent" / "julia-env"
    if not (env / "Manifest.toml").exists():
        golden.mkdir(parents=True, exist_ok=True)
        bootstrap_workspace(adapter, workspace=golden, precompile=True)
    elif simulator not in _ALIGNED_ENVS:
        update_env(env)
    if simulator not in _ALIGNED_ENVS:
        prepare_workspace_env(adapter, workspace=golden, julia_project=env, sim_name=simulator)
    _ALIGNED_ENVS.add(simulator)
    return env


async def _run_jutul_session(
    *,
    simulator: str,
    prompt: str,
    fixtures: Mapping[str, str],
    needs_env: bool,
    needs_display: bool,
    scratch: Path,
    store: Any,
    ground_truth: str | None = None,
) -> str:
    from jutul_agent.agent.builder import build_agent
    from jutul_agent.agent.turns import TurnRunner
    from jutul_agent.display import managed_display, should_wrap_xvfb
    from jutul_agent.juliakernel import JuliaKernel, KernelConfig
    from jutul_agent.paths import set_workspace_root
    from jutul_agent.session import Session
    from jutul_agent.simulators import registry
    from jutul_agent.trace.schema import EVAL_TARGET

    adapter = registry.get(simulator)
    workspace = scratch / "workspace"
    workspace.mkdir(parents=True)
    for rel, content in fixtures.items():
        path = workspace / rel.lstrip("/")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    set_workspace_root(workspace)

    julia_project = None
    if needs_env:
        import shutil

        from jutul_agent.simulators.env_setup import prepare_workspace_env
        from jutul_agent.workspace import resolve_julia_project

        julia_project = resolve_julia_project(workspace)
        if julia_project == workspace / ".jutul-agent" / "julia-env":
            golden = await asyncio.to_thread(_golden_env, adapter, simulator)
            await asyncio.to_thread(shutil.copytree, golden, julia_project, dirs_exist_ok=True)
        await asyncio.to_thread(
            prepare_workspace_env,
            adapter,
            workspace=workspace,
            julia_project=julia_project,
            sim_name=simulator,
        )

    from jutul_agent.eval.runconfig import build_runconfig

    store.set(STORE_RUNCONFIG, build_runconfig(adapter, julia_project=julia_project))
    package_sources = await _eval_package_sources(julia_project)

    with ExitStack() as stack:
        env = None
        if needs_display and should_wrap_xvfb():
            env = {"DISPLAY": stack.enter_context(managed_display())}
        # Persist the session trace to a discoverable eval workspace under the state
        # home (not the temp scratch, which is cleaned up) so eval runs — where the
        # answer is known and silent failures matter most — can be reviewed afterwards
        # with `jutul-agent review`. Only the trace persists; the workspace and
        # artifacts stay in scratch.
        from jutul_agent.review.discovery import eval_sessions_state_root

        config = KernelConfig(julia_project=julia_project, cwd=workspace, env=env)
        async with JuliaKernel(config) as julia:
            session = Session.create(
                julia=julia,
                simulator=adapter,
                state_root=eval_sessions_state_root(simulator),
                ephemeral_memory=True,
            )
            store.set(STORE_WORKSPACE, str(workspace))
            store.set(STORE_TRACE_DB, str(session.state_dir / "trace.sqlite"))
            store.set(STORE_SESSION_ID, session.session_id)
            store.set(STORE_OUTPUT_DIR, str(session.output_dir))
            # Record the known answer so a later review can judge the result against
            # ground truth, the sharpest signal an eval run offers.
            if ground_truth:
                session.trace.append(EVAL_TARGET, {"expected": ground_truth})
            try:
                agent, _backend = build_agent(
                    session,
                    model=_bridge_model(),
                    approval_mode="auto",
                    package_sources=package_sources,
                )
                runner = TurnRunner(agent, thread_id=session.session_id, trace=session.trace)
                result = await runner.run_prompt(prompt)
            finally:
                session.finalize()

    if result.interrupts:
        return "[eval-error] the turn paused for tool approval in auto mode"
    return _final_text(result.messages)


@solver
def jutul_agent_solver(simulator: str = "jutuldarcy") -> Solver:
    """Solve each sample by handing it to a fresh jutul-agent session."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        fixtures = state.metadata.get("fixtures", {})
        needs_env = bool(state.metadata.get("needs_env", False))
        needs_display = bool(state.metadata.get("needs_display", False))
        sim = state.metadata.get("simulator", simulator)
        scratch = Path(tempfile.gettempdir()) / "jutul-agent-eval" / uuid.uuid4().hex[:8]
        scratch.mkdir(parents=True)

        agent_state = AgentState(messages=list(state.messages))
        async with _SESSION_LOCK, agent_bridge(agent_state):
            final = await _run_jutul_session(
                simulator=sim,
                prompt=state.input_text,
                fixtures=fixtures,
                needs_env=needs_env,
                needs_display=needs_display,
                scratch=scratch,
                store=state.store,
                ground_truth=state.metadata.get("expected"),
            )

        # The bridge tracked the real conversation; the completion graded by
        # scorers is the agent's own final message.
        state.messages = agent_state.messages
        state.output = ModelOutput.from_content(model=get_model().name, content=final)
        return state

    return solve
