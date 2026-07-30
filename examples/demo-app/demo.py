"""A minimal jutul-agent extension, end to end.

This example wires a custom simulator and tools into the agent to show the
extension seam in one place: a toy library (``DemoSim``) added as a simulator, and
a web ``Capability`` that adds two tools, an interactive WGLMakie plot served as
self-contained HTML (the non-trivial bit) and a UI action. Everything is added
through the extension seam rather than the built-in registry, so it doubles as a
template for adding a simulator and tools of your own, or for embedding the agent
in another application.

It serves the bundled web UI, so the interactive plot lands in the canvas and the
UI action shows as a note (a host app can consume it instead through
``window.onJutulUi``). Run it with ``python examples/demo-app/demo.py`` (the web
stack ships in the core install). The first run instantiates the Julia env, then
the app is served at http://127.0.0.1:8742.
"""

from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

from langchain_core.tools import tool

from jutul_agent.agent.capabilities import Capability
from jutul_agent.session import Session
from jutul_agent.simulators.base import SimulatorAdapter
from jutul_agent.simulators.env_setup import prepare_workspace_env
from jutul_agent.trace import schema
from jutul_agent.workspace import resolve_julia_project

DEMO_DIR = Path(__file__).resolve().parent
JULIA_ENV = DEMO_DIR / "julia_env"
WORKSPACE = DEMO_DIR / "workspace"

_PROMPT_FRAGMENT = (
    "This app runs DemoSim, a toy library with one function, "
    "`DemoSim.response(p; n)`, returning `(; x, y)` for a damped sinusoid whose "
    "frequency scales with the parameter `p`. To show the user a figure, call "
    "`plot_response(p)`; it renders an interactive plot they can rotate and zoom. "
    "To move the parameter control in their interface, call `set_param(p)`. The "
    "user can also move that control themselves, which you will be told about."
)

DEMO_ADAPTER = SimulatorAdapter(
    name="demo",
    display_name="DemoSim",
    module_dir=DEMO_DIR,
    package_imports=("DemoSim",),
    primary_package="DemoSim",
    domain_hints="A toy example simulator exposing one function, DemoSim.response(p).",
)


def _make_set_param_tool(session: Session):
    @tool
    async def set_param(p: float) -> str:
        """Move the parameter control on the user's interface to ``p``."""
        session.trace.append(schema.UI_COMMAND, {"action": "set_param", "payload": {"p": p}})
        return f"Set the parameter control to p = {p}."

    return set_param


def _make_plot_tool(session: Session):
    @tool
    async def plot_response(p: float) -> str:
        """Plot DemoSim.response(p) as an interactive figure embedded in the app."""
        rel = f"artifacts/demo-{uuid4().hex[:8]}.html"
        target = session.output_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        result = await session.julia.eval(_export_snippet(p, target))
        if result.error:
            return f"ERROR: {result.error}"
        session.trace.append(
            schema.ARTIFACT,
            schema.artifact_payload(
                path=rel, mime="text/html", caption=f"DemoSim response (p={p})"
            ),
        )
        return f"Plotted the interactive response for p = {p}."

    return plot_response


def _export_snippet(p: float, target: Path) -> str:
    """Julia to render DemoSim.response(p) with WGLMakie and export it as HTML."""
    return (
        "begin\n"
        "    using DemoSim, WGLMakie, Bonito\n"
        f"    local d = DemoSim.response({float(p)})\n"
        "    local fig = WGLMakie.lines(d.x, d.y)\n"
        f'    Bonito.export_static(raw"{target.as_posix()}", Bonito.App(fig))\n'
        '    "ok"\n'
        "end"
    )


def demo_capability() -> Capability:
    """The web capability for the demo: an interactive-plot tool and a UI control."""
    return Capability(
        name="demosim-web",
        tools=(_make_set_param_tool, _make_plot_tool),
        prompt_fragment=_PROMPT_FRAGMENT,
        surfaces=("web",),
    )


def create_demo_app():
    """Build the demo server: the standard app and bundled UI, running DemoSim."""
    from jutul_agent.interfaces.server.app import create_app
    from jutul_agent.interfaces.server.manager import SessionManager
    from jutul_agent.session_host import SessionHost

    WORKSPACE.mkdir(exist_ok=True)

    async def host_factory(
        *, sim, model, approval_mode, workspace=None, resume, session_id, extensions=()
    ):
        # The demo always runs DemoSim from its own prepared env, ignoring the
        # requested simulator/workspace; a real deployment would honour them.
        return await SessionHost.start(
            simulator=DEMO_ADAPTER,
            model=model,
            approval_mode=approval_mode,
            workspace=WORKSPACE,
            prepare_env=False,
            surface="web",
            extensions=[demo_capability(), *extensions],
            resume=resume,
            session_id=session_id,
        )

    # Serve the bundled web UI; every session runs DemoSim, so the server is bound
    # to a single "demo" simulator. The interactive plot the agent produces is
    # pinned in the canvas, and a `set_param` UI action shows as a note.
    return create_app(SessionManager(host_factory=host_factory), default_sim="demo")


def ensure_env() -> None:
    """Prepare the demo's workspace Julia env without touching the template."""

    print("Preparing the demo Julia env (this can take a while)...")
    prepare_workspace_env(
        DEMO_ADAPTER,
        workspace=WORKSPACE,
        julia_project=resolve_julia_project(WORKSPACE),
        sim_name=DEMO_ADAPTER.name,
    )


def main() -> int:
    try:
        import uvicorn
    except ModuleNotFoundError:
        print("The web stack is missing; reinstall jutul-agent (or `uv sync`).", file=sys.stderr)
        return 1
    ensure_env()
    print("Demo app on http://127.0.0.1:8742")
    uvicorn.run(create_demo_app(), host="127.0.0.1", port=8742)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
