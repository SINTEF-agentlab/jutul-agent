"""The FastAPI application: REST lifecycle plus the per-session turn WebSocket.

REST creates, lists, resumes, and closes sessions, and serves the files a
session produces. The WebSocket at ``/sessions/{id}/stream`` carries one turn at
a time: the client sends a prompt (or an approval decision, or a cancel), and
the server streams the agent's events back, serialized by ``protocol``.

``create_app`` takes a ``SessionManager`` so tests can inject one whose sessions
wrap fakes; the default manager stands up real sessions.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from jutul_agent.agent.approval import (
    ToolAllowlist,
    build_resume_payload,
    categories_for_interrupt,
)
from jutul_agent.agent.capabilities import HttpToolSpec, http_tool_capability
from jutul_agent.interfaces.server import protocol
from jutul_agent.interfaces.server.manager import SessionBusyError, SessionManager
from jutul_agent.preview import TOOL_STREAM_RENDER_INTERVAL, TOOL_STREAM_TAIL_CAP
from jutul_agent.session_host import SessionHost
from jutul_agent.trace import schema

# The web UI ships pre-built next to this module: ``web_dist`` is the Vite build of
# ``webapp/`` (the React app), committed and shipped so an install needs no Node.
_SERVER_DIR = Path(__file__).resolve().parent
WEB_DIST_DIR = _SERVER_DIR / "web_dist"


def _ui_dir() -> Path | None:
    """The directory to serve the web UI from, or ``None`` if it is not built."""
    return WEB_DIST_DIR if (WEB_DIST_DIR / "index.html").is_file() else None


def _register_web_mime_types() -> None:
    """Force correct MIME types for the built UI's assets before serving.

    Vite loads its bundle via ``<script type="module">``, which a browser runs only
    when the file is served with a JavaScript MIME type. On Windows ``mimetypes``
    reads the registry, where ``.js`` is frequently ``text/plain`` — which makes the
    browser refuse the module and render a blank page. Registering the types in
    process makes serving correct regardless of the host registry. Idempotent.
    """
    import mimetypes

    mimetypes.add_type("text/javascript", ".js")
    mimetypes.add_type("text/javascript", ".mjs")
    mimetypes.add_type("text/css", ".css")
    mimetypes.add_type("application/json", ".json")
    mimetypes.add_type("image/svg+xml", ".svg")


class CreateSessionRequest(BaseModel):
    # Optional: a server bound to a simulator (the web case) uses its own; a
    # request may still name one, which must match the bound simulator.
    sim: str | None = None
    model: str | None = None
    approval_mode: str | None = None
    workspace: str | None = None
    # The host app's declarative HTTP tools, validated straight into the domain
    # ``HttpToolSpec`` (one schema, no parallel request model to keep in sync).
    tools: list[HttpToolSpec] | None = None


class ResumeSessionRequest(BaseModel):
    sim: str | None = None
    model: str | None = None
    approval_mode: str | None = None
    workspace: str | None = None


class CredentialRequest(BaseModel):
    # ``provider`` is a catalog name, label, or model id; the server resolves it
    # to the provider's key variable so the UI never sends a raw env-var name.
    provider: str
    value: str


def _request_extensions(tools: list[HttpToolSpec] | None) -> list:
    """Turn declared HTTP tool specs into a host-app capability, if any were sent."""
    if not tools:
        return []
    return [http_tool_capability("host-app", tools)]


def create_app(
    manager: SessionManager | None = None,
    *,
    ui: bool = True,
    default_sim: str | None = None,
    default_approval_mode: str | None = None,
    default_model: str | None = None,
    julia_project: Path | None = None,
    threads: str | None = None,
    add_dirs: Sequence[Path] = (),
    ephemeral_memory: bool = False,
) -> FastAPI:
    # The launch-wide knobs (folder-fixed) ride in the default manager's host
    # factory; an injected manager (tests) brings its own. The model is a default
    # a request can still override, so it is applied at the create/resume endpoint.
    if manager is None:
        from jutul_agent.interfaces.server.manager import (
            SessionLaunchDefaults,
            make_host_factory,
        )

        manager = SessionManager(
            host_factory=make_host_factory(
                SessionLaunchDefaults(
                    julia_project=julia_project,
                    threads=threads,
                    add_dirs=tuple(add_dirs),
                    ephemeral_memory=ephemeral_memory,
                )
            )
        )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield
        await manager.aclose()

    app = FastAPI(
        title="jutul-agent",
        summary="Drive a jutul-agent session over HTTP and WebSocket.",
        lifespan=lifespan,
    )
    app.state.manager = manager

    @app.get("/models")
    def list_models() -> dict[str, Any]:
        from jutul_agent.models import DEFAULT_MODEL, PROVIDERS, discover_models

        # The selectable models for the UI's model picker (provider profile data,
        # no model instantiation), grouped flat with their provider.
        models = [
            {"id": info.id, "label": info.label, "provider": provider, "note": info.note}
            for provider, infos in discover_models().items()
            for info in infos
        ]
        # Report the server's actual default: the launch ``--model`` if one was given,
        # else the catalog default. The UI seeds its model from this, so it must match
        # what new sessions use, or the UI would show the wrong model, query the wrong
        # context window, and resume a from-disk session onto the catalog default.
        return {
            "default": default_model or DEFAULT_MODEL,
            "providers": sorted(PROVIDERS),
            "models": models,
            # The wire-contract version for third-party front ends (see
            # docs/server-interface.md); the bundled UI ships in lockstep.
            "protocol": protocol.PROTOCOL_VERSION,
        }

    @app.get("/models/window")
    def model_window(model: str | None = None) -> dict[str, Any]:
        """The context window for a model (for the % indicator), or null if unknown.

        Separate from ``/models`` because it instantiates the model to read its
        profile, so the UI asks for just the active model, lazily.
        """
        from jutul_agent.models import DEFAULT_MODEL, context_window

        return {"model": model or DEFAULT_MODEL, "window": context_window(model or DEFAULT_MODEL)}

    @app.get("/credentials")
    def list_credentials() -> dict[str, Any]:
        """Which provider keys are configured, so the UI can prompt for a missing one.

        The masked previews are safe to show (a few characters, the rest hidden)
        and let the user confirm which key is saved; full secrets never cross the
        wire. ``shadowed`` flags a saved key that an environment value overrides.
        """
        from jutul_agent.credentials import key_status, user_env_path

        return {
            "path": str(user_env_path()),
            "providers": [
                {
                    "provider": st.provider,
                    "label": st.label,
                    "env_var": st.env_var,
                    "is_set": st.is_set,
                    "masked": st.masked,
                    "source": st.source,
                    "shadowed": st.shadowed,
                }
                for st in key_status()
            ],
        }

    @app.post("/credentials")
    def set_credential(req: CredentialRequest) -> dict[str, Any]:
        """Save a provider's API key to the global ``.env`` and use it immediately."""
        from jutul_agent.credentials import store_credential_for_provider

        try:
            info, path = store_credential_for_provider(req.provider, req.value)
        except KeyError as exc:
            raise HTTPException(
                status_code=400, detail=f"unknown provider {req.provider!r}"
            ) from exc
        except ValueError as exc:  # empty key
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"provider": info.name, "env_var": info.key_env_var, "path": str(path)}

    @app.get("/simulators")
    def list_simulators() -> dict[str, Any]:
        from jutul_agent.simulators import registry

        names = registry.names()
        details = {}
        for name in names:
            adapter = registry.get(name)
            details[name] = {
                "display_name": adapter.display_name,
                "examples": list(adapter.example_prompts),
            }
        return {"simulators": names, "default": default_sim, "details": details}

    def _bound_sim(requested: str | None) -> str:
        """The simulator a new/resumed session must use.

        The server is bound to one folder, and a folder is bound to one simulator
        (chosen when ``jutul-agent web`` starts), so every session here uses that
        one — the web UI does not switch simulators in place. A request for a
        different simulator is refused. Without a bound simulator (tests, or a future
        multi-folder launcher) the caller's choice is honoured.
        """
        if default_sim is None:
            if not requested:
                raise HTTPException(status_code=400, detail="no simulator specified")
            return requested
        if requested and requested != default_sim:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"this server is bound to simulator '{default_sim}'; run "
                    "`jutul-agent web` from another folder to use a different simulator"
                ),
            )
        return default_sim

    def _require_credential(model: str | None) -> None:
        """Raise a structured 400 if ``model``'s provider needs a key we don't have.

        Resolves the same effective model a new session would use (the request's,
        else the server default, else the catalog default), so the guard matches
        what the kernel would actually load.
        """
        from jutul_agent.credentials import missing_credential
        from jutul_agent.models import DEFAULT_MODEL, provider_info

        effective = model or DEFAULT_MODEL
        env_var = missing_credential(effective)
        if env_var is None:
            return
        info = provider_info(effective)
        raise HTTPException(
            status_code=400,
            detail=protocol.credential_required_to_wire(
                provider=info.name if info else "",
                label=info.label if info else effective,
                env_var=env_var,
            ),
        )

    def _workspace_for(requested: str | None) -> Path | None:
        """The folder a session runs in: an explicit request, else the server's folder.

        The server runs in one folder (its launch directory, where the bound
        simulator's Julia environment lives), so a normal session runs there
        (``None`` lets ``SessionHost.start`` fall back to ``workspace_root()``).
        The ``requested`` override is retained for a future launcher that opens a
        session in a chosen folder.
        """
        return Path(requested) if requested else None

    @app.get("/sessions/history")
    def session_history(limit: int = 40) -> dict[str, Any]:
        """Resumable sessions on disk, newest first, with a title and simulator.

        A session's title comes from its stored ``title`` file (the first-prompt or
        LLM name). When that is missing (its titling never persisted) we fall back to
        deriving a title from the first user prompt, so a real conversation still
        shows in history instead of vanishing. Only a session with no prompt at all
        (an abandoned new-chat) is omitted.

        Ordered by last activity (the most recently used first), from each trace's
        last event time, since that is what a user looks for — not when the session
        was first created.
        """
        from jutul_agent.session import derive_session_title, list_sessions

        # Consider every session, then sort by last activity and cap last — slicing
        # before the sort would order by creation and cut an old-but-recently-used
        # session even though it belongs near the top.
        sessions: list[dict[str, Any]] = []
        for info in list_sessions():
            sim, first_prompt, last_active = _session_overview(info.state_dir)
            title = info.title or (derive_session_title(first_prompt) if first_prompt else None)
            if not title:
                continue  # no stored title and no prompt: an empty/abandoned new-chat
            sessions.append(
                {
                    "id": info.session_id,
                    "title": title,
                    "started": info.started.isoformat(),
                    "last_active": last_active or info.started.isoformat(),
                    "sim": sim or default_sim,
                }
            )
        sessions.sort(key=lambda s: s["last_active"], reverse=True)
        return {"sessions": sessions[: max(0, limit)]}

    @app.get("/sessions/{session_id}/messages")
    def session_messages(session_id: str) -> dict[str, Any]:
        """The full conversation for replay on resume, in the live wire shape.

        Emits the same message types the WebSocket streams during a turn — user
        and assistant text, reasoning, tool calls paired with their results, and
        views — so a reopened chat reconstructs inline exactly as it looked when
        the user left it, tool cards and all. Artifacts replay with ``live=False``
        because the Julia process restarted, so live plot embeds fall back to their
        saved posters.
        """
        host = manager.get(session_id)
        state_dir = host.session.state_dir if host else _session_state_dir(session_id)
        if state_dir is None:
            raise HTTPException(status_code=404, detail="no such session")
        from jutul_agent.trace import TraceLog

        with TraceLog(state_dir / "trace.sqlite") as log:
            events = list(log.iter_events())
        return {"messages": replay_events(events, session_id)}

    @app.post("/sessions")
    async def create_session(req: CreateSessionRequest) -> dict[str, str]:
        sim = _bound_sim(req.sim)
        # Refuse before standing up the kernel if the model has no key: the UI
        # shows a key prompt on this structured error, then retries the create.
        _require_credential(req.model or default_model)
        try:
            host = await manager.create(
                sim=sim,
                model=req.model or default_model,
                approval_mode=req.approval_mode or default_approval_mode,
                workspace=_workspace_for(req.workspace),
                extensions=_request_extensions(req.tools),
            )
        except KeyError as exc:  # unknown simulator
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"session_id": host.session_id}

    @app.get("/sessions")
    def list_sessions() -> dict[str, list[str]]:
        return {"sessions": manager.list_ids()}

    @app.post("/sessions/{session_id}/resume")
    async def resume_session(session_id: str, req: ResumeSessionRequest) -> dict[str, Any]:
        if not _is_valid_session_id(session_id):
            raise HTTPException(status_code=404, detail="no such session")
        existing = manager.get(session_id)
        if existing is not None:
            # Another connection is using it: re-resuming would build a fresh kernel
            # and tear the live one down under it, so refuse.
            if existing.attached:
                raise HTTPException(
                    status_code=409, detail="session is already open in another connection"
                )
            # Still live and idle (the user navigated away and came back, or the socket
            # dropped and reconnected): reattach as-is. The live host is authoritative
            # because an in-session model or approval change already updated it via the
            # set_model/set_approval command. Re-applying the resume request's values
            # here would let a stale UI value clobber the user's in-session choice on
            # every reconnect: the UI always reports the default model by name (it does
            # not know the server's --model), and it never sends the approval mode, so
            # the request's defaults do not reflect what the session is actually using.
            # The kernel, history, and live REPL are untouched.
            manager.promote(session_id)
            return {"session_id": existing.session_id, "kernel_restarted": False}
        # Not live anymore (evicted, or a new server): resume from disk, which starts
        # a fresh kernel — the conversation is restored but in-memory REPL state is not.
        try:
            host = await manager.resume(
                session_id,
                sim=_bound_sim(req.sim),
                model=req.model or default_model,
                approval_mode=req.approval_mode or default_approval_mode,
                workspace=_workspace_for(req.workspace),
            )
        except (KeyError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"session_id": host.session_id, "kernel_restarted": True}

    @app.delete("/sessions/{session_id}")
    async def delete_session(session_id: str) -> dict[str, bool]:
        # Refuse to tear down a session a connection is still driving — closing its
        # kernel/checkpointer under a running turn would crash that turn. The client
        # closes its WebSocket first (which detaches), so this only blocks a stray
        # delete of an in-use session. The check-and-close is one atomic step in the
        # manager, so a connection can't attach in a gap and lose its kernel mid-turn.
        # Shutdown still force-closes via manager.aclose.
        try:
            closed = await manager.close(session_id, require_idle=True)
        except SessionBusyError as exc:
            raise HTTPException(
                status_code=409, detail="session is open in a connection; close it first"
            ) from exc
        if not closed:
            raise HTTPException(status_code=404, detail="no such session")
        return {"ok": True}

    @app.get("/sessions/{session_id}/artifacts/{path:path}")
    def get_artifact(session_id: str, path: str) -> FileResponse:
        # Resolve against the live session when it's loaded, else its on-disk output
        # dir — the same fallback get_transcript uses — so a plot or report stays
        # fetchable after the session is evicted from the live registry or the server
        # restarts, instead of going dead (404) while the file is still on disk.
        from jutul_agent.session import existing_output_dir

        host = manager.get(session_id)
        if host is not None:
            out_dir: Path | None = host.session.output_dir
        elif _is_valid_session_id(session_id):
            out_dir = existing_output_dir(session_id)
        else:
            out_dir = None
        if out_dir is None:
            raise HTTPException(status_code=404, detail="no such session")
        target = _resolve_artifact(out_dir, path)
        if target is None:
            raise HTTPException(status_code=404, detail="no such artifact")
        return FileResponse(target)

    @app.api_route("/live/{session_id}/{path:path}", methods=["GET", "HEAD"])
    async def live_plot_proxy(session_id: str, path: str, request: Request) -> Response:
        """Reverse-proxy a session's live Bonito plot server through this port.

        A live plot's Bonito server binds a fresh, unpredictable localhost port
        per session; handing the browser that raw ``127.0.0.1:<port>`` URL only
        works when the browser shares this machine's network namespace. Under any
        port-forwarded setup (an SSH tunnel, VS Code/Cursor remote, Docker) only
        this app server's own port is known to whatever is forwarding connections,
        so the raw port is a permanent, unfixable connection-refused. Bonito's
        ``proxy_url`` (set in ``plot_julia_src.web_server_start``) already writes
        every URL it hands the browser as ``/live/<session_id>/...``, so this route
        only needs to strip that prefix and forward to the real port.
        """
        host = manager.get(session_id)
        port = host.session.web_plot_port if host is not None else None
        if port is None:
            raise HTTPException(status_code=404, detail="no live plot server for this session")

        import httpx

        # Answer a readiness probe here rather than forwarding it. Bonito renders the
        # figure's app for *any* request reaching its route, HEAD included, and each
        # render leaves behind a WGLMakie screen that nothing removes, because no browser
        # attaches to a HEAD. The canvas polls this route to find out when the live server
        # is up, so a figure that takes a moment collects a fistful of dead screens — and
        # those shadow the real view, since Makie resolves a scene's screen by taking the
        # first backend match, breaking picking and the click-to-select built on it.
        #
        # A readiness check only has to answer "is the server there", which a connection
        # settles without rendering anything.
        if request.method == "HEAD":
            try:
                _reader, writer = await asyncio.wait_for(
                    asyncio.open_connection("127.0.0.1", port), timeout=5.0
                )
            except (TimeoutError, OSError) as exc:
                raise HTTPException(
                    status_code=502, detail=f"live plot server unreachable: {exc}"
                ) from exc
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()
            return Response(status_code=200)

        hop_by_hop = {"host", "content-length", "connection"}
        headers = [(k, v) for k, v in request.headers.items() if k.lower() not in hop_by_hop]
        async with httpx.AsyncClient() as client:
            try:
                upstream = await client.request(
                    request.method,
                    f"http://127.0.0.1:{port}/{path}",
                    params=request.query_params,
                    headers=headers,
                    content=await request.body(),
                    timeout=30.0,
                )
            except httpx.TransportError as exc:
                raise HTTPException(
                    status_code=502, detail=f"live plot server unreachable: {exc}"
                ) from exc
        strip = {"content-length", "content-encoding", "transfer-encoding", "connection"}
        response_headers = {k: v for k, v in upstream.headers.items() if k.lower() not in strip}
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            headers=response_headers,
            media_type=upstream.headers.get("content-type"),
        )

    @app.websocket("/live/{session_id}/{path:path}")
    async def live_plot_ws_proxy(websocket: WebSocket, session_id: str, path: str) -> None:
        """Pump a live plot's widget websocket (a slider, a field selector) through
        the same reverse proxy as ``live_plot_proxy``, for the same reason: the
        browser has no route to Bonito's own port, only to this one."""
        host = manager.get(session_id)
        port = host.session.web_plot_port if host is not None else None
        if port is None:
            await websocket.close(code=1011)
            return

        import websockets
        from websockets.exceptions import ConnectionClosed

        await websocket.accept(subprotocol=websocket.headers.get("sec-websocket-protocol"))
        upstream_url = f"ws://127.0.0.1:{port}/{path}"
        if websocket.url.query:
            upstream_url += f"?{websocket.url.query}"

        try:
            async with websockets.connect(upstream_url, max_size=None) as upstream:

                async def to_upstream() -> None:
                    try:
                        while True:
                            message = await websocket.receive()
                            if message["type"] == "websocket.disconnect":
                                return
                            data = message.get("bytes")
                            if data is not None:
                                await upstream.send(data)
                                continue
                            text = message.get("text")
                            if text is not None:
                                await upstream.send(text)
                    finally:
                        with contextlib.suppress(Exception):
                            await upstream.close()

                async def from_upstream() -> None:
                    with contextlib.suppress(ConnectionClosed):
                        async for message in upstream:
                            if isinstance(message, bytes):
                                await websocket.send_bytes(message)
                            else:
                                await websocket.send_text(message)

                pumps = {asyncio.create_task(to_upstream()), asyncio.create_task(from_upstream())}
                _, pending = await asyncio.wait(pumps, return_when=asyncio.FIRST_COMPLETED)
                for task in pending:
                    task.cancel()
                # Wait for the cancellation to actually land: otherwise the loser can
                # still be mid-``send`` when we close the client socket below, racing
                # a "send after close" ASGI error instead of a clean cancel.
                await asyncio.gather(*pending, return_exceptions=True)
        except (WebSocketDisconnect, OSError):
            pass
        finally:
            with contextlib.suppress(RuntimeError):
                await websocket.close()

    @app.get("/sessions/{session_id}/transcript")
    def get_transcript(session_id: str, format: str = "html") -> Response:
        """Download the session transcript to share (html or md)."""
        host = manager.get(session_id)
        state_dir = host.session.state_dir if host else _session_state_dir(session_id)
        if state_dir is None:
            raise HTTPException(status_code=404, detail="no such session")
        from jutul_agent.session import existing_output_dir
        from jutul_agent.trace import TraceLog
        from jutul_agent.transcript import render_html, render_markdown

        with TraceLog(state_dir / "trace.sqlite") as log:
            events = list(log.iter_events())
        md = format in ("md", "markdown")
        # Inline images so the downloaded transcript shows its plots on its own,
        # off the server (the artifacts live in the session's output folder).
        out_dir = host.session.output_dir if host else existing_output_dir(session_id)
        artifact_dirs = [out_dir / "artifacts", out_dir] if out_dir else []
        body = render_markdown(events) if md else render_html(events, artifact_dirs=artifact_dirs)
        ext = "md" if md else "html"
        return PlainTextResponse(
            body,
            media_type="text/markdown" if md else "text/html",
            headers={"Content-Disposition": f"attachment; filename=transcript.{ext}"},
        )

    @app.get("/sessions/{session_id}/memory")
    def get_memory(session_id: str) -> Response:
        """The session's workspace memory, rendered as a page for the canvas."""
        host = manager.get(session_id)
        if host is None:
            raise HTTPException(status_code=404, detail="no such session")
        from jutul_agent.agent.memory import render_memory_overview
        from jutul_agent.transcript.markdown_html import render_markdown_html

        body = render_markdown_html(render_memory_overview(host.memory_dir))
        return HTMLResponse(_doc_page("Memory", body))

    @app.get("/sessions/{session_id}/context")
    def get_context(session_id: str) -> dict[str, Any]:
        """The full context-usage panel (same render as the TUI), as markdown.

        Usage figures come from the session's ``model_usage`` trace events (the
        first/last call and the count); the system-prompt and memory-index sizes
        are approximated the same way the TUI does. Rendered server-side so the web
        UI and the terminal show identical detail.
        """
        host = manager.get(session_id)
        if host is None:
            raise HTTPException(status_code=404, detail="no such session")
        from jutul_agent.agent.context_editing import clear_tool_uses_trigger_tokens
        from jutul_agent.agent.memory import list_memory_notes
        from jutul_agent.agent.summarization import auto_compact_trigger_tokens
        from jutul_agent.context_panel import context_component_estimates, render_context_panel
        from jutul_agent.models import DEFAULT_MODEL, context_window
        from jutul_agent.trace import TraceLog

        # Read usage from a fresh connection on the trace file: this endpoint runs in
        # a threadpool, and the session's own SQLite connection is bound to the thread
        # it was created on (the event loop), so reusing it here raises.
        with TraceLog(host.session.state_dir / "trace.sqlite") as log:
            usages = [e.payload for e in log.iter_events() if e.kind == schema.MODEL_USAGE]
        model = host.model or DEFAULT_MODEL
        window = context_window(model)

        memory_dir = host.memory_dir
        system_tokens, memory_tokens = context_component_estimates(host.session, memory_dir)

        body = render_context_panel(
            model_label=model,
            usage=usages[-1] if usages else None,
            window=window,
            first_usage=usages[0] if usages else None,
            model_calls=len(usages),
            system_prompt_tokens=system_tokens,
            memory_index_tokens=memory_tokens,
            memory_notes=len(list_memory_notes(memory_dir)),
            compact_trigger_tokens=auto_compact_trigger_tokens(window),
            clear_trigger_tokens=clear_tool_uses_trigger_tokens(window),
        )
        return {"markdown": body}

    @app.post("/sessions/{session_id}/upload")
    async def upload_file(session_id: str, file: Annotated[UploadFile, File()]) -> dict[str, str]:
        """Save an uploaded file into the session workspace so the agent can use it.

        Files land under ``uploads/`` in the workspace the agent runs in, so the
        user can refer to ``uploads/<name>`` and the file tools / REPL read it.
        """
        from jutul_agent.paths import workspace_root

        host = manager.get(session_id)
        if host is None:
            raise HTTPException(status_code=404, detail="no such session")
        ws = host.workspace or workspace_root()
        # Basename only, then a conservative safe name (no path separators escape).
        name = Path(file.filename or "upload").name
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", name).lstrip(".") or "upload"
        dest = ws / "uploads" / safe
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Stream to disk with a size cap so a large upload can't exhaust memory
        # (the whole server runs on one event loop).
        max_bytes = 100 * 1024 * 1024
        written = 0
        with dest.open("wb") as fh:
            while chunk := await file.read(1024 * 1024):
                written += len(chunk)
                if written > max_bytes:
                    fh.close()
                    dest.unlink(missing_ok=True)
                    raise HTTPException(status_code=413, detail="upload too large (max 100 MB)")
                fh.write(chunk)
        rel = f"uploads/{safe}"
        host.session.trace.append(schema.UPLOAD, {"path": rel})
        return {"path": rel}

    @app.websocket("/sessions/{session_id}/stream")
    async def stream(websocket: WebSocket, session_id: str) -> None:
        await _serve_stream(websocket, manager, session_id)

    # The bundled web UI is mounted last so the API routes above take precedence.
    ui_dir = _ui_dir()
    if ui and ui_dir is not None:
        _register_web_mime_types()
        app.mount("/", StaticFiles(directory=ui_dir, html=True), name="web")

    return app


def _artifact_url(session_id: str, rel: str) -> str:
    """The fetch URL for a session artifact given its workspace-relative path."""
    rel = rel[len("artifacts/") :] if rel.startswith("artifacts/") else rel
    return f"/sessions/{session_id}/artifacts/{rel}"


def artifact_wire_events(
    payloads: list[dict[str, Any]], session_id: str, *, live: bool = True
) -> list[dict[str, Any]]:
    """Wire events for produced artifacts: interactive HTML as ``viz``, the rest as ``artifact``.

    An HTML artifact (an interactive plot, or a written report) becomes a ``viz``
    the front end pins to its canvas, carrying the artifact's ``kind``, ``slot``,
    and a ``poster`` image URL when one was saved alongside.

    ``live=False`` is for replaying a resumed session: the Julia process (and with
    it any Bonito server that backed a live plot) has restarted, so a recorded
    ``live_url`` is dead. The figure then falls back to its saved PNG poster, shown
    inline as a static image, instead of an embed pointing at a gone server.
    """
    events: list[dict[str, Any]] = []
    for payload in payloads:
        url = _artifact_url(session_id, str(payload.get("path") or ""))
        live_url = payload.get("live_url") if live else None
        poster = payload.get("poster")
        poster_url = _artifact_url(session_id, str(poster)) if poster else None
        kind = payload.get("kind")
        # A plot or report is a canvas view; a bare image or file is a plain artifact.
        if live_url or payload.get("mime") == "text/html" or kind in ("plot", "report"):
            # What to embed: the live Bonito server while it's up, else the static
            # HTML record, else the saved poster image. On resume the live server is
            # gone, so a live plot falls back to its still-on-disk PNG poster instead
            # of a dead live URL — the figure stays viewable, just not interactive.
            if live_url:
                view_url = str(live_url)
            elif payload.get("mime") == "text/html":
                view_url = url
            else:
                view_url = poster_url or url
            events.append(
                protocol.viz_to_wire(
                    view_url,
                    title=payload.get("caption"),
                    kind=str(kind or "plot"),
                    poster=poster_url,
                    slot=payload.get("slot"),
                )
            )
        else:
            events.append(protocol.artifact_to_wire(payload, url=url))
    return events


def replay_events(events: list[Any], session_id: str) -> list[dict[str, Any]]:
    """Wire messages that reconstruct a recorded conversation for a resumed session.

    The trace-event analogue of the live ``protocol.to_wire`` path: it maps each
    persisted event (user/assistant/reasoning text, tool calls paired with their
    results, artifacts) to the same wire messages the WebSocket streams during a
    turn, so a reopened chat renders identically, tool cards and all. Kept as one
    function (not inlined in the endpoint) so the replay mapping lives in a single,
    testable place. Artifacts replay with ``live=False`` because the Julia process
    restarted, so a recorded live URL is dead and the figure falls back to its poster.
    """

    items: list[dict[str, Any]] = []
    # Tool calls whose result was recorded. Some never record one (e.g. a
    # write_todos that ends a turn); without a terminal event their replayed
    # card would spin forever, so we synthesize a finished event for them.
    result_ids = {e.payload.get("tool_call_id") for e in events if e.kind == schema.TOOL_RESULT}
    replayed_kinds = {
        schema.MESSAGE_USER: "user",
        schema.MESSAGE_ASSISTANT: "assistant",
        schema.MESSAGE_REASONING: "reasoning",
    }
    for ev in events:
        if ev.kind in replayed_kinds:
            text = str(ev.payload.get("content", "")).strip()
            if text:
                items.append(protocol.replay_message(replayed_kinds[ev.kind], text))
        elif ev.kind == schema.TOOL_CALL:
            name = ev.payload.get("name")
            cid = ev.payload.get("id")
            items.append(
                protocol.replay_tool_event(
                    event="requested", name=name, tool_call_id=cid, args=ev.payload.get("args")
                )
            )
            if cid not in result_ids:
                items.append(
                    protocol.replay_tool_event(event="finished", name=name, tool_call_id=cid)
                )
        elif ev.kind == schema.TOOL_RESULT:
            finished = "error" if ev.payload.get("status") == "error" else "finished"
            items.append(
                protocol.replay_tool_event(
                    event=finished,
                    name=ev.payload.get("name"),
                    tool_call_id=ev.payload.get("tool_call_id"),
                    content=ev.payload.get("content"),
                )
            )
        elif ev.kind == schema.ARTIFACT:
            items.extend(artifact_wire_events([ev.payload], session_id, live=False))
    return items


def _session_overview(state_dir: Path) -> tuple[str | None, str | None, str | None]:
    """A persisted session's simulator, first user prompt, and last-activity time.

    Used to label the history list, give an untitled session a fallback title, and
    order the list by when each was last used. Three indexed point-queries, so it
    stays cheap even on a long trace. Returns ``(None, None, None)`` on a
    missing/unreadable trace.
    """
    from jutul_agent.trace import TraceLog

    try:
        with TraceLog.open_readonly(state_dir / "trace.sqlite") as log:
            start = log.first_payload(schema.SESSION_START) or {}
            user = log.first_payload(schema.MESSAGE_USER) or {}
            sim = start.get("simulator")
            content = user.get("content")
            first_prompt = content if isinstance(content, str) else None
            return sim, first_prompt, log.last_timestamp()
    except Exception:
        return None, None, None


# A session id is server-generated and shaped like ``2026-06-21-2315-3f2a`` (plus
# an optional title slug). Validate the shape so a client-supplied id can never be
# a path traversal (``..``, separators, encoded slashes) into ``mkdir`` or a read.
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _is_valid_session_id(session_id: str) -> bool:
    return bool(_SESSION_ID_RE.match(session_id)) and ".." not in session_id


def _session_state_dir(session_id: str) -> Path | None:
    """The on-disk state dir for a (possibly not-loaded) session, if it exists."""
    from jutul_agent.session import sessions_root

    if not _is_valid_session_id(session_id):
        return None
    root = sessions_root().resolve()
    candidate = (root / session_id).resolve()
    if not candidate.is_relative_to(root):  # belt-and-braces against traversal
        return None
    return candidate if (candidate / "trace.sqlite").exists() else None


# Inert page policy for the canvas iframe: no scripts (the body is markdown the
# agent may have read), only inline styles and images. Defense in depth alongside
# the markdown renderer's html=False escaping.
_DOC_CSP = (
    "default-src 'none'; img-src 'self' data: http: https:; style-src 'unsafe-inline'; "
    "font-src data:; base-uri 'none'; form-action 'none'"
)


def _doc_page(title: str, body_html: str) -> str:
    """Wrap rendered HTML in a minimal, self-contained page for the canvas iframe."""
    import html as _html

    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<meta http-equiv='Content-Security-Policy' content=\"{_DOC_CSP}\">"
        "<title>" + _html.escape(title) + "</title><style>"
        "body{margin:0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;"
        "color:#1f2328;background:#fff;line-height:1.6}"
        ".page{max-width:760px;margin:0 auto;padding:2rem 1.6rem}"
        "h1,h2,h3{line-height:1.3;letter-spacing:-0.01em}h1{font-size:1.5rem}"
        "code{font-family:ui-monospace,Consolas,monospace;background:#f0f1ee;padding:.1em .35em;"
        "border-radius:5px;font-size:.88em}"
        "pre{background:#f0f1ee;border:1px solid #e3e3df;border-radius:10px;padding:.8rem;"
        "overflow:auto}"
        "pre code{background:none;padding:0}a{color:#0e7490}"
        "</style></head><body><div class='page'>" + body_html + "</div></body></html>"
    )


def _resolve_artifact(output_dir: Path, path: str):
    """The artifact file for ``path``, or ``None`` if it escapes the artifacts dir."""
    base = (output_dir / "artifacts").resolve()
    target = (base / path).resolve()
    if not target.is_file() or not target.is_relative_to(base):
        return None
    return target


async def _serve_stream(websocket: WebSocket, manager: SessionManager, session_id: str) -> None:
    await websocket.accept()
    # Claim the session atomically: acquire promotes + attaches it under the manager
    # lock, so it can't be evicted in the window between lookup and attach (which would
    # leave us running turns against a torn-down kernel). One connection per session: a
    # second (e.g. a duplicate tab) is refused cleanly.
    host = await manager.acquire(session_id)
    if host is None:
        existing = manager.get(session_id)
        message = (
            "this session is already open in another window"
            if existing is not None and existing.attached
            else "no such session"
        )
        await _safe_send(websocket, protocol.error_to_wire(message))
        await websocket.close()
        return

    state = _StreamState(websocket, host)
    # Re-surface an approval the session was paused on if an earlier connection
    # dropped while it was pending, so a reconnect can still answer it.
    await state.resync_pending()
    try:
        while True:
            try:
                message = await websocket.receive_json()
            except ValueError:  # a non-JSON text frame (json.JSONDecodeError)
                await _safe_send(
                    websocket, protocol.error_to_wire("invalid message (expected JSON)")
                )
                continue
            await state.handle(message)
    except WebSocketDisconnect:
        pass
    finally:
        # detach() must run even if teardown raises, or the session stays marked
        # attached forever and no later connection can ever acquire it again.
        try:
            await state.aclose()
        finally:
            host.detach()


class _StreamState:
    """Per-connection turn state: at most one turn in flight, plus pending approvals."""

    def __init__(self, websocket: WebSocket, host: SessionHost) -> None:
        self._ws = websocket
        self._host = host
        self._pending: list[Any] = []
        self._turn: asyncio.Task[None] | None = None
        # Held so the fire-and-forget titling task isn't garbage-collected mid-run.
        self._title_task: asyncio.Task[None] | None = None
        # High-water mark over trace event ids for side outputs (artifacts, ui),
        # so each is forwarded exactly once whether flushed mid-turn or at the end.
        self._side_output_id = 0
        # Per-tool-call streaming state, for rendering tool output the way the TUI
        # does (terminal-rendered, throttled): the accumulated raw output, the last
        # render time, the last delta's wire (a send template), and any pending
        # trailing-flush task.
        self._tool_streams: dict[str, str] = {}
        self._tool_render_at: dict[str, float] = {}
        self._tool_delta_wire: dict[str, dict[str, Any]] = {}
        self._tool_flush: dict[str, asyncio.Task[None]] = {}
        # Tool categories the user chose to "always allow" this session; future
        # matching interrupts auto-approve without asking again (like the TUI).
        self._allowlist = ToolAllowlist()

    async def handle(self, message: dict[str, Any]) -> None:
        kind = message.get("type")
        if kind == "prompt":
            await self._start_prompt(str(message.get("text") or ""))
        elif kind == "decision":
            await self._start_decision(message)
        elif kind == "cancel":
            await self.cancel_turn()
        elif kind == "ui_event":
            self._host.session.trace.append(schema.UI_EVENT, {"payload": message.get("payload")})
        elif kind == "command":
            await self._handle_command(message)
        else:
            await _safe_send(self._ws, protocol.error_to_wire(f"unknown message {kind!r}"))

    async def _handle_command(self, message: dict[str, Any]) -> None:
        """Apply a session setting (model, approval policy) mid-conversation.

        Rebuilds the agent in place — the kernel, the conversation history, and the
        live Julia state all survive — so a front end can offer these as commands.
        """
        if self._busy():
            await _safe_send(
                self._ws,
                protocol.error_to_wire("finish the current turn before changing settings"),
            )
            return
        command = message.get("command")
        arg = str(message.get("arg") or "")
        try:
            if command == "set_model":
                # A model switch under a paused approval would rebuild the agent
                # out from under the pending interrupt (the TUI refuses this too).
                if self._pending or await self._host.pending_interrupts():
                    await _safe_send(
                        self._ws,
                        protocol.error_to_wire(
                            "answer the pending approval before switching models"
                        ),
                    )
                    return
                # Mirror the TUI: a model whose key is missing prompts for it
                # instead of failing the switch. The UI saves the key, then retries.
                from jutul_agent.credentials import missing_credential
                from jutul_agent.models import local_model_error, provider_info

                env_var = missing_credential(arg)
                if env_var is not None:
                    info = provider_info(arg)
                    await _safe_send(
                        self._ws,
                        protocol.credential_required_to_wire(
                            provider=info.name if info else "",
                            label=info.label if info else arg,
                            env_var=env_var,
                        ),
                    )
                    return
                # A local model that isn't reachable/pulled/tool-capable fails
                # here with a reason, not opaquely on the next turn.
                problem = await local_model_error(arg)
                if problem is not None:
                    await _safe_send(self._ws, protocol.error_to_wire(problem))
                    return
                self._host.reconfigure(model=arg)
            elif command == "set_approval":
                self._host.reconfigure(approval_mode=arg)
                # A pending approval the new policy already allows resumes now
                # instead of waiting for a decision that is a foregone conclusion.
                await self._auto_resolve_pending()
            elif command == "add_dir":
                await _safe_send(self._ws, protocol.notice_to_wire(self._host.add_dir(arg)))
            elif command == "compact":
                note, _result = await self._host.compact()
                await _safe_send(self._ws, protocol.notice_to_wire(note))
            else:
                await _safe_send(self._ws, protocol.error_to_wire(f"unknown command {command!r}"))
                return
        except Exception as exc:  # surface a bad model/mode, keep the session alive
            await _safe_send(self._ws, protocol.error_to_wire(f"could not apply {command}: {exc}"))

    async def _start_prompt(self, text: str) -> None:
        if self._busy():
            await _safe_send(self._ws, protocol.error_to_wire("a turn is already running"))
            return
        # Name the session from its first prompt, like the CLI/TUI do, so it reads
        # well in the history list. Idempotent (only the first prompt sets it).
        with contextlib.suppress(Exception):
            self._host.session.adopt_title(text)
        runner = self._host.runner
        self._spawn(lambda: runner.run_prompt(text, on_message=self._on_message))

    async def _start_decision(self, message: dict[str, Any]) -> None:
        if self._busy():
            await _safe_send(self._ws, protocol.error_to_wire("a turn is already running"))
            return
        if not self._pending:
            await _safe_send(self._ws, protocol.error_to_wire("no approval is pending"))
            return
        kind = str(message.get("decision") or "approve")
        # "always_allow" is approve plus a session policy: remember this interrupt's
        # categories so future matching ones auto-approve (see _run_turn's loop).
        if kind == "always_allow":
            for interrupt in self._pending:
                for category in categories_for_interrupt(interrupt.value):
                    self._allowlist.add(category)
            kind = "approve"
        decision: dict[str, str] = {"type": kind}
        text = message.get("message")
        if text:
            decision["message"] = str(text)
        elif kind == "respond":
            # langchain's HITL reads decision["message"] by subscript for a respond
            # (unlike reject, which uses .get), so it must always be present; an empty
            # reply is valid.
            decision["message"] = ""
        payload = build_resume_payload(self._pending, decision)
        self._pending = []
        runner = self._host.runner
        self._spawn(lambda: runner.resume(payload, on_message=self._on_message))

    def _busy(self) -> bool:
        return self._turn is not None and not self._turn.done()

    async def resync_pending(self) -> None:
        """Re-send an approval the session was paused on when a prior connection dropped.

        A turn that pauses on an interrupt finishes its task with the interrupt recorded
        in the graph state, but the per-connection pending list is lost with the socket.
        On a fresh connection, re-read the persisted interrupts and re-send them so the
        user can still answer, instead of the paused turn being orphaned. A no-op when
        nothing is pending.
        """
        if self._busy() or self._pending:
            return
        try:
            pending = await self._host.runner.pending_interrupts()
        except Exception:
            return
        if not pending:
            return
        self._pending = list(pending)
        for interrupt in pending:
            await _safe_send(self._ws, protocol.interrupt_to_wire(interrupt))

    async def _auto_resolve_pending(self) -> None:
        """Resume a paused approval the (changed) policy now auto-allows."""
        from jutul_agent.agent.approval import (
            build_resume_payload,
            parse_approval_mode,
            should_auto_approve_interrupt,
        )

        if self._busy() or not self._pending:
            return
        mode = parse_approval_mode(self._host.approval_mode)
        if not all(
            should_auto_approve_interrupt(i.value, mode, allowlist=self._allowlist)
            for i in self._pending
        ):
            return
        payload = build_resume_payload(self._pending, {"type": "approve"})
        self._pending = []
        runner = self._host.runner
        self._spawn(lambda: runner.resume(payload, on_message=self._on_message))

    def _spawn(self, factory) -> None:
        self._turn = asyncio.create_task(self._run_turn(factory))

    async def _run_turn(self, factory) -> None:
        self._side_output_id = self._latest_event_id()
        try:
            # The host owns the policy loop (auto-resume past interrupts the
            # mode or the session allowlist already allows) and the settle
            # hooks, shared with the TUI.
            result = await self._host.drive_turn(
                factory,
                approval_mode=self._host.approval_mode,
                allowlist=self._allowlist,
                on_message=self._on_message,
            )
        except asyncio.CancelledError:
            await self._flush_side_outputs()
            await _safe_send(self._ws, protocol.turn_cancelled_to_wire())
            raise
        except Exception as exc:  # surface the failure, then end the turn
            await self._flush_side_outputs()  # surface anything produced before it failed
            await _safe_send(self._ws, protocol.error_to_wire(str(exc)))
            await _safe_send(self._ws, protocol.turn_end_to_wire([]))
            return
        finally:
            # A cancelled/errored turn can leave a tool mid-stream; clear streaming
            # state so a stale trailing flush can't fire on a later turn.
            self._end_all_tool_streams()
        await self._flush_side_outputs()
        self._pending = list(result.interrupts)
        if self._pending:
            # The turn paused for approval. Send the requests and wait for a
            # decision; the turn ends only once it runs to completion.
            for interrupt in self._pending:
                await _safe_send(self._ws, protocol.interrupt_to_wire(interrupt))
            return
        usage = protocol.usage_to_wire(result.messages)
        if usage is not None:
            await _safe_send(self._ws, usage)
        await _safe_send(self._ws, protocol.turn_end_to_wire(result.messages))
        task = self._host.maybe_title(self._on_titled)
        if task is not None:
            self._title_task = task

    async def _on_titled(self, title: str) -> None:
        """Nudge the front end to refresh its history list with the new title."""
        await _safe_send(self._ws, protocol.ui_command("history_changed", {"title": title}))

    def _latest_event_id(self) -> int:
        return self._host.session.trace.max_id()

    async def _flush_side_outputs(self) -> None:
        """Forward side outputs produced since the last flush: artifacts (plots,
        reports) and UI commands a tool emitted. Tracks a high-water mark over trace
        event ids, so a plot or report appears inline the moment its tool finishes
        (flushed from ``_on_message``) rather than all at once at turn end. Only the
        events since the last flush are read, so a long turn does not re-scan the
        whole trace on every tool completion."""
        for event in self._host.session.trace.events_after(self._side_output_id):
            self._side_output_id = event.id
            if event.kind == schema.ARTIFACT:
                for wire in artifact_wire_events([event.payload], self._host.session_id):
                    await _safe_send(self._ws, wire)
            elif event.kind == schema.UI_COMMAND:
                action = str(event.payload.get("action") or "")
                payload = event.payload.get("payload")
                await _safe_send(self._ws, protocol.ui_command(action, payload))

    async def _on_message(self, event: Any) -> None:
        wire = protocol.to_wire(event)
        if wire is None:
            return
        if wire.get("type") == "tool":
            cid = wire.get("tool_call_id")
            kind = wire.get("event")
            if kind == "delta" and cid:
                await self._on_tool_delta(cid, wire)
                return  # _on_tool_delta sends (now or on its trailing flush)
            if kind in ("finished", "error") and cid:
                # The final result (terminal-rendered by the kernel) replaces the
                # live stream; stop any pending flush and drop the per-call buffers.
                self._end_tool_stream(cid)
        await _safe_send(self._ws, wire)
        # A tool just finished: surface any artifacts/ui it produced right away, so a
        # plot or report appears inline as it happens instead of all at turn end.
        if wire.get("type") == "tool" and wire.get("event") in ("finished", "error"):
            await self._flush_side_outputs()

    async def _on_tool_delta(self, cid: str, wire: dict[str, Any]) -> None:
        """Accumulate a tool's raw output delta and send the terminal-rendered state.

        Mirrors the TUI: cursor moves and carriage returns are replayed through the
        screen emulator so progress output reads as one updating block, not a gap,
        and re-rendering is throttled. Throttling is leading+trailing — a delta that
        arrives within the interval schedules a trailing flush — so the last partial
        line never lingers unshown until the next event.
        """
        import time

        raw = self._tool_streams.get(cid, "") + (wire.get("content") or "")
        buf = raw[-TOOL_STREAM_TAIL_CAP:]
        self._tool_streams[cid] = buf
        self._tool_delta_wire[cid] = wire  # send template (name/label) for the flush
        if time.monotonic() - self._tool_render_at.get(cid, 0.0) >= TOOL_STREAM_RENDER_INTERVAL:
            await self._flush_tool_stream(cid)
        elif cid not in self._tool_flush:
            self._tool_flush[cid] = asyncio.create_task(self._delayed_flush(cid))

    async def _delayed_flush(self, cid: str) -> None:
        try:
            await asyncio.sleep(TOOL_STREAM_RENDER_INTERVAL)
            await self._flush_tool_stream(cid)
        except asyncio.CancelledError:
            pass
        finally:
            self._tool_flush.pop(cid, None)

    async def _flush_tool_stream(self, cid: str) -> None:
        import time

        from jutul_agent.juliakernel.text import render_terminal_output

        buf = self._tool_streams.get(cid)
        if buf is None:
            return
        self._tool_render_at[cid] = time.monotonic()
        wire = {
            **self._tool_delta_wire[cid],
            "content": render_terminal_output(buf),
            "replace": True,
        }
        await _safe_send(self._ws, wire)

    def _end_tool_stream(self, cid: str) -> None:
        task = self._tool_flush.pop(cid, None)
        if task is not None:
            task.cancel()
        self._tool_streams.pop(cid, None)
        self._tool_render_at.pop(cid, None)
        self._tool_delta_wire.pop(cid, None)

    def _end_all_tool_streams(self) -> None:
        """Drop all per-tool streaming state at turn end.

        A normal turn ends each stream via its tool's ``finished`` event; a
        cancelled or errored turn leaves a tool mid-stream (no such event), so
        without this its pending ``_delayed_flush`` would fire a stale frame on a
        later turn and the per-call dicts would grow across cancellations."""
        for task in self._tool_flush.values():
            task.cancel()
        self._tool_flush.clear()
        self._tool_streams.clear()
        self._tool_render_at.clear()
        self._tool_delta_wire.clear()

    async def cancel_turn(self) -> None:
        if self._busy():
            self._turn.cancel()  # type: ignore[union-attr]
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._turn  # type: ignore[arg-type]

    async def aclose(self) -> None:
        """Tear down on disconnect: cancel a running turn and any in-flight titling.

        The titling task is a fire-and-forget model call; without this it would
        keep running (and spend) after the connection is gone.
        """
        await self.cancel_turn()
        self._end_all_tool_streams()
        if self._title_task is not None and not self._title_task.done():
            self._title_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._title_task


async def _safe_send(websocket: WebSocket, message: dict[str, Any]) -> None:
    """Send a JSON message, ignoring a socket that is already closing."""
    with contextlib.suppress(Exception):
        await websocket.send_json(message)
