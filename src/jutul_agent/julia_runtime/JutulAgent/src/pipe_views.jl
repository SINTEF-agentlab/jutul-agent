"""Live web figures over a pipe: Bonito pages without Bonito's HTTP server.

The web interface serves live figures from a `Bonito.Server` in this process, an
HTTP server on a loopback port that the app server reverse-proxies. The desktop app
may not open a listening endpoint of any kind, so there the same figures travel over
the kernel's auxiliary pipe: two anonymous pipes this process inherited from the
jutul-agent backend when it was spawned (their descriptors are in
`ENV["JK_AUX_FDS"]`). Nothing binds, listens or accepts; the backend relays between
the pipe and the window.

Two things travel on the pipe, as the kernel's own frames do (an ASCII header line
`KIND ID NBYTES\\n`, then exactly NBYTES payload bytes):

    backend -> Julia   GET <request> <n>     render the page of a route (payload)
                       OPN <view> 0          the page's connection opened
                       MSG <view> <n>        a message from the page
                       CLS <view> 0          the page went away
    Julia -> backend   HEL 1 0               a (new) server is serving
                       HTM <request> <n>     the page, as one HTML document
                       ERR <request> <n>     "<status> <reason>" instead
                       MSG <view> <n>        a message for the page
                       CLS <view> 0          this end closed the view

A page is a complete document with its scripts inlined (`Bonito.NoServer`), so it
needs nothing further fetched. Its connection is a `PipeConnection`, built on
Bonito's interface for custom connections (the one its Jupyter and Pluto support
use): the page opens a WebSocket to `<prefix>pipe/<view>` on its own origin, which
in the desktop app is not a network socket either but a message channel to the
backend, which relays here.

`PipeServer` stands in for the `Bonito.Server` in `Main.__JUTUL_WEB_SERVER__`:
`Bonito.route!` and `Bonito.delete_route!` accept it, so the code that serves,
replaces and closes figures is the same for both.
"""
module JutulAgentPipeViews

import WGLMakie

const Bonito = WGLMakie.Bonito

# How long a view outlives its page, as Bonito's own cleanup allows a closed tab:
# a reloaded page's new view takes the figure over within that time (see
# web_views.jl), and closing the old one first would close the figure under it.
const GRACE = 30.0

mutable struct PipeServer
    io::IO  # the write end of the auxiliary pipe
    # What the page prepends to reach this server through the backend, e.g.
    # "/live/<session>/".
    prefix::String
    routes::Dict{String, Any}
    views::Dict{String, Any}
    lock::ReentrantLock   # routes and views
    wlock::ReentrantLock  # one frame is one write
    # A Bonito.Server has a port and callers print it; a pipe has none.
    port::Int
end

mutable struct PipeConnection <: Bonito.FrontendConnection
    server::Union{Nothing, PipeServer}
    id::String
    route::String
    session::Union{Nothing, Bonito.Session}
    attached::Bool      # the page's end is connected right now
    everattached::Bool
    closed::Bool
end

# Bonito's connection interface asks for a zero-argument constructor.
PipeConnection() = PipeConnection(nothing, "", "", nothing, false, false, true)

Base.isopen(server::PipeServer) = isopen(server.io) && CURRENT[] === server

function sendframe(server::PipeServer, kind::String, id::AbstractString, body)
    bytes = body isa AbstractVector{UInt8} ? body : Vector{UInt8}(codeunits(body))
    msg = Vector{UInt8}(codeunits(string(kind, " ", id, " ", length(bytes), "\n")))
    append!(msg, bytes)
    lock(server.wlock) do
        write(server.io, msg)
        flush(server.io)
    end
    return nothing
end

# ---- the connection Bonito writes to ---------------------------------------

function Base.isopen(c::PipeConnection)
    server = c.server
    return c.attached && !c.closed && !isnothing(server) && isopen(server)
end

# Throwing while the page is away is the contract: Bonito then queues the message
# and replays it once the page reports in.
function Base.write(c::PipeConnection, bytes::AbstractVector{UInt8})
    isopen(c) || error("the plot view is not connected")
    sendframe(c.server, "MSG", c.id, bytes)
    return nothing
end

function Base.close(c::PipeConnection)
    c.closed && return nothing
    c.closed = true
    server = c.server
    isnothing(server) && return nothing
    lock(server.lock) do
        delete!(server.views, c.id)
    end
    if c.attached
        c.attached = false
        try
            sendframe(server, "CLS", c.id, UInt8[])
        catch
            # The backend is gone, and with it the page.
        end
    end
    return nothing
end

function Bonito.setup_connection(session::Bonito.Session{PipeConnection})
    c = session.connection
    url = string(c.server.prefix, "pipe/", c.id)
    compression = session.compression_enabled
    # The same handling Bonito's own WebSocket client gives a message; only the
    # address differs, and there is no reconnecting: a view lives as long as its page.
    return Bonito.js"""
    (() => {
        // The desktop shell replaces `WebSocket` with a channel to the backend,
        // from a script it runs before the page's own. It does not always get to
        // run for a new iframe's first document, so a framed page without the
        // replacement borrows its parent's (the app itself, same origin).
        let Socket = window.WebSocket;
        try {
            if (!Socket.__jutulDesktop && window.parent !== window && window.parent.WebSocket.__jutulDesktop) {
                Socket = window.parent.WebSocket;
            }
        } catch (e) {}
        const ws = new Socket(new URL($(url), window.location.href));
        // The borrowed channel belongs to the parent and would outlive this page.
        window.addEventListener("pagehide", () => ws.close());
        ws.binaryType = "arraybuffer";
        const compression = $(compression);
        // Bonito takes any answer but `true` for a failed send, and then holds
        // every later message back as if the connection had closed.
        const send = (binary) => {
            if (ws.readyState !== 1) return false;
            ws.send(binary);
            return true;
        };
        ws.onopen = () => Bonito.on_connection_open(send, compression, false);
        ws.onmessage = (evt) => {
            const binary = new Uint8Array(evt.data);
            Bonito.lock_loading(() => {
                return Promise.resolve()
                    .then(() => Bonito.process_message(Bonito.decode_binary(binary, compression)))
                    .catch((error) => Bonito.send_error("Error while processing a plot message", error));
            }, `a message of ${binary.length} bytes`);
        };
        ws.onclose = () => Bonito.on_connection_close();
    })()
    """
end

# ---- routes, with the Bonito.Server spelling --------------------------------

function (::typeof(Bonito.route!))(server::PipeServer, pair::Pair)
    lock(server.lock) do
        server.routes[String(first(pair))] = last(pair)
    end
    return server
end

function (::typeof(Bonito.delete_route!))(server::PipeServer, route)
    lock(server.lock) do
        delete!(server.routes, String(route))
    end
    return nothing
end

# ---- serving -----------------------------------------------------------------

# The auxiliary pipe's two ends, opened once per process, and the server its
# frames currently go to. One reader for the life of the process: a second `serve`
# replaces the server, not the loop.
const PIPE = Ref{Any}(nothing)
const CURRENT = Ref{Union{Nothing, PipeServer}}(nothing)

available() = !isempty(get(ENV, "JK_AUX_FDS", ""))

function pipe_ends()
    if isnothing(PIPE[])
        available() || error("this kernel was not started with a pipe for live plots")
        rfd, wfd = parse.(Int, split(ENV["JK_AUX_FDS"], ","))
        # Not for this process's own children: see the kernel's server.jl.
        if Sys.isunix()
            foreach(fd -> ccall(:fcntl, Cint, (Cint, Cint, Cint...), fd, 2, 1), (rfd, wfd))
        end
        PIPE[] = (Base.PipeEndpoint(RawFD(rfd)), Base.PipeEndpoint(RawFD(wfd)))
        # `@async`, not a spawn onto another thread: the figures belong to the eval
        # loop's thread, where Bonito's own handlers run for the HTTP server too.
        @async readloop(PIPE[][1])
    end
    return PIPE[]
end

"""
    serve(prefix) -> PipeServer

Serve live figures over the kernel's auxiliary pipe. `prefix` is the URL prefix a
page reaches this server under.
"""
function serve(prefix::AbstractString)
    _, out = pipe_ends()
    previous = CURRENT[]
    isnothing(previous) || shutdown!(previous)
    server = PipeServer(
        out, String(prefix), Dict{String, Any}(), Dict{String, Any}(),
        ReentrantLock(), ReentrantLock(), 0,
    )
    CURRENT[] = server
    sendframe(server, "HEL", "1", UInt8[])
    return server
end

function readloop(io::IO)
    try
        while true
            head = readline(io)
            isempty(head) && break  # the backend closed the pipe
            parts = split(head)
            length(parts) == 3 || continue
            kind, id = parts[1], String(parts[2])
            n = parse(Int, parts[3])
            body = n == 0 ? UInt8[] : read(io, n)
            length(body) == n || break
            server = CURRENT[]
            isnothing(server) && continue
            if kind == "GET"
                # Its own task, as a server gives each request: a figure that takes
                # a while to render must not hold up another view's messages. And
                # `invokelatest`, since this loop started before the figure's code
                # (and the methods it defined) existed.
                @async Base.invokelatest(render, server, id, String(body))
            elseif kind == "OPN"
                view = lookup(server, id)
                if !isnothing(view)
                    view.attached = true
                    view.everattached = true
                end
            elseif kind == "MSG"
                view = lookup(server, id)
                isnothing(view) || deliver(view, body)
            elseif kind == "CLS"
                view = lookup(server, id)
                isnothing(view) || detached(view)
            end
        end
    catch err
        err isa Union{Base.IOError, EOFError, InterruptException} ||
            @warn "the live plot pipe stopped" exception = (err, catch_backtrace())
    finally
        server = CURRENT[]
        isnothing(server) || shutdown!(server)
    end
    return nothing
end

lookup(server::PipeServer, id) = lock(() -> get(server.views, id, nothing), server.lock)

function deliver(view::PipeConnection, body::Vector{UInt8})
    session = view.session
    isnothing(session) && return nothing
    inbox = session.inbox
    isopen(inbox) || return nothing
    try
        put!(inbox, body)
    catch err
        # Closed between the check and the put: the session is going away.
        err isa InvalidStateException || rethrow()
    end
    return nothing
end

# The page went away. Its view is closed after the grace period, unless a newer
# view of the figure has closed it already.
function detached(view::PipeConnection)
    view.attached = false
    close_later(view, GRACE)
    return nothing
end

function close_later(view::PipeConnection, delay; unless = () -> false)
    @async begin
        sleep(delay)
        session = view.session
        if !view.closed && !unless() && !isnothing(session)
            try
                close(session)
            catch err
                @debug "closing a plot view failed" exception = err
            end
        end
    end
    return nothing
end

function render(server::PipeServer, request::String, route::String)
    app = lock(() -> get(server.routes, route, nothing), server.lock)
    if isnothing(app)
        sendframe(server, "ERR", request, "404 no such plot route")
        return nothing
    end
    retire!(server, route)
    view = PipeConnection(
        server, string(rand(UInt128); base = 16), route, nothing, false, false, false,
    )
    session = nothing
    html = try
        session = Bonito.Session(view; asset_server = Bonito.NoServer())
        view.session = session
        lock(server.lock) do
            server.views[view.id] = view
        end
        session.title = app.title
        sprint(io -> Bonito.page_html(io, session, app))
    catch err
        isnothing(session) || close(session)
        sendframe(server, "ERR", request, string("500 ", sprint(showerror, err)))
        return nothing
    end
    Bonito.mark_displayed!(session)
    sendframe(server, "HTM", request, html)
    # A page that never opens (a readiness probe, a tab closed mid-load) must not
    # keep its screen on the figure forever.
    close_later(view, 2 * GRACE; unless = () -> view.everattached)
    return nothing
end

# A figure has one view at a time: the new one takes it over from the earlier ones
# (web_views.jl), which closes their screens. WGLMakie asks an open view's page to
# dispose of its canvas and waits ten seconds for the answer, which never comes
# from a page that is unloading (a reloaded frame) or was replaced before it
# finished loading. The Bonito server never meets this, because such a page's
# socket is already closed when the next request arrives; a pipe's frames arrive
# in whatever order the two were sent. So the earlier views of a route are
# disconnected before the new one renders, and their screens close without asking.
function retire!(server::PipeServer, route::String)
    earlier = lock(server.lock) do
        [v for v in values(server.views) if v.route == route && v.attached]
    end
    for view in earlier
        view.attached = false
        try
            sendframe(server, "CLS", view.id, UInt8[])
        catch
        end
    end
    return nothing
end

function shutdown!(server::PipeServer)
    views = lock(server.lock) do
        found = collect(values(server.views))
        empty!(server.views)
        found
    end
    for view in views
        view.attached = false
        session = view.session
        try
            isnothing(session) || close(session)
        catch
        end
    end
    return nothing
end

end # module
