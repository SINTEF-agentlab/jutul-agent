# Julia kernel server (standard library only). Shipped as a resource and run as
#   julia [flags] server.jl <control_port>      with env JK_TOKEN set
#
# One loopback TCP connection carries the whole protocol as length-prefixed
# frames: an ASCII header line "TYPE [args...] NBYTES\n" followed by exactly
# NBYTES raw payload bytes. No payload ever travels as a bare line, so frame
# size is unbounded and binary-safe in both directions.
#
#   Julia -> parent:  RDY <token> 0              handshake
#                     OUT <stdout|stderr> <n>    live output bytes
#                     RES <id> <ok|err|int> <n>  one result per eval
#   parent -> Julia:  EXE <id> <n>               code to evaluate
#
# stdout/stderr are captured *in this process*: the fds are redirected to pipes
# (fd-level via dup2, so output written by C libraries is included) and pump
# tasks forward the bytes as OUT frames. After each eval the server writes a
# MARKER to both streams and sends RES only once the pumps have forwarded
# everything before it, so a result frame is always preceded by all the output
# its eval produced — ordering comes from TCP itself and the parent never has
# to stitch separate streams back together. The process's real stdout/stderr
# (the pipes the parent owns) carry only pre-handshake boot noise.
#
# Threading: the parent launches Julia with an interactive thread (`-t N,1`),
# which pins the root task — this eval loop — to the interactive thread, while
# the pumps run on the default pool. SIGINT -> InterruptException is delivered
# on the root task's thread, so a pump can never swallow an interrupt meant for
# the eval (its own sigatomic windows would otherwise eat it).

using Sockets

# Known edge: the marker is matched literally, so an eval that *prints* these
# exact bytes (two RS control characters around the tag, twice) releases the
# drain one step early. Accepted: a per-eval nonce would complicate the
# signal-atomic hold-back in the pumps for a collision no real output hits.
const MARKER = codeunits("\x1e\x1eJK-EVAL-DONE\x1e\x1e")

# Keep the process's original stdio objects alive. After the redirect nothing
# else references them, and GC would finalize them — closing the inherited
# pipes the parent tails, which reads as the process dying.
const ORIG_STDOUT = stdout
const ORIG_STDERR = stderr

# Render an error like the REPL: the message, then a backtrace whose giant
# specialized type signatures are depth-limited to `{…}`. A deeply specialized
# frame's argument types can otherwise print to tens of kilobytes;
# `:stacktrace_types_limited` is the same IOContext key the REPL uses, so this
# reuses Julia's own machinery.
function format_error(e, bt)
    try
        frames = Base.stacktrace(bt)
        # Drop the server/boot plumbing below the user's top-level eval (`none`).
        cut = findfirst(sf -> sf.file === Symbol("none"), frames)
        cut !== nothing && (frames = frames[1:cut])
        return sprint(Base.showerror, e, frames;
                      context = (:stacktrace_types_limited => Ref(false),
                                 :limit => true, :displaysize => (40, 120)))
    catch
        # Error formatting must never break the eval loop; fall back to the message.
        return try
            sprint(showerror, e)
        catch
            string(typeof(e), " (could not be displayed)")
        end
    end
end

# Byte cap on a result payload, bounding a pathologically long error message or
# value repr. Compact formatting keeps normal errors well under this, so it
# rarely fires. (Output is streamed, not framed in one piece, so it has no cap.)
const PAYLOAD_CAP = 65_536

function cap_payload(s::AbstractString)
    sizeof(s) <= PAYLOAD_CAP && return s
    # Truncate at a valid character boundary at or before the byte cap.
    idx = thisind(s, min(PAYLOAD_CAP, lastindex(s)))
    return string(SubString(s, firstindex(s), idx),
                  "\n… (output truncated; ", sizeof(s) - idx, " more bytes)")
end

Base.exit_on_sigint(false)   # SIGINT -> InterruptException instead of process exit

# A process started from a system image comes up with `Base.pkgorigins` empty,
# so `pathof`, `pkgdir` and `pkgversion` answer `nothing` for every baked
# package -- and code that locates its own scripts or resources through them
# breaks in ways that look nothing like the cause (measured: capability
# tools failing with `dirname(::Nothing)`). The sources are still on disk and
# `locate_package` still finds them, so rebuild the table once at boot. A
# normal start has origins already (the stdlibs at least) and skips this;
# packages loaded later are registered by Julia's own loading code as usual.
try
    if isempty(Base.pkgorigins)
        for (pkgid, _) in Base.loaded_modules
            pkgid.uuid === nothing && continue
            path = Base.locate_package(pkgid)
            path === nothing && continue
            origin = get!(Base.PkgOrigin, Base.pkgorigins, pkgid)
            origin.path = path
            try
                project = joinpath(dirname(dirname(path)), "Project.toml")
                if isfile(project)
                    v = get(Base.parsed_toml(project), "version", nothing)
                    v isa AbstractString && (origin.version = VersionNumber(v))
                end
            catch
            end
        end
    end
catch
end


"REPL-style text/plain repr; \"\" for nothing so the parent can omit it."
function value_repr(val)
    val === nothing && return ""
    # `invokelatest` so a `show` method defined later in the session (a newer world
    # age than this function) is honoured instead of a stale default repr.
    try
        return sprint(io -> Base.invokelatest(show,
            IOContext(io, :limit => true, :compact => false), MIME("text/plain"), val))
    catch
    end
    try
        return Base.invokelatest(string, val)
    catch
    end
    # Neither show nor string worked (e.g. an object with no text form). The type
    # is always printable and tells the caller what it got.
    return try
        string("<", typeof(val), ": value cannot be displayed as text>")
    catch
        "<value cannot be displayed as text>"
    end
end

# Julia binds a script's top-level `include` to that script's directory, so a
# user's `include("foo.jl")` would look next to this server file rather than the
# working directory (the workspace, where files are written). Rewrite top-level
# `include(x)` to `include(abspath(x))` — the cwd is the workspace — so relative
# paths resolve there. Absolute paths and nested includes are unaffected.
function rewrite_includes!(ex)
    ex isa Expr || return ex
    if ex.head === :call && length(ex.args) == 2 && ex.args[1] === :include
        ex.args[2] = Expr(:call, :abspath, ex.args[2])
    else
        foreach(rewrite_includes!, ex.args)
    end
    return ex
end

function main()
    port = parse(Int, ARGS[1])
    token = get(ENV, "JK_TOKEN", "")
    ctrl = connect(ip"127.0.0.1", port)
    # One frame = one write below; without this, a small frame after another
    # unacknowledged one would stall on Nagle + delayed ACK (~40 ms per eval).
    Sockets.nagle(ctrl, false)

    wlock = ReentrantLock()
    function sendframe(kind::AbstractString, args, body)
        bytes = body isa AbstractVector{UInt8} ? body : Vector{UInt8}(codeunits(body))
        head = isempty(args) ? string(kind, " ", length(bytes), "\n") :
               string(kind, " ", join(args, " "), " ", length(bytes), "\n")
        msg = Vector{UInt8}(codeunits(head))
        append!(msg, bytes)
        lock(wlock) do
            write(ctrl, msg)
            flush(ctrl)
        end
    end

    sendframe("RDY", (token,), UInt8[])

    # fd-level capture: from here on, everything written to fd 1/2 (by Julia or
    # by C code) lands in these pipes and is forwarded by the pumps.
    pout = Pipe(); perr = Pipe()
    redirect_stdout(pout); redirect_stderr(perr)
    drained = Dict("stdout" => Channel{Nothing}(Inf), "stderr" => Channel{Nothing}(Inf))

    function pump(p::Pipe, stream::String)
        hold = length(MARKER) - 1
        tail = UInt8[]
        finished = false
        while !finished
            try
                chunk = readavailable(p)
                if isempty(chunk)
                    finished = true
                else
                    # Bookkeeping is signal-atomic so a marker is never half
                    # consumed; a stray InterruptException lands between reads.
                    Base.disable_sigint() do
                        append!(tail, chunk)
                        while (r = findfirst(MARKER, tail)) !== nothing
                            first(r) > 1 && sendframe("OUT", (stream,), tail[1:first(r)-1])
                            deleteat!(tail, 1:last(r))
                            put!(drained[stream], nothing)
                        end
                        n = length(tail) - hold   # hold back a possible marker prefix
                        if n > 0
                            sendframe("OUT", (stream,), tail[1:n])
                            deleteat!(tail, 1:n)
                        end
                    end
                end
            catch e
                e isa InterruptException || rethrow()
            end
        end
    end
    Threads.@spawn pump(pout, "stdout")
    Threads.@spawn pump(perr, "stderr")

    # Close the eval's output: push everything (including C stdio buffers) into
    # the pipes, then wait until the pumps have forwarded up to the markers.
    function drain()
        flush(stdout); flush(stderr); Libc.flush_cstdio()
        write(stdout, MARKER); flush(stdout)
        write(stderr, MARKER); flush(stderr)
        take!(drained["stdout"]); take!(drained["stderr"])
    end

    while true
        local head
        try
            head = readline(ctrl)
        catch e
            e isa InterruptException && continue   # stray interrupt between evals
            rethrow()
        end
        if head == ""
            eof(ctrl) && break                     # parent closed the control socket
            continue
        end
        parts = split(head)
        length(parts) == 3 && parts[1] == "EXE" || continue
        id = parts[2]
        nbytes = parse(Int, parts[3])
        # The body read must not be torn by a stray interrupt: a half-read frame
        # would desync the protocol. A deferred interrupt surfaces in the eval.
        code = String(Base.disable_sigint() do; read(ctrl, nbytes) end)

        status = "ok"
        payload = ""
        try
            val = Core.eval(Main, rewrite_includes!(Meta.parseall(code)))
            payload = value_repr(val)
        catch e
            if e isa InterruptException
                status = "int"
            else
                status = "err"
                payload = format_error(e, catch_backtrace())
            end
        end
        # Atomic w.r.t. interrupts: a late SIGINT can't tear output from result.
        Base.disable_sigint() do
            drain()
            sendframe("RES", (id, status), cap_payload(payload))
        end
    end
end

main()
