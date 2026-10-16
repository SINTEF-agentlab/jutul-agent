# A browser connection owns its screen, polling and input callbacks. The figure
# outlives connections so reloading or closing one view cannot destroy another.
# WGLMakie 0.13.16 closes the shared scene and deletes shared render objects when
# one screen closes (MakieOrg/Makie.jl#2957). Keep that lifecycle in our own types;
# no WGLMakie methods are replaced. Older backends retain their standard renderer.
#
# Canvas initialization below follows WGLMakie 0.13.16's three_plot.jl, MIT license,
# copyright Makie contributors: https://github.com/MakieOrg/Makie.jl/blob/v0.24.16/LICENSE.md
module JutulAgentWebViews
import WGLMakie
const Makie = WGLMakie.Makie
const Bonito = WGLMakie.Bonito
using WGLMakie.Bonito: @js_str

mutable struct LiveFigure
    fig::Makie.Figure
    config::WGLMakie.ScreenConfig
    screens::Vector{Makie.MakieScreen}
    released::Bool
end
function LiveFigure(fig::Makie.Figure; kwargs...)
    Makie.update_state_before_display!(fig)
    config = Makie.merge_screen_config(WGLMakie.ScreenConfig, Dict{Symbol, Any}(kwargs))
    return LiveFigure(fig, config, Makie.MakieScreen[], false)
end
mutable struct LiveScreen <: Makie.MakieScreen
    backend::WGLMakie.Screen
    owner::LiveFigure
    closed::Bool
    initialized::Bool
    dirty::Bool
end
import ..JutulAgentPlots: backend_screen
backend_screen(screen::LiveScreen) = screen.backend
Base.isopen(s::LiveScreen) = !s.closed && isopen(s.backend)
Base.size(s::LiveScreen) = size(s.backend)
Base.resize!(s::LiveScreen, width::Integer, height::Integer) = resize!(s.backend, width, height)
Makie.colorbuffer(s::LiveScreen) = Makie.colorbuffer(s.backend)

function Base.close(s::LiveScreen)
    s.closed && return
    s.closed = true
    Makie.stop!(s.backend.tick_clock)
    stack = [s.owner.fig.scene]
    while !isempty(stack)
        scene = pop!(stack)
        filter!(x -> x !== s, scene.current_screens)
        append!(stack, scene.children)
    end
    filter!(x -> x !== s, s.owner.screens)
    return nothing
end
Base.empty!(s::LiveScreen) = close(s)

function Base.insert!(s::LiveScreen, scene::Makie.Scene, plot::Makie.AbstractPlot)
    if !s.closed && s.initialized && isopen(s)
        insert!(s.backend, scene, plot)
    else
        s.dirty = true
    end
    return nothing
end
function Base.delete!(s::LiveScreen, scene::Makie.Scene, plot::Makie.AbstractPlot)
    if !s.closed && s.initialized && isopen(s)
        delete!(s.backend, scene, plot)
    else
        s.dirty = true
    end
    return nothing
end
function Base.delete!(s::LiveScreen, scene::Makie.Scene)
    if !s.closed && s.initialized && isopen(s)
        delete!(s.backend, scene)
    else
        s.dirty = true
    end
    return nothing
end

function release!(view::LiveFigure)
    view.released && return
    view.released = true
    for s in copy(view.screens)
        close(s)
        close(s.backend.session)
    end
    return nothing
end

function connect_events!(s::LiveScreen, comm)
    scene = s.owner.fig.scene
    e = Makie.events(scene)
    Bonito.on(s.backend.session, comm) do msg
        @async try
            isopen(s) || return
            if haskey(msg, "window_open") && !msg["window_open"]
                close(s)
                return
            end
            if haskey(msg, "mouseposition")
                e.mouseposition[] = Tuple(Float64.(msg["mouseposition"]))
            end
            for kind in ("mousedown", "mouseup")
                haskey(msg, kind) || continue
                for (mask, button) in ((1, Makie.Mouse.left), (2, Makie.Mouse.right), (4, Makie.Mouse.middle))
                    pressed = msg[kind] & mask != 0
                    if pressed != (button in e.mousebuttonstate)
                        action = pressed ? Makie.Mouse.press : Makie.Mouse.release
                        e.mousebutton[] = Makie.MouseButtonEvent(button, action)
                    end
                end
            end
            haskey(msg, "scroll") && (e.scroll[] = Tuple(Float64.(sign.(msg["scroll"]))))
            if haskey(msg, "keydown")
                key = msg["keydown"]
                button = WGLMakie.code_to_keyboard(key[1])
                button != Makie.Keyboard.unknown && (e.keyboardbutton[] = Makie.KeyEvent(button, Makie.Keyboard.press))
                length(key[2]) == 1 && isascii(key[2]) && (e.unicode_input[] = key[2][1])
            end
            if haskey(msg, "keyup")
                key = msg["keyup"]
                buttons = key == "delete_keys" ? collect(e.keyboardstate) : [WGLMakie.code_to_keyboard(key)]
                for button in buttons
                    e.keyboardbutton[] = Makie.KeyEvent(button, Makie.Keyboard.release)
                end
            end
            haskey(msg, "resize") && Makie.resize!(scene, Tuple(msg["resize"]))
        catch err
            @warn "Error in web plot input" exception = (err, catch_backtrace())
        end
        return nothing
    end
end

function canvas(s::LiveScreen, session, scene)
    config = s.backend.config
    order = WGLMakie.get_order!(session)
    initial_size = size(scene)
    canvas_width = Makie.lift(x -> [round.(Int, Makie.widths(x))...], scene, Makie.viewport(scene))
    real_size = Makie.Observable{Any}(nothing)
    serialized = Makie.Observable{Union{Nothing, Dict{Symbol, Any}}}(nothing)
    done_init = Makie.Observable{Any}(nothing)
    Bonito.on(session, real_size) do size_arr
        @async try
            s.closed && return
            dimensions = Tuple(round.(Int, size_arr))
            dimensions != size(scene) && Makie.resize!(scene, dimensions...)
            serialized[] = WGLMakie.serialize_scene(scene)
            s.dirty = false
        catch err
            done_init[] = err
        end
        return nothing
    end
    width, height = initial_size
    element = Bonito.DOM.m("canvas"; tabindex = "0", width = "$(width)px", height = "$(height)px",
        style = "display: block", dataLmSuppressShortcuts = true, dataJpSuppressContextMenu = nothing)
    wrapper = Bonito.DOM.div(element, config.spinner; style = "width: 100%; height: 100%; position: relative;")
    comm = Makie.Observable(Dict{String, Any}())
    Bonito.evaljs(session, js"""
        $(WGLMakie.WGL).then(WGL => WGL.execute_in_order($order, () => {
            WGL.setup_scene_init($wrapper, $element, $width, $height,
                $(config.resize_to), $(config.px_per_unit), $(config.scalefactor),
                $real_size, $canvas_width, $serialized, $comm, $(config.framerate), $done_init);
        }))
    """)
    connect_events!(s, comm)
    return wrapper, done_init
end

function poll_view!(s::LiveScreen)
    scene = s.owner.fig.scene
    tick = Makie.TickCallback(scene.events.tick)
    @async try
        while !s.closed && !s.owner.released && !Bonito.isclosed(s.backend.session)
            if s.initialized && isopen(s)
                if s.dirty
                    # A disconnected view missed structural scene changes. Reload
                    # it from the current figure instead of replaying stale JS ids.
                    Bonito.evaljs(s.backend.session, js"window.location.reload()")
                    break
                end
                WGLMakie.poll_all_plots(scene)
                if !s.backend.tick_clock.running
                    Makie.start!(s.backend.tick_clock) do _
                        isopen(s) && !s.closed && tick(Makie.RegularRenderTick)
                        return nothing
                    end
                end
            else
                Makie.stop!(s.backend.tick_clock)
            end
            # A superseded disconnected page will never need its screen again.
            # A lone disconnected view may reconnect and resume its observables.
            if s.backend.session.status == Bonito.SOFT_CLOSED &&
                    any(other -> other !== s && isopen(other), s.owner.screens)
                close(s.backend.session)
                break
            end
            sleep(isopen(s) ? 0.01 : 0.25)
        end
    catch err
        @warn "Error updating web plot" exception = (err, catch_backtrace())
    finally
        close(s)
    end
    return nothing
end

function Bonito.jsrender(session::Bonito.Session, view::LiveFigure)
    view.released && error("This plot has been released")
    if !(v"0.13.16" <= Base.pkgversion(WGLMakie) < v"0.14")
        return Bonito.jsrender(session, WGLMakie.WithConfig(view.fig, view.config))
    end
    scene = view.fig.scene
    scene.isclosed = false
    screen = WGLMakie.Screen(scene, view.config)
    screen.session = session
    s = LiveScreen(screen, view, false, false, false)
    push!(view.screens, s)
    Makie.push_screen!(scene, s)
    try
        wrapper, on_init = canvas(s, session, scene)
        screen.canvas = wrapper
        if screen.config.use_html_widgets
            push!(Bonito.children(wrapper), WGLMakie.replace_widget!(view.fig))
        end
        Bonito.on(session, on_init) do initialized
            s.closed && return
            isready(screen.plot_initialized) && return
            put!(screen.plot_initialized, initialized)
            if initialized === true
                s.initialized = true
                scene.isclosed = false
                scene.events.window_open[] = true
                WGLMakie.mark_as_displayed!(screen, scene)
            else
                close(s)
            end
            return nothing
        end
        Bonito.on(session, session.on_close) do closed
            closed && close(s)
            return nothing
        end
        # Monitor connecting views too: a reload can abandon them before on_init.
        poll_view!(s)
        return Bonito.jsrender(session, wrapper)
    catch
        close(s)
        rethrow()
    end
end
end
