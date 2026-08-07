"""GLMakie figure capture for jutul-agent's julia_plot tool.

A submodule of the JutulAgent package (precompiled with it); the plot tool drives
it as `JutulAgent.JutulAgentPlots.capture(...)` after `using JutulAgent`. Before
each plot it activates GLMakie (visible for an interactive window, offscreen
otherwise), evaluates the user's expression, and hands the result to capture.

capture resolves a Makie Figure from whatever the plotter produced: a returned
Figure or FigureAxisPlot, a (fig, ax, plot) tuple, or, for plotters that open a
window or call display and return an axis/screen/nothing, the figure Makie just
drew (current_figure). It then saves that figure with GLMakie.
"""

module JutulAgentPlots

import GLMakie
const Makie = GLMakie.Makie  # Makie.save dispatches on the active GLMakie backend

export capture, recapture, close_windows

# Interactive windows keyed by a caller-chosen string (the plot's slot). The same
# key refreshes that window in place; a new key opens a new window. Recapturing a
# key re-saves its Figure, so a window the user rotated yields their current view.
const SCREENS = Dict{String, Any}()   # key -> GLMakie screen
const FIGURES = Dict{String, Any}()   # key -> Figure
const LAST_KEY = Ref{String}("")      # most recently shown key (recapture default)

"""Return true if fig has no content blocks (axes, scenes, labels)."""
is_empty_figure(fig::Makie.Figure) = isempty(fig.content)

"""Current Makie figure, or nothing (never throws). Snapshot this before running a
plot expression to tell a freshly drawn figure from a stale one."""
function _current_fig()
    f = try
        Makie.current_figure()
    catch
        nothing
    end
    return f isa Makie.Figure ? f : nothing
end

"""Resolve a Makie Figure from the value a plot expression evaluated to.

A returned Figure, FigureAxisPlot, or (fig, ax, plot) tuple is used directly.
Otherwise we fall back to the figure Makie just drew (current_figure), which is how
plotters that open a window or call display surface theirs. prev is current_figure()
from before the expression ran: if the current figure is unchanged from prev,
nothing new was drawn (a non-figure return value, or a plotter that logged and drew
nothing), so we return nothing and the caller reports an honest error rather than
saving a stale, unrelated figure under this slot."""
function _as_figure(x, prev = nothing)
    x isa Makie.Figure && return x
    x isa Makie.FigureAxisPlot && return x.figure
    if x isa Tuple && length(x) >= 1 && x[1] isa Makie.Figure   # plot_cell_data / plot_mesh
        return x[1]
    end
    cf = _current_fig()
    (cf === nothing || cf === prev) && return nothing
    return cf
end

function _ensure_parent_dir(path::AbstractString)
    d = dirname(path)
    (!isempty(d) && !isdir(d)) && mkpath(d)
end

function _save_kwargs(size, dpi)
    kwargs = Pair{Symbol, Any}[]
    if size !== nothing && size isa Tuple && length(size) == 2
        push!(kwargs, :size => (Int(size[1]), Int(size[2])))
    end
    dpi !== nothing && push!(kwargs, :dpi => Int(dpi))
    return kwargs
end

"""Save a Makie Figure to path. `Makie.save` dispatches on the active backend and
picks the format from the extension; the tool only ever passes `.png` (GLMakie).

`update` is Makie's own keyword, forwarded so a caller can turn it off. It defaults
to true there, and resets every axis to its automatic limits before saving, which
is right for a figure being drawn for the first time and wrong for one already on
screen: it discards whatever the user zoomed to."""
function save_figure(
        fig::Makie.Figure; path::AbstractString, size = nothing, dpi = nothing,
        update::Bool = true,
    )
    _ensure_parent_dir(path)
    Makie.save(path, fig; update = update, _save_kwargs(size, dpi)...)
    return path
end

"""Resolve a Makie Figure from value and save it to path. When open_window is set
(an interactive session) the figure is also shown in a live window keyed by
window_key; the file is written either way. prev_figure is current_figure() from
before the plot code ran, used to reject a stale fallback (see _as_figure)."""
function capture(
    value;
    path::AbstractString,
    size = nothing,
    dpi = nothing,
    open_window::Bool = false,
    window_key = "",
    prev_figure = nothing,
)
    fig = _as_figure(value, prev_figure)
    fig === nothing && error(
        "julia_plot: the code did not produce a Makie figure. Return a Figure, or " *
        "call a native plotter that builds one (it may also call display()). If you " *
        "called a plotter and the output above shows a 'No plottable …' notice, it " *
        "drew nothing for this model; use a plotter that fits, or build the figure " *
        "inline with Figure()/Axis().",
    )
    is_empty_figure(fig) &&
        error("julia_plot: the Figure is empty (no axes or scene); nothing was drawn.")
    if open_window
        try  # best-effort live window; a failed display must not abort the save
            _show_window(fig, String(window_key))
        catch
        end
    end
    return save_figure(fig; path = path, size = size, dpi = dpi)
end

"""Show fig in the window for key: refresh it if still open, else open a fresh
GLMakie Screen. A distinct key is a distinct window; plain display(fig) would
reuse GLMakie's single main screen. invokelatest bridges the world-age gap to
methods added by native plotters loaded after this module."""
function _show_window(fig, key::AbstractString)
    FIGURES[key] = fig   # register first so recapture/close always find it
    LAST_KEY[] = key
    # A native plotter (e.g. plot_reservoir) may open its own window, binding the
    # scene to a screen. GLMakie forbids one scene in two screens, so adopt it.
    own = Base.invokelatest(Makie.getscreen, fig.scene)
    if own !== nothing
        SCREENS[key] = own
        return nothing
    end
    existing = get(SCREENS, key, nothing)
    if existing !== nothing && Base.invokelatest(isopen, existing)
        Base.invokelatest(display, existing, fig)        # refresh in place
    else
        screen = Base.invokelatest(GLMakie.Screen)
        Base.invokelatest(display, screen, fig)          # new window
        SCREENS[key] = screen
    end
    return nothing
end

"""Flush the live window for key so a following save captures the user's current
view, not a stale frame.

A user's rotate (cam3d) or recolor (a Menu) sits in GLFW's queue while the worker
is blocked between evals, and those controllers only apply on the render tick
after their event, so a single save renders the previous state. Two colorbuffer
cycles fix that: the first drains the queued events into the controllers, the
second renders their applied state. No-op when key has no open screen.

Nothing here may update the figure's state. `update_state_before_display!` reads
like the way to flush it, but what it does is reset every axis to its automatic
limits, which throws away the zoom this is being called to preserve. Rotation
survived it because a 3D camera is not held in the axis limits."""
function _refresh_live_view(key::AbstractString, fig)
    scr = get(SCREENS, key, nothing)
    scr === nothing && return nothing
    try
        Base.invokelatest(isopen, scr) || return nothing
    catch
        return nothing
    end
    for _ in 1:2
        try
            Base.invokelatest(Makie.colorbuffer, scr)
        catch
        end
    end
    return nothing
end

"""Re-save an open interactive window at its current camera/zoom/timestep. key
selects the window (a plot's slot); empty means the most recently opened or
refreshed one. Errors if there is no such open window.

Saves the window exactly as it stands. Passing a size would resize it, since Makie
resizes the scene to match and never puts it back, and letting the save update the
figure would reset the axes; either one edits the view this is meant to record."""
function recapture(; key::AbstractString = "", path)
    k = isempty(key) ? LAST_KEY[] : String(key)
    fig = get(FIGURES, k, nothing)
    fig isa Makie.Figure || error(
        "recapture: no interactive window" * (isempty(k) ? "" : " '$k'") *
        " is open. Open one first with julia_plot (give it a slot to recapture it later).",
    )
    _refresh_live_view(k, fig)   # render queued interactions before saving
    return save_figure(fig; path = path, update = false)
end

"""Close the interactive window for key, or all of them when key is empty."""
function close_windows(key::AbstractString = "")
    if isempty(key)
        try
            Base.invokelatest(GLMakie.closeall)
        catch
        end
        empty!(SCREENS)
        empty!(FIGURES)
        LAST_KEY[] = ""
    else
        scr = get(SCREENS, String(key), nothing)
        scr === nothing || (try
            Base.invokelatest(close, scr)
        catch
        end)
        delete!(SCREENS, String(key))
        delete!(FIGURES, String(key))
    end
    return nothing
end

end # module
