"""Browser views of live web figures: a new connection takes the figure over.

WGLMakie 0.13 closes a figure's scene and deletes its cached render data when any of
its views closes (MakieOrg/Makie.jl#2957), and a later view gets each plot's first
render plus only the glyphs an earlier browser had not seen. Bonito closes a reloaded
page's old session about 30 s later, so a reloaded plot froze. The web UI keeps one
frame per live figure, so a new view closes the earlier ones now and drops the cached
render data, then renders from the figure's current values.
"""
module JutulAgentWebViews

import WGLMakie
const Makie = WGLMakie.Makie
const Bonito = WGLMakie.Bonito

# The WGLMakie internals used here are unchanged from 0.13.13 through 0.13.16.
supported() = v"0.13.13" <= Base.pkgversion(WGLMakie) < v"0.14"

"""
    view(session, fig; screen_config...)

Render `fig` into a browser `session`, taking it over from any earlier view.
"""
function view(session::Bonito.Session, fig::Makie.Figure; screen_config...)
    supported() || return WGLMakie.WithConfig(fig; screen_config...)
    scene = fig.scene
    for screen in copy(scene.current_screens)
        screen isa WGLMakie.Screen || continue
        old = screen.session
        close(screen)
        # Without a scene, the old polling loop's own close leaves the figure alone.
        empty!(screen)
        # A stale screen on a child scene would make later plot inserts throw.
        detach!(scene, screen)
        # Its delayed close would otherwise close the scene under the new view.
        isnothing(old) || close(Bonito.root_session(old))
    end
    Makie.for_each_atomic_plot(scene) do plot
        WGLMakie.delete_wgl_robj!(nothing, plot)
        if haskey(plot.attributes, :glyph_data)
            delete!(plot.attributes, :glyph_data; recursive = true)
        end
    end
    # Not WithConfig, which resets axis limits on every view: the figure is laid out
    # once when it is served, so a reload keeps the user's zoom.
    config = Makie.merge_screen_config(WGLMakie.ScreenConfig, Dict{Symbol, Any}(screen_config))
    return WGLMakie.render_with_init(WGLMakie.Screen(scene, config), session, scene, fig)
end

function detach!(scene, screen)
    filter!(s -> s !== screen, scene.current_screens)
    foreach(child -> detach!(child, screen), scene.children)
    return nothing
end

end # module
