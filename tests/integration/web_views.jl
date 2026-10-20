using Test
import JutulAgent, WGLMakie

const V = JutulAgent.JutulAgentWebViews
const M = WGLMakie.Makie
const B = WGLMakie.Bonito

wgl_screens(scene) = filter(s -> s isa WGLMakie.Screen, scene.current_screens)

@testset "a new browser view takes the figure over" begin
    # CI must exercise the take-over, not the fallback for other WGLMakie versions.
    @test V.supported()
    WGLMakie.activate!()
    fig = M.Figure()
    ax = M.Axis(fig[1, 1])
    line = M.lines!(ax, 1:3)
    label = M.text!(ax, [M.Point2f(1, 1)]; text = ["Step 1"])
    first, second = B.Session(), B.Session()
    try
        V.view(first, fig)
        old = only(wgl_screens(fig.scene))
        # Serialize as the first browser does, then change the text so that browser
        # holds glyphs the next one has never received.
        WGLMakie.serialize_scene(fig.scene)
        updates = line.attributes[:wgl_update_obs][]
        label.text = ["Step 3 of 9"]
        WGLMakie.poll_all_plots(fig.scene)

        # A reload: the old page is closed now rather than after Bonito's grace
        # period, and no child scene keeps its screen.
        V.view(second, fig)
        @test B.isclosed(first)
        @test only(wgl_screens(fig.scene)).session === second
        @test old ∉ ax.scene.current_screens

        # The new view starts from current values, with its own update channel and
        # every glyph of the current text.
        WGLMakie.serialize_scene(fig.scene)
        @test line.attributes[:wgl_update_obs][] !== updates
        glyphs = label.attributes[:glyph_data][]
        @test Set(keys(glyphs[:atlas_updates])) ⊇ Set(glyphs[:glyph_hashes])
    finally
        close(first)
        close(second)
    end
end

@testset "dropping a route closes its page" begin
    WGLMakie.activate!()
    fig = M.Figure()
    M.lines!(M.Axis(fig[1, 1]), 1:3)
    page = B.Session()
    try
        V.view(page, fig)
        figures = Dict{String, Any}("/viz/a" => fig, "/viz/b" => fig)
        V.drop!(figures, "/viz/a")
        @test !B.isclosed(page)  # still served at /viz/b
        V.drop!(figures, "/viz/b")
        @test isempty(figures)
        @test B.isclosed(page)
        # WGLMakie's own close handler ran, which stops the view's updates.
        @test M.isclosed(fig.scene)
    finally
        close(page)
    end
end
