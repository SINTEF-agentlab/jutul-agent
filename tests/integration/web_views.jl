using Test
import JutulAgent, WGLMakie

@testset "browser view lifetimes" begin
    views = JutulAgent.JutulAgentWebViews
    plots = JutulAgent.JutulAgentPlots
    M = WGLMakie.Makie
    B = WGLMakie.Bonito
    @test v"0.13.16" <= Base.pkgversion(WGLMakie) < v"0.14"
    WGLMakie.activate!()
    fig = M.Figure()
    ax = M.Axis(fig[1, 1])
    line = M.lines!(ax, 1:3)
    view = views.LiveFigure(fig; resize_to = :parent)
    session1, session2 = B.Session(), B.Session()
    try
        B.jsrender(session1, view)
        B.jsrender(session2, view)
        @test length(view.screens) == 2
        first, second = view.screens
        @test all(s -> s in ax.scene.current_screens, view.screens)
        @test plots.backend_screen(second) === second.backend

        # Serialization owns render objects on the plot, shared by both views.
        # Closing one connection must leave those objects and the scene alive.
        WGLMakie.serialize_scene(fig.scene)
        object = line.attributes[:wgl_renderobject][]
        session1.status = B.SOFT_CLOSED
        added = M.lines!(ax, [3, 2, 1])
        @test first.dirty
        @test added in ax.scene.plots
        close(session1)
        @test first.closed
        @test !second.closed
        @test first ∉ fig.scene.current_screens
        @test first ∉ ax.scene.current_screens
        @test second in ax.scene.current_screens
        @test !M.isclosed(fig.scene)
        @test line.attributes[:wgl_renderobject][] === object
        close(first)
        @test view.screens == [second]
        M.delete!(ax.scene, added)
        @test added ∉ ax.scene.plots

        views.release!(view)
        views.release!(view)
        @test view.released
        @test isempty(view.screens)
        @test isempty(fig.scene.current_screens)
        @test isempty(ax.scene.current_screens)
        @test B.isclosed(session2)
        @test_throws ErrorException B.jsrender(B.Session(), view)
    finally
        views.release!(view)
        close(session1)
        close(session2)
    end
end
