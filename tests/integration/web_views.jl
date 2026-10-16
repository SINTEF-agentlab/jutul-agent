using Test
import JutulAgent, WGLMakie

@testset "browser view lifetimes" begin
    views = JutulAgent.JutulAgentWebViews
    plots = JutulAgent.JutulAgentPlots
    M = WGLMakie.Makie
    B = WGLMakie.Bonito
    @test v"0.13.15" <= Base.pkgversion(WGLMakie) < v"0.14"
    WGLMakie.activate!()
    fig = M.Figure()
    ax = M.Axis(fig[1, 1])
    line = M.lines!(ax, 1:3)
    label = M.text!(ax, [M.Point2f(1, 1)]; text = ["Step 1/3"])
    view = views.LiveFigure(fig; resize_to = :parent)
    session1, session2 = B.Session(), B.Session()
    try
        # A slow pick must not let a later release overtake the preceding press.
        incoming = M.Observable(0)
        received = Int[]
        queue = views.on_serially(session1, incoming) do value
            value == 1 && sleep(0.05)
            push!(received, value)
        end
        incoming[] = 1
        incoming[] = 2
        @test timedwait(() -> length(received) == 2, 5.0) == :ok
        @test received == [1, 2]
        B.jsrender(session1, view)
        B.jsrender(session2, view)
        @test length(view.screens) == 2
        first, second = view.screens
        @test all(s -> s in ax.scene.current_screens, view.screens)
        @test plots.backend_screen(second) === second.backend

        # Serialization owns render objects on the plot, shared by both views.
        # Closing one connection must leave those objects and the scene alive.
        data1 = views.serialize_scene(first, fig.scene)
        data2 = views.serialize_scene(second, fig.scene)
        B.serialize_cached(B.SerializationContext(session1), data1)
        B.serialize_cached(B.SerializationContext(session2), data2)
        object = line.attributes[:wgl_renderobject][]
        updater = line.attributes[:wgl_update_obs][]
        proxy1 = first.bindings[updater][1]
        proxy2 = second.bindings[updater][1]
        @test proxy1 !== proxy2 !== updater
        @test !any(last(x) isa B.JSUpdateObservable for x in updater.listeners)
        old_color = views.plot_snapshot(second, ax.scene, line)[:uniforms][:line_color]
        session1.status = B.SOFT_CLOSED
        added = M.lines!(ax, [3, 2, 1])
        @test first.dirty
        @test added in ax.scene.plots
        close(session1)
        @test !isopen(queue)
        @test first.closed
        @test !second.closed
        @test first ∉ fig.scene.current_screens
        @test first ∉ ax.scene.current_screens
        @test second in ax.scene.current_screens
        @test !M.isclosed(fig.scene)
        @test line.attributes[:wgl_renderobject][] === object
        @test isempty(first.bindings)
        @test isempty(first.callbacks)
        @test any(last(x) isa B.JSUpdateObservable for x in proxy2.listeners)
        line.color = :red
        WGLMakie.poll_all_plots(fig.scene)
        @test proxy2[] isa B.LargeUpdate
        new_color = views.plot_snapshot(second, ax.scene, line)[:uniforms][:line_color]
        @test new_color != old_color
        # Opening a view after a text update must include every required glyph,
        # even when an earlier browser already received those glyphs.
        label.text = ["Step 3/3"]
        WGLMakie.poll_all_plots(fig.scene)
        glyphs = views.plot_snapshot(second, ax.scene, label)[:glyph_data]
        @test all(h in second.glyphs for h in glyphs[:glyph_hashes])
        @test isempty(first.glyphs)
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
        @test isempty(second.bindings)
        @test isempty(second.callbacks)
        @test_throws ErrorException B.jsrender(B.Session(), view)
    finally
        views.release!(view)
        close(session1)
        close(session2)
    end
end
