# Called after loading the shared plotting helpers. No native GL window is needed.
using Test
import WGLMakie, GLMakie, Bonito
const WebCaptureMakie = WGLMakie.Makie

@testset "web figure capture" begin
    M = WebCaptureMakie
    helpers = JutulAgent.JutulAgentPlots
    original_displays = copy(Base.Multimedia.displays)
    original_inline = M.ALWAYS_INLINE_PLOTS[]
    original_gl_screens = copy(GLMakie.ALL_SCREENS)
    try
        # Even a session configured to open windows must use the capture display.
        M.inline!(false)
        first = M.Figure()
        axis = M.Axis(first[1, 1])
        line = M.lines!(axis, 1:3; transparency = true, alpha = 0.9)
        d = helpers.start_web_capture()
        try
            @test M.current_backend() === WGLMakie
            display(first)
            @test isempty(first.scene.current_screens)
            @test isempty(axis.scene.current_screens)
            # Inserting after display used to target an uninitialized WGL screen.
            M.scatter!(axis, 1:3)
            display(first)
            M.Figure()  # another current figure must not displace the displayed one
            @test helpers.web_figure(d, nothing) === first
            @test helpers.web_figure(d, (; fig = first)) === first
            @test helpers.web_figure(d, (first, axis, line)) === first
            @test line.transparency[] === true
            @test line.alpha[] ≈ 0.9
        finally
            helpers.stop_web_capture(d)
        end
        @test Base.Multimedia.displays == original_displays
        @test M.ALWAYS_INLINE_PLOTS[] === false
        @test GLMakie.ALL_SCREENS == original_gl_screens

        # Reusing the stale current figure after a non-plot call is an error.
        d = helpers.start_web_capture()
        try
            @test_throws ErrorException helpers.web_figure(d, 42)
        finally
            helpers.stop_web_capture(d)
        end
        @test Base.Multimedia.displays == original_displays
        @test M.ALWAYS_INLINE_PLOTS[] === false

        # A connecting live screen must not be mistaken for a capture-only screen.
        screen = WGLMakie.Screen(first.scene)
        session = Bonito.Session()
        screen.session = session
        M.push_screen!(first.scene, screen)
        d = helpers.start_web_capture()
        try
            display(d, first)
            @test screen in first.scene.current_screens
            @test screen in axis.scene.current_screens
        finally
            helpers.stop_web_capture(d)
            close(session)
        end
    finally
        M.inline!(original_inline)
    end
end
