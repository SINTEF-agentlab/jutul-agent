module JutulAgent

# jutul-agent's simulator-agnostic Julia runtime: figure capture (plots.jl),
# ensemble helpers (ensemble.jl), and a generic-Makie warm-up. The per-simulator
# solve/plot warm-up lives in the JutulAgent<Sim> packages. See
# docs/architecture.md ("Simulators are data").

using PrecompileTools: @compile_workload, @setup_workload

# Every Makie backend a session can hold at once, so the workload below bakes in the
# world it will run in. Loading a backend invalidates code specialised against the
# ones already loaded, so a backend that arrives after the bake discards it. GLMakie
# last: a backend activates itself on load, and the workload wants GLMakie current.
import CairoMakie
import WGLMakie
import GLMakie

include("ensemble.jl")   # submodule JutulAgentEnsemble (Distributed addprocs + pmap)
include("plots.jl")      # submodule JutulAgentPlots  (GLMakie figure capture)

using .JutulAgentEnsemble: run_ensemble, warm_addprocs
export run_ensemble, warm_addprocs

# A tiny 2D figure exercising the lines!/scatter! + save path both Makie backends
# share. CairoMakie warms the Makie core headlessly; GLMakie (the backend julia_plot
# drives) needs a GL context for its offscreen save, so it is wrapped and skipped
# when none is available.
function _warm_draw(Backend)
    fig = Backend.Figure(size = (96, 96))
    ax = Backend.Axis(fig[1, 1])
    Backend.lines!(ax, 1:3, [1.0, 2.0, 1.5])
    Backend.scatter!(ax, 1:3, [1.0, 2.0, 1.5])
    return fig
end

# The shapes a saved PNG is made of: a labelled 2D axis with a legend (rate curves)
# and a 3D axis with a colorbar (a reservoir). CairoMakie renders every web poster,
# and its own workload -- which used to cover these draw paths -- is off on Windows,
# where it does not fit (jutul_agent.sysimage_build.WINDOWS_ENV_PREFERENCES). So bake
# what we save rather than the general case.
function _warm_poster(Backend)
    fig = Backend.Figure(size = (96, 96))
    ax = Backend.Axis(fig[1, 1], title = "t", xlabel = "x", ylabel = "y")
    Backend.lines!(ax, 1:3, [1.0, 2.0, 1.5], label = "a")
    Backend.scatter!(ax, 1:3, [1.0, 2.0, 1.5], label = "b")
    Backend.axislegend(ax)
    ax3 = Backend.Axis3(fig[2, 1])
    plot = Backend.surface!(ax3, 1:4, 1:4, [Float64(i + j) for i in 1:4, j in 1:4])
    Backend.Colorbar(fig[2, 2], plot)
    return fig
end

@setup_workload begin
    @compile_workload begin
        CairoMakie.activate!()
        mktempdir() do dir
            CairoMakie.save(joinpath(dir, "warm-cairo.png"), _warm_draw(CairoMakie))
            CairoMakie.save(joinpath(dir, "warm-poster.png"), _warm_poster(CairoMakie))
        end
        try
            GLMakie.activate!(visible = false)
            mktempdir() do dir
                fig = _warm_draw(GLMakie)
                GLMakie.save(joinpath(dir, "warm-gl.png"), fig)
                # Warm the capture path the plot tool actually drives (offscreen).
                JutulAgentPlots.capture(fig; path = joinpath(dir, "warm-capture.png"))
            end
        catch
            # No GL context at precompile time (headless without xvfb): CairoMakie
            # already warmed the shared Makie core; skip the GL bake.
        end
    end
end

end # module
