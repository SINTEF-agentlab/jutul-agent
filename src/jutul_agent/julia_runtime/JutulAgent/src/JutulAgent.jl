module JutulAgent

# jutul-agent's simulator-agnostic Julia runtime: figure capture (plots.jl),
# ensemble helpers (ensemble.jl), and a generic-Makie warm-up. The per-simulator
# solve/plot warm-up lives in the JutulAgent<Sim> packages. See
# docs/architecture.md ("Simulators are data").

using PrecompileTools: @compile_workload, @setup_workload

import CairoMakie
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

@setup_workload begin
    @compile_workload begin
        CairoMakie.activate!()
        mktempdir() do dir
            CairoMakie.save(joinpath(dir, "warm-cairo.png"), _warm_draw(CairoMakie))
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
