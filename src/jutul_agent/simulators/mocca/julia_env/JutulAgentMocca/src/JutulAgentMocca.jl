module JutulAgentMocca

# Per-simulator warm-up package for Mocca, which ships no PrecompileTools workload.
# Bakes the process simulation and native bed/stream plots. See JutulAgentJutulDarcy
# for the @recompile_invalidations rationale.

using Mocca, Jutul
using PrecompileTools: @recompile_invalidations, @setup_workload, @compile_workload

@recompile_invalidations begin
    import JutulAgent
    import CairoMakie
    import WGLMakie
    using GLMakie
end

function _warm_solve()
    process = Mocca.four_stage_vsa_process(ncells = 10)
    Mocca.simulate(process; cycles = 1, info_level = -1)
    result = Mocca.cyclic_steady_state(process; progress = false)
    Mocca.performance(result)
    return process, result
end

# Warm native web figures and their Cairo posters without requiring a GL context.
function _warm_plot(process, result)
    bed = process.flowsheet[:Bed]
    states = [s[:Bed] for s in result.states]
    stages = Mocca.stages(process)
    backend = GLMakie.Makie.current_backend()
    WGLMakie.activate!()
    try
        figures = (
            Mocca.plot_cell(states, bed, result.timesteps, length(states[end][:Pressure])),
            Mocca.plot_flowsheet_streams(result.states, process.model, result.timesteps; stages),
            Mocca.plot_profiles(result.states, result.timesteps, bed,
                cumsum([s.duration for s in stages]); labels = [s.name for s in stages], unit = :Bed),
        )
        mktempdir() do dir
            for (i, fig) in enumerate(figures)
                CairoMakie.save(joinpath(dir, "plot-$i.png"), fig; backend = CairoMakie)
            end
        end
    finally
        backend.activate!()
    end
    return nothing
end

# The whole warm-up, strictly. Every warm package defines this; the simulator smoke
# test calls it directly so a workload that has quietly stopped baking anything
# fails a test instead of just being slow.
function _warm()
    process, result = _warm_solve()
    _warm_plot(process, result)
    return nothing
end

@setup_workload begin
    @compile_workload begin
        # Keep the solve cache even if plotting is unavailable during precompile.
        # The smoke test calls _warm() directly so failures cannot go unnoticed.
        try
            _warm()
        catch
        end
    end
end

end # module
