module JutulAgentMocca

# Per-simulator warm-up package for Mocca, which ships no PrecompileTools workload.
# Bakes the agent's simulate_process + plot_outlet paths (the shipped DCB quick-start;
# one workload because plot_outlet needs the solve's states). See JutulAgentJutulDarcy
# for the @recompile_invalidations rationale.

using Mocca, Jutul
using PrecompileTools: @recompile_invalidations, @setup_workload, @compile_workload

@recompile_invalidations begin
    import CairoMakie
    import WGLMakie
    using GLMakie
end

function _warm_solve()
    json_dir = joinpath(dirname(pathof(Mocca)), "../models/json/")
    constants, info = Mocca.parse_input(joinpath(json_dir, "dcb_haghpanah_2013_co2_n2_input_simple.json"))
    case, ts_config = Mocca.setup_mocca_case(constants, info)
    states, timesteps =
        Mocca.simulate_process(case; timestep_selector_cfg = ts_config, info_level = 0)
    return case, states, timesteps
end

# The plotter the skills teach. It needs the solve's states, so it takes them rather
# than rebuilding a case, and it is called after the solve so a context-less
# precompile still keeps the solve's bake.
function _warm_plot(case, states, timesteps)
    Mocca.plot_outlet(case, states, timesteps)
    return nothing
end

# The whole warm-up, strictly. Every warm package defines this; the simulator smoke
# test calls it directly so a workload that has quietly stopped baking anything
# fails a test instead of just being slow.
function _warm()
    case, states, timesteps = _warm_solve()
    _warm_plot(case, states, timesteps)
    return nothing
end

@setup_workload begin
    @compile_workload begin
        # A context-less precompile (headless, no xvfb) throws in the plot half.
        # Whatever ran before the throw is still baked, so the solve survives.
        try
            _warm()
        catch
        end
    end
end

end # module
