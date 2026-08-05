module JutulAgentBattMo

# Per-simulator warm-up package for BattMo. Bakes the agent's solve path (the shipped
# chen_2020 cell with a constant-current discharge, mirroring the battmo-overview
# skill). See JutulAgentJutulDarcy for the @recompile_invalidations rationale.

using BattMo, Jutul
using PrecompileTools: @recompile_invalidations, @setup_workload, @compile_workload

@recompile_invalidations begin
    import CairoMakie
    import WGLMakie
    using GLMakie
end

function _warm_solve()
    cell = load_cell_parameters(; from_default_set = "chen_2020")
    protocol = load_cycling_protocol(; from_default_set = "cc_discharge")
    sim = Simulation(LithiumIonBattery(), cell, protocol)
    solve(sim; info_level = -1)
    return nothing
end

# The whole warm-up, strictly. Every warm package defines this; the simulator smoke
# test calls it directly so a workload that has quietly stopped baking anything
# fails a test instead of just being slow. BattMo's plots are 1D time series, whose
# Makie path the shared JutulAgent package already warms, so there is no plot half.
function _warm()
    _warm_solve()
    return nothing
end

@setup_workload begin
    @compile_workload begin
        try
            _warm()
        catch
        end
    end
end

end # module
