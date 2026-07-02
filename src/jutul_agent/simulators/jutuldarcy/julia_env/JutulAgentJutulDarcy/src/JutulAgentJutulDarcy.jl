module JutulAgentJutulDarcy

# Per-simulator warm-up package for JutulDarcy. Loads the solver, then GLMakie under
# @recompile_invalidations (which caches the solver code GLMakie would otherwise
# invalidate), and bakes the simulate_reservoir + plot_cell_data paths. This makes
# the first solve ~0.5s instead of ~30s. See
# docs/adding-a-simulator.md (the warm package bakes the GLMakie-aware solve).

using JutulDarcy, Jutul
using PrecompileTools: @recompile_invalidations, @setup_workload, @compile_workload

@recompile_invalidations begin
    using GLMakie
end

# A tiny two-well immiscible reservoir run through `simulate_reservoir`, the
# high-level path the agent uses.
function _warm_solve()
    Darcy, bar, kg, meter, day = si_units(:darcy, :bar, :kilogram, :meter, :day)
    g = CartesianMesh((3, 3, 2), (300.0, 300.0, 20.0))
    domain = reservoir_domain(g, permeability = 0.3 * Darcy, porosity = 0.2)
    Prod = setup_vertical_well(domain, 1, 1, name = :Producer)
    Inj = setup_well(domain, [(3, 3, 1)], name = :Injector)
    sys = ImmiscibleSystem((LiquidPhase(), VaporPhase()),
        reference_densities = [1000.0, 100.0] .* kg / meter^3)
    model, parameters = setup_reservoir_model(domain, sys, wells = [Inj, Prod], extra_out = true)
    state0 = setup_reservoir_state(model, Pressure = 150 * bar, Saturations = [1.0, 0.0])
    dt = repeat([30.0] * day, 3)
    inj_rate = sum(pore_volume(model, parameters)) / sum(dt)
    controls = Dict(
        :Injector => InjectorControl(TotalRateTarget(inj_rate), [0.0, 1.0], density = 100.0),
        :Producer => ProducerControl(BottomHolePressureTarget(50 * bar)),
    )
    forces = setup_reservoir_forces(model, control = controls)
    simulate_reservoir(state0, model, dt, parameters = parameters, forces = forces, info_level = -1)
    return nothing
end

# The native 3D plotter behind julia_plot. Needs a GL context, so it is baked
# separately; a context-less precompile (headless, no xvfb) skips it but still
# bakes _warm_solve.
function _warm_plot()
    g = CartesianMesh((2, 2, 1), (1.0, 1.0, 1.0))
    dom = reservoir_domain(g, permeability = 1e-13, porosity = 0.2)
    fig, ax, plt = plot_cell_data(physical_representation(dom), dom[:porosity])
    GLMakie.save(joinpath(tempdir(), "jutul_agent_native_warm.png"), fig)
    return nothing
end

@setup_workload begin
    @compile_workload begin
        try
            _warm_solve()
        catch
        end
        try
            _warm_plot()
        catch
        end
    end
end

end # module
