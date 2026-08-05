module JutulAgentJutulDarcy

# Per-simulator warm-up package for JutulDarcy. Loads the solver, then the Makie
# backends under @recompile_invalidations (which caches the solver code they would
# otherwise invalidate), and bakes the simulate_reservoir + plot_reservoir paths.
# This makes the first solve ~0.5s instead of ~30s, and the first plot seconds
# instead of ~30s. See docs/adding-a-simulator.md (the warm package bakes a
# backend-aware solve).

using JutulDarcy, Jutul
using PrecompileTools: @recompile_invalidations, @setup_workload, @compile_workload

# Every Makie backend a session holds at once, loaded before the workload so that it
# bakes in the world a session actually runs in. Loading a backend invalidates code
# specialised against the ones already there, and that reaches past plotting into the
# solver, so a backend arriving after the bake throws away part of the solve as well
# as the plots. The plot tool loads all three: GLMakie builds the figure, WGLMakie
# serves it to the web UI, and CairoMakie writes the PNG kept as the durable record.
#
# GLMakie last, and it alone with `using`: a backend activates itself on load, so the
# last one loaded is current, and the interactive plotters refuse to build under a
# non-interactive backend.
@recompile_invalidations begin
    import CairoMakie
    import WGLMakie
    using GLMakie
end

# A tiny two-well immiscible reservoir run through `simulate_reservoir`, the
# high-level path the agent uses. Returns the model and the result so the plot
# warm-up below can bake the plotters on the types a real session hands them.
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
    result = simulate_reservoir(state0, model, dt,
        parameters = parameters, forces = forces, info_level = -1)
    return model, result
end

# One figure through the whole plot path, the way the plot tool drives it: build it
# with GLMakie active, save the native PNG, then save the CairoMakie poster written
# as the durable record. Neither rasterisation is warm from the other, so both are
# baked.
#
# Both `activate!` calls are load-bearing, not tidiness. The first: saving the poster
# leaves CairoMakie current, and the interactive plotters refuse to build under a
# non-interactive backend, so the next figure would throw. The last: `_warm()` also
# runs in a live kernel from the smoke test, and must not leave a backend behind that
# the session's own plotting cannot use.
function _warm_figure(build)
    dir = tempdir()
    GLMakie.activate!(visible = false)
    fig = build()
    GLMakie.save(joinpath(dir, "jutul_agent_native_warm.png"), fig)
    CairoMakie.activate!()
    CairoMakie.save(joinpath(dir, "jutul_agent_poster_warm.png"), fig)
    GLMakie.activate!(visible = false)
    return nothing
end

# The plotters behind julia_plot, on the types `_warm_solve` produces. They need a
# GL context, so they are baked separately from the solve; a context-less precompile
# (headless, no xvfb) skips them but still bakes _warm_solve.
#
# Bake the plotters the skills teach, since those are the calls whose first-call
# compilation the user waits through. For JutulDarcy that is `plot_reservoir` in both
# the mesh-only and the coloured-state form, and `plot_cell_data` for the
# chrome-free static figure.
function _warm_plot(model, result)
    _warm_figure(() -> plot_reservoir(model))
    _warm_figure(() -> plot_reservoir(model, result.states[end]))
    _warm_figure() do
        dom = reservoir_domain(CartesianMesh((2, 2, 1), (1.0, 1.0, 1.0)),
            permeability = 1e-13, porosity = 0.2)
        fig, ax, plt = plot_cell_data(physical_representation(dom), dom[:porosity])
        return fig
    end
    return nothing
end

# The whole warm-up, strictly. Every warm package defines this; the simulator smoke
# test calls it directly so a workload that has quietly stopped baking anything
# fails a test instead of just being slow.
function _warm()
    model, result = _warm_solve()
    _warm_plot(model, result)
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
