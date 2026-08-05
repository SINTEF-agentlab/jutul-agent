module JutulAgentFimbul

# Per-simulator warm-up package for Fimbul, which reuses JutulDarcy's solver plus an
# energy equation and ships no PrecompileTools workload of its own. Bakes the agent's
# simulate_reservoir + plot_reservoir paths; see JutulAgentJutulDarcy for the
# @recompile_invalidations rationale.

using Fimbul, JutulDarcy, Jutul
using PrecompileTools: @recompile_invalidations, @setup_workload, @compile_workload

@recompile_invalidations begin
    import CairoMakie
    import WGLMakie
    using GLMakie
end

# Smallest Fimbul run that compiles the geothermal (thermal-Darcy) solve path: the
# shipped analytical 1D case on a tiny mesh + few steps. Returns the case and the
# result so the plot warm-up below can bake the plotters on the types a real session
# hands them.
function _warm_solve()
    case, _sol, _x, _t = analytical_1d(num_cells = 20, num_steps = 8)
    result = simulate_reservoir(case, info_level = -1)
    return case, result
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

# The plotters behind julia_plot (shared with JutulDarcy). They need a GL context, so
# they are baked separately from the solve; a context-less precompile (headless, no
# xvfb) skips them but still bakes _warm_solve.
#
# Bake the plotters the skills teach, since those are the calls whose first-call
# compilation the user waits through. For Fimbul that is `plot_reservoir` over the
# whole state series, and `plot_cell_data` for the chrome-free static figure.
function _warm_plot(case, result)
    _warm_figure(() -> plot_reservoir(case.model, result.states))
    _warm_figure(function ()
        dom = reservoir_domain(CartesianMesh((2, 2, 1), (1.0, 1.0, 1.0)),
            permeability = 1e-13, porosity = 0.2)
        fig, ax, plt = plot_cell_data(physical_representation(dom), dom[:porosity])
        return fig
    end)
    return nothing
end

# The whole warm-up, strictly. Every warm package defines this; the simulator smoke
# test calls it directly so a workload that has quietly stopped baking anything
# fails a test instead of just being slow.
function _warm()
    case, result = _warm_solve()
    _warm_plot(case, result)
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
