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
# as the plots. GLMakie builds native figures and writes the terminal's PNG
# captures. WGLMakie builds and serves web figures; CairoMakie writes their
# provisional poster, whose 3D rendering may differ from the live WebGL view.
#
# GLMakie last, and it alone with `using`: a backend activates itself on load, so the
# last one loaded is current, and the interactive plotters refuse to build under a
# non-interactive backend.
#
# The shared package is loaded here too, and this is the only place it is loaded from,
# because a pkgimage is only valid for the world it was baked in: whichever package a
# session loads first fixes the order, and an order that differs from this one drops
# the parts of the bake that were inferred through a method the other package brings.
# Depending on it here makes the session's order the baked order by construction.
@recompile_invalidations begin
    import JutulAgent
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

# Build and save a native figure. Keep the Cairo path warm too: its geometry
# conversions differ from GLMakie's and are used for web posters.
#
# Both `activate!` calls are load-bearing, not tidiness. The first: the Cairo save
# leaves CairoMakie current, and the interactive plotters refuse to build under a
# non-interactive backend, so the next figure would throw. The last: `_warm()` also
# runs in a live kernel from the smoke test, and must not leave a backend behind that
# the session's own plotting cannot use.
function _warm_figure(build)
    dir = tempdir()
    GLMakie.activate!(visible = false)
    output = build()
    # Native plotters may wrap their Makie figure in an output object.
    fig = hasproperty(output, :fig) ? output.fig : output
    GLMakie.save(joinpath(dir, "jutul_agent_native_warm.png"), fig)
    CairoMakie.activate!()
    CairoMakie.save(joinpath(dir, "jutul_agent_poster_warm.png"), fig)
    GLMakie.activate!(visible = false)
    return nothing
end

# Web plots are constructed with WGLMakie active. Warming a GL figure does not
# compile this path: Jutul chooses different buffers and rendering attributes.
function _warm_web_figure(build)
    plots = JutulAgent.JutulAgentPlots
    capture = plots.start_web_capture()
    fig = try
        plots.web_figure(capture, build())
    finally
        plots.stop_web_capture(capture)
    end
    # Figure construction does not compile the live view's serialization path.
    # Exercise it without a browser so opening the first plot reuses that cache.
    views = JutulAgent.JutulAgentWebViews
    B = WGLMakie.Bonito
    session = B.Session(B.NoConnection(); asset_server = B.NoServer())
    view = views.LiveFigure(fig; resize_to = :parent)
    try
        B.jsrender(session, view)
        screen = only(view.screens)
        # Precompilation has no plot-tool screen lookup guard. Expose the raw
        # backend on the root scene so WGL can find this session's font tracker.
        push!(fig.scene.current_screens, screen.backend)
        data = views.serialize_scene(screen, fig.scene)
        B.serialize_cached(B.SerializationContext(session), data)
    finally
        for screen in view.screens
            filter!(s -> s !== screen.backend, fig.scene.current_screens)
        end
        close(session)
    end
    mktempdir() do dir
        try
            CairoMakie.activate!()
            CairoMakie.save(joinpath(dir, "poster.png"), fig)
        finally
            WGLMakie.activate!()
        end
    end
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
    # Warm web construction before GL rendering so headless precompilation can
    # still cache it when there is no OpenGL context.
    _warm_web_figure(() -> plot_reservoir(model))
    _warm_web_figure(() -> plot_reservoir(model, result.states[end]))
    _warm_web_figure(() -> plot_reservoir(model, result.states[end]; gui = false))
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
