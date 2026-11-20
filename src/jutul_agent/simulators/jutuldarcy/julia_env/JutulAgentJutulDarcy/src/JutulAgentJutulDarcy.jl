module JutulAgentJutulDarcy

# Per-simulator warm-up package for JutulDarcy. Loads the solver, then the Makie
# backends under @recompile_invalidations (which caches the solver code they would
# otherwise invalidate), and bakes the simulate_reservoir + plot_reservoir paths.
# This makes the first solve ~0.5s instead of ~30s, and the first plot seconds
# instead of ~30s. See docs/adding-a-simulator.md (the warm package bakes a
# backend-aware solve).

using JutulDarcy, Jutul
using PrecompileTools: @recompile_invalidations, @setup_workload, @compile_workload

# Julia 1.12's compiler can fail an internal assertion while @recompile_invalidations
# below re-infers the ~23k methods that loading the backends invalidates. The runtime
# recovers by itself (that one method just compiles on first use), but it writes the
# assertion and a long backtrace straight to stderr, which Pkg then shows as this
# package's precompile output, and a new user takes it for a crash. Which method trips
# it changes from run to run, since they are re-inferred in hash order, so there is no
# one call to leave out of the bake. And the block itself is not optional: without it
# the solves JutulDarcy bakes compile again in the session (SPE1 0.5s -> 12s). So
# capture what is written to the stderr descriptor around the block, and pass on all
# of it but these reports.
const _IRINTERP_ASSERTION = "irinterp is unable to handle heavy recursion correctly"

function _drop_irinterp_reports(text)
    lines = split(text, '\n')
    kept = SubString{String}[]
    i = 1
    while i <= length(lines)
        if startswith(lines[i], "Internal error: during type inference of") &&
                i + 3 <= length(lines) && occursin(_IRINTERP_ASSERTION, lines[i + 3])
            # The header, signature, "Encountered ..." and the error, then the
            # backtrace: one "<frame> at <location>" line per frame.
            i += 4
            while i <= length(lines) && occursin(" at ", lines[i])
                i += 1
            end
        else
            push!(kept, lines[i])
            i += 1
        end
    end
    return join(kept, '\n')
end

# Only the descriptor is redirected, not Julia's `stderr` stream (which writes through
# its own duplicate of it), so the packages loaded inside cannot keep hold of a stream
# that is closed afterwards, and Julia-level output is untouched.
function _without_irinterp_reports(f)
    mktemp() do path, io
        saved = Libc.dup(RawFD(2))
        Libc.dup(fd(io), RawFD(2))
        try
            f()
        finally
            Libc.dup(saved, RawFD(2))
            ccall((@static Sys.iswindows() ? :_close : :close), Cint, (RawFD,), saved)
            close(io)
            print(stderr, _drop_irinterp_reports(read(path, String)))
        end
    end
    return nothing
end

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
_without_irinterp_reports() do
    @recompile_invalidations begin
        import JutulAgent
        import CairoMakie
        import WGLMakie
        using GLMakie
    end
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
