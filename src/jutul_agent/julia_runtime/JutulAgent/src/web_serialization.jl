# WGL's cached programs contain the first render, with later values delivered as
# deltas. A new browser needs a current snapshot and its own update subscriptions.
# In particular, Bonito 5.2 removes all JS listeners with the same observable id
# when one root closes. Bind each view to a distinct observable (the same pattern
# as Bonito.bind_global), without changing Bonito or WGLMakie's methods.
function bind!(s::LiveScreen, source::Makie.Observable, transform = identity)
    return get!(s.bindings, source) do
        proxy = Makie.Observable{Any}(transform(source[]))
        listener = Bonito.on(s.backend.session, source) do value
            proxy[] = transform(value)
            return nothing
        end
        (proxy, listener)
    end |> first
end

function unbind!(s::LiveScreen, source)
    pair = pop!(s.bindings, source, nothing)
    isnothing(pair) && return
    _, listener = pair
    Bonito.Observables.off(listener)
    filter!(x -> x !== listener, s.backend.session.deregister_callbacks)
    return nothing
end

bind_values(s, value) = value
bind_values(s, value::Makie.Observable) = bind!(s, value)
bind_values(s, value::AbstractDict) = Dict(k => bind_values(s, v) for (k, v) in value)
bind_values(s, value::AbstractArray{<:Number}) = value
bind_values(s, value::AbstractArray) = map(v -> bind_values(s, v), value)
bind_values(s, value::Tuple) = map(v -> bind_values(s, v), value)

# Like Bonito.map, with immediate teardown when this screen closes (a disconnected
# Bonito session may stay alive briefly to allow a websocket reconnection).
function view_map(f, s::LiveScreen, sources...)
    output = Makie.Observable{Any}(f(map(x -> x[], sources)...))
    for source in sources
        listener = Bonito.on(s.backend.session, source) do _
            output[] = f(map(x -> x[], sources)...)
            return nothing
        end
        push!(s.callbacks, listener)
    end
    return output
end

function glyph_snapshot(s::LiveScreen, plot::Makie.Text)
    attr = plot.attributes
    hashes, glyphs = Makie.get_glyph_data(Makie.get_texture_atlas(), s.glyphs,
        attr[:glyphindices][], attr[:font_per_char][])
    return Dict(:glyph_hashes => hashes, :atlas_updates => glyphs,
        :scales => WGLMakie.serialize_three(attr[:glyph_scales][]))
end
function glyph_snapshot(s::LiveScreen, plot::Makie.Scatter)
    attr = plot.attributes
    attr[:sdf_marker_shape][] == 3 || return nothing
    _, glyphs = Makie.get_glyph_data(Makie.get_texture_atlas(), s.glyphs,
        attr[:marker][], attr[:font][])
    return Dict(:atlas_updates => glyphs)
end
glyph_snapshot(s, plot) = nothing

function plot_update(s, plot, update)
    update isa Bonito.LargeUpdate || return update
    # Glyph deltas are tracked per backend session. Each view must receive the
    # glyphs it needs, including a view opened after those glyphs first appeared.
    data = map(update.data) do entry
        first(entry) == :glyph_data || return entry
        glyphs = glyph_snapshot(s, plot)
        return isnothing(glyphs) ? entry : Any[:glyph_data, glyphs]
    end
    return Bonito.LargeUpdate(data)
end

function plot_snapshot(s::LiveScreen, scene, plot)
    data = copy(WGLMakie.serialize_three(scene, plot))
    # WGL 0.13's constructor callback accepts current inputs and `last=nothing`.
    # Evaluate it without modifying the shared graph or replacing its updater.
    # This small ComputePipeline contact is covered by every plotter integration
    # test; the adapter is restricted to the WGL 0.13 protocol in web_views.jl.
    edge = plot.attributes[:wgl_renderobject].parent
    inputs = (; (node.name => node[] for node in edge.inputs)...)
    program, _ = edge.callback(inputs, map(_ -> true, inputs), nothing)
    # serialize_three also adds current clip-plane uniforms after constructing
    # the program. Keep those additions while replacing cached shader inputs.
    program[:uniforms] = merge(data[:uniforms], program[:uniforms])
    merge!(data, program)
    glyphs = glyph_snapshot(s, plot)
    isnothing(glyphs) || (data[:glyph_data] = glyphs)
    data[:updater] = bind!(s, plot.attributes[:wgl_update_obs][], value -> plot_update(s, plot, value))
    return data
end

function serialize_plots(s::LiveScreen, scene, plots, result = [])
    for plot in plots
        if Makie.is_atomic_plot(plot)
            push!(result, plot_snapshot(s, scene, plot))
        end
        isempty(plot.plots) || serialize_plots(s, scene, plot.plots, result)
    end
    return result
end

function serialize_scene(s::LiveScreen, scene)
    # The wire format follows WGLMakie 0.13.16 serialization.jl (MIT, Makie
    # contributors). Scope the reactive camera/light/viewport maps to this view;
    # the upstream serializer attaches them to the longer-lived scene instead.
    M = Makie
    W = WGLMakie
    camera = scene.camera
    matrices = view_map(s, camera.view, camera.projection, camera.resolution, camera.eyeposition) do view, projection, resolution, eye
        [W.flat_m4f(view), W.flat_m4f(projection), Int32[resolution...], Float32[eye...]]
    end
    controls = M.cameracontrols(scene)
    cam3d_state = nothing
    if controls isa M.Camera3D
        cam3d_state = Dict{Symbol, Any}(key => view_map(x -> W.serialize_three(Float32.(x)), s,
            getfield(controls, key)) for key in (:lookat, :upvector, :eyeposition, :fov, :near, :far))
        cam3d_state[:resolution] = view_map(x -> Int32[x...], s, camera.resolution)
    end
    light(key) = view_map(_ -> W.serialize_three(scene.compute[key][]), s, scene.compute.onchange)
    return Dict{Symbol, Any}(
        :viewport => view_map(x -> Int32[minimum(x)..., M.widths(x)...], s, M.viewport(scene)),
        :backgroundcolor => view_map(x -> "#" * W.hex(W.Colors.color(M.to_color(x))), s, scene.backgroundcolor),
        :backgroundcolor_alpha => view_map(W.Colors.alpha, s, scene.backgroundcolor),
        :clearscene => bind_values(s, scene.clear),
        :camera => matrices,
        :light_direction => light(:dirlight_direction),
        :camera_relative_light => light(:dirlight_cam_relative),
        :ambient => light(:ambient_color),
        :light_color => light(:dirlight_color),
        :plots => serialize_plots(s, scene, scene.plots),
        :cam3d_state => cam3d_state,
        :visible => bind_values(s, scene.visible),
        :uuid => W.js_uuid(scene),
        :children => [serialize_scene(s, child) for child in scene.children],
    )
end

function insert_plot!(s::LiveScreen, scene, plot)
    session = s.backend.session
    if WGLMakie.js_uuid(scene) in s.backend.displayed_scenes
        data = serialize_plots(s, scene, Makie.Plot[plot])
        Bonito.evaljs_value(session, js"""
            $(WGLMakie.WGL).then(WGL => WGL.insert_plot($(WGLMakie.js_uuid(scene)), $data))
        """; timeout = 10, error_on_closed = false)
    else
        # Find the highest new scene: serializing it also includes its children.
        while !isnothing(scene.parent) &&
                !(WGLMakie.js_uuid(scene.parent) in s.backend.displayed_scenes)
            scene = scene.parent
        end
        data = serialize_scene(s, scene)
        Bonito.evaljs_value(session, js"""
            $(WGLMakie.WGL).then(WGL => {
                const parent = WGL.find_scene($(WGLMakie.js_uuid(scene.parent)));
                parent.scene_children.push(WGL.deserialize_scene($data, parent.screen));
            })
        """; timeout = 10, error_on_closed = false)
        WGLMakie.mark_as_displayed!(s.backend, scene)
    end
    return nothing
end

function delete_plot!(s::LiveScreen, scene, plot)
    atomics = Makie.collect_atomic_plots(plot)
    for p in atomics
        haskey(p.attributes, :wgl_update_obs) && unbind!(s, p.attributes[:wgl_update_obs][])
    end
    # Do not delete shared render objects from another view's scene graph.
    Bonito.evaljs(s.backend.session, js"""
        $(WGLMakie.WGL).then(WGL => WGL.delete_plots($(WGLMakie.js_uuid.(atomics))))
    """)
    return nothing
end

function delete_scene!(s::LiveScreen, scene)
    ids, _ = WGLMakie.all_plots_scenes(scene)
    plots = Makie.Plot[]
    Makie.for_each_atomic_plot(scene) do plot
        push!(plots, plot)
        haskey(plot.attributes, :wgl_update_obs) && unbind!(s, plot.attributes[:wgl_update_obs][])
    end
    setdiff!(s.backend.displayed_scenes, ids)
    Bonito.evaljs(s.backend.session, js"""
        $(WGLMakie.WGL).then(WGL => WGL.delete_scenes($ids, $(WGLMakie.js_uuid.(plots))))
    """)
    return nothing
end
