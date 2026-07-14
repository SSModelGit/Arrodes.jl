function _state_xy(state)
    if hasproperty(state, :x)
        coordinates = getproperty(state, :x)
        if coordinates isa AbstractMatrix
            return (Float64(coordinates[1, 1]), Float64(coordinates[1, 2]))
        elseif coordinates isa AbstractVector && !isempty(coordinates) && first(coordinates) isa Tuple
            point = first(coordinates)
            return (Float64(point[1]), Float64(point[2]))
        elseif coordinates isa AbstractVector
            return (Float64(coordinates[1]), Float64(coordinates[2]))
        end
    elseif state isa AbstractMatrix
        return (Float64(state[1, 1]), Float64(state[1, 2]))
    elseif state isa AbstractVector
        return (Float64(state[1]), Float64(state[2]))
    end
    throw(ArgumentError("cannot extract x/y coordinates from $(typeof(state))"))
end

_objective_scalar(value) = value isa Real ? Float64(value) : Float64(first(value))

function _filter_state_at(result::AbstractInferenceResult, timestep::Int)
    1 <= timestep <= size(result.posterior_history, 2) ||
        throw(BoundsError(result.posterior_history, (:, timestep)))
    final_state = result.state
    config = final_state isa SMCFilterState ? final_state.config.model : final_state.config
    weights = log.(result.posterior_history[:, timestep])
    return DiscreteFilterState(
        config,
        final_state.cache,
        weights,
        timestep,
        Any[final_state.states[1:timestep]...],
        Any[final_state.actions[1:timestep]...],
        result.log_evidence_history[timestep],
    )
end

function _plot_bounds(mdp)
    hasproperty(mdp, :dimensions) || throw(ArgumentError(
        "visualization requires `mdp.dimensions` or explicit bounds"))
    lo, hi = getproperty(mdp, :dimensions)
    return (Float64(lo), Float64(hi))
end

function _objective_grid(mdp, config, xs, ys)
    values = Matrix{Float64}(undef, length(ys), length(xs))
    for (j, y) in enumerate(ys), (i, x) in enumerate(xs)
        state = config.state_adapter(mdp, [x, y], 1)
        values[j, i] = _objective_scalar(mdp.obj(state))
    end
    return values
end

function _true_grid(true_objective_fn, true_mdp, config, xs, ys)
    if !isnothing(true_objective_fn)
        return [Float64(true_objective_fn(x, y)) for y in ys, x in xs]
    end
    isnothing(true_mdp) && return nothing
    return _objective_grid(true_mdp, config, xs, ys)
end

function _observed_xy(state::DiscreteFilterState, timestep::Int)
    points = [_state_xy(state.states[t]) for t in 1:timestep]
    return (first.(points), last.(points))
end

function _planner_context(filter_state, hypothesis, mdp, initial_state, horizon)
    return PlanningContext(
        hypothesis_id = hypothesis.id,
        timestep = 1,
        states = Any[initial_state],
        actions = Any[],
        horizon = horizon,
        rng = MersenneTwister(hash((filter_state.config.seed, hypothesis.id, :visualization))),
        metadata = hypothesis.metadata,
    )
end

function _planned_path(filter_state, hypothesis, mdp; from_current::Bool, horizon::Int)
    observation = from_current ? last(filter_state.states) : first(filter_state.states)
    observed_index = from_current ? filter_state.timestep : 1
    initial_state = filter_state.config.state_adapter(mdp, observation, observed_index)
    context = _planner_context(filter_state, hypothesis, mdp, initial_state, horizon)
    artifact = prepare_cached!(
        filter_state.cache,
        hypothesis.behavior.planner,
        mdp,
        context,
    )
    path = rollout(
        hypothesis.behavior.planner,
        deepcopy(artifact),
        mdp,
        initial_state,
        horizon,
        context,
    )
    return [_state_xy(state) for state in path.states]
end

function _top_hypotheses(filter_state, n_top)
    count = min(n_top, length(filter_state.config.hypotheses))
    indices = sortperm(filter_state.log_weights; rev = true)[1:count]
    return [(index = i, hypothesis = filter_state.config.hypotheses[i],
        probability = exp(filter_state.log_weights[i])) for i in indices]
end

"""Plot one objective field, optionally with an observed path overlay."""
function quick_heatmap(mdp, config::DiscreteInferenceConfig; gridsize::Int = 100,
                       title::AbstractString = "Objective", observed_states = nothing)
    lo, hi = _plot_bounds(mdp)
    xs = range(lo, hi; length = gridsize)
    ys = range(lo, hi; length = gridsize)
    panel = heatmap(xs, ys, _objective_grid(mdp, config, xs, ys);
        aspect_ratio = :equal, color = :viridis, title = title, xlabel = "x", ylabel = "y")
    if !isnothing(observed_states) && !isempty(observed_states)
        points = [_state_xy(state) for state in observed_states]
        plot!(panel, first.(points), last.(points); color = :red, linewidth = 2,
            marker = :circle, label = "observed")
    end
    return panel
end

"""
Frame showing the true field, observed trajectory, top hypothesis plan rollouts, and
the complete discrete posterior at a given timestep.
"""
function plot_particle_filter_frame(
    result::AbstractInferenceResult,
    timestep::Int;
    true_objective_fn = nothing,
    true_mdp = nothing,
    gridsize::Int = 100,
    n_top::Int = 5,
    trace_from_current::Bool = false,
    rollout_horizon::Int = size(result.posterior_history, 2),
)
    filter_state = _filter_state_at(result, timestep)
    reference_mdp = isnothing(true_mdp) ?
        hypothesis_mdp(filter_state, first(filter_state.config.hypotheses)) : true_mdp
    lo, hi = _plot_bounds(reference_mdp)
    xs = range(lo, hi; length = gridsize)
    ys = range(lo, hi; length = gridsize)
    background = _true_grid(true_objective_fn, true_mdp, filter_state.config, xs, ys)
    if isnothing(background)
        best = _top_hypotheses(filter_state, 1)[1].hypothesis
        background = _objective_grid(hypothesis_mdp(filter_state, best), filter_state.config, xs, ys)
    end

    field_panel = heatmap(xs, ys, background; aspect_ratio = :equal, color = :viridis,
        xlabel = "x", ylabel = "y", title = "Behavioral plans at t=$timestep")
    observed_x, observed_y = _observed_xy(filter_state, timestep)
    plot!(field_panel, observed_x, observed_y; color = :red, linewidth = 3,
        marker = :circle, label = "observed")

    palette = distinguishable_colors(max(n_top, 1), [RGB(1, 1, 1), RGB(0, 0, 0)])
    for (rank, entry) in enumerate(_top_hypotheses(filter_state, n_top))
        mdp = hypothesis_mdp(filter_state, entry.hypothesis)
        try
            points = _planned_path(filter_state, entry.hypothesis, mdp;
                from_current = trace_from_current, horizon = rollout_horizon)
            isempty(points) && continue
            label = "$(entry.hypothesis.id) ($(round(entry.probability; digits = 3)))"
            plot!(field_panel, first.(points), last.(points); color = palette[rank],
                linewidth = 2, linestyle = :dash, marker = :diamond, label = label)
        catch error
            @warn "Could not visualize hypothesis plan" hypothesis = entry.hypothesis.id exception = error
        end
    end

    hypotheses = filter_state.config.hypotheses
    probabilities = exp.(filter_state.log_weights)
    posterior_panel = bar(string.(getfield.(hypotheses, :id)), probabilities;
        ylim = (0, 1), legend = false, color = :steelblue,
        xlabel = "objective hypothesis", ylabel = "posterior probability",
        title = "Exact hypothesis posterior")
    return plot(field_panel, posterior_panel; layout = (1, 2), size = (1200, 520))
end

"""Final diagnostic view, equivalent to the former final particle explanation plot."""
function plot_particle_filter_explanation(result::AbstractInferenceResult; kwargs...)
    return plot_particle_filter_frame(result, size(result.posterior_history, 2); kwargs...)
end

"""Frame containing the true objective and individual top-hypothesis objective maps."""
function plot_particle_heatmaps_frame(
    result::AbstractInferenceResult,
    timestep::Int;
    true_objective_fn = nothing,
    true_mdp = nothing,
    gridsize::Int = 100,
    n_top::Int = 5,
)
    filter_state = _filter_state_at(result, timestep)
    reference_mdp = isnothing(true_mdp) ?
        hypothesis_mdp(filter_state, first(filter_state.config.hypotheses)) : true_mdp
    lo, hi = _plot_bounds(reference_mdp)
    xs = range(lo, hi; length = gridsize)
    ys = range(lo, hi; length = gridsize)
    observed_x, observed_y = _observed_xy(filter_state, timestep)
    panels = Any[]

    true_grid = _true_grid(true_objective_fn, true_mdp, filter_state.config, xs, ys)
    if !isnothing(true_grid)
        panel = heatmap(xs, ys, true_grid; aspect_ratio = :equal, color = :viridis,
            title = "True objective", xlabel = "x", ylabel = "y")
        plot!(panel, observed_x, observed_y; color = :red, linewidth = 2,
            marker = :circle, label = false)
        push!(panels, panel)
    end

    for entry in _top_hypotheses(filter_state, n_top)
        mdp = hypothesis_mdp(filter_state, entry.hypothesis)
        panel = heatmap(xs, ys, _objective_grid(mdp, filter_state.config, xs, ys);
            aspect_ratio = :equal, color = :viridis, xlabel = "x", ylabel = "y",
            title = "$(entry.hypothesis.id): p=$(round(entry.probability; digits = 3))")
        plot!(panel, observed_x, observed_y; color = :red, linewidth = 2,
            marker = :circle, label = false)
        push!(panels, panel)
    end
    columns = ceil(Int, sqrt(length(panels)))
    rows = ceil(Int, length(panels) / columns)
    return plot(panels...; layout = (rows, columns), size = (430 * columns, 390 * rows),
        plot_title = "Objective hypotheses at t=$timestep")
end

function make_particle_filter_frame_fn(result::AbstractInferenceResult; kwargs...)
    return timestep -> plot_particle_filter_frame(result, timestep; kwargs...)
end

function make_particle_heatmaps_frame_fn(result::AbstractInferenceResult; kwargs...)
    return timestep -> plot_particle_heatmaps_frame(result, timestep; kwargs...)
end

function animate_particle_filter_from_frames(frames::AbstractVector; fps::Int = 2)
    animation = Animation()
    for plot_frame in frames
        frame(animation, plot_frame)
    end
    return (animation, fps)
end

function save_particle_filter_animation(frames::AbstractVector, path::AbstractString; fps::Int = 2)
    animation, actual_fps = animate_particle_filter_from_frames(frames; fps = fps)
    mkpath(dirname(abspath(path)))
    return gif(animation, path; fps = actual_fps)
end
