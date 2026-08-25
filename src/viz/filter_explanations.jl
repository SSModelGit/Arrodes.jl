function state_xy(state::MuKumari.KAgentState)
    values = Float64.(vec(state.x))
    (values[1], values[2])
end

function state_xy(state::AbstractArray)
    values = Float64.(vec(state))
    (values[1], values[2])
end

objective_scalar(value) = value isa Real ? Float64(value) : Float64(first(value))

function plot_bounds(mdp)
    lo, hi = mdp.dimensions
    Float64(lo), Float64(hi)
end

function objective_grid(mdp, problem, xs, ys)
    [objective_scalar(mdp.obj(problem.state_adapter(mdp, [x, y], 1)))
     for y in ys, x in xs]
end

function true_grid(true_objective_fn, true_mdp, problem, xs, ys)
    !isnothing(true_objective_fn) && return [Float64(true_objective_fn(x, y)) for y in ys, x in xs]
    isnothing(true_mdp) ? nothing : objective_grid(true_mdp, problem, xs, ys)
end

function counterfactual_behavior_context(problem, hypothesis, initial_state, horizon, timestep)
    Dict{Symbol,Any}(
        :hypothesis_id => hypothesis.id,
        :timestep => 1,
        :states => Any[initial_state],
        :actions => Any[],
        :horizon => horizon,
        :rng => MersenneTwister(hash((hypothesis.id, :visualization, timestep))),
        :metadata => hypothesis.metadata,
    )
end

function planned_path(problem, hypothesis, timestep; from_current, horizon)
    index = hypothesis_index(problem, hypothesis.id)
    mdp = hypothesis_mdp(problem, index)
    observed_index = from_current ? timestep : 1
    observation = problem.states[observed_index]
    initial_state = problem.state_adapter(mdp, observation, observed_index)
    context = counterfactual_behavior_context(
        problem, hypothesis, initial_state, horizon, timestep,
    )
    artifact = prepare_behavior(hypothesis.behavior.solver, mdp, context)
    path = rollout_behavior(
        hypothesis.behavior.solver, artifact, mdp, initial_state, horizon, context,
    )
    [state_xy(state) for state in path[:states]]
end

"""Plot one objective field, optionally with an observed path overlay."""
function quick_heatmap(mdp, problem::ObjectiveInferenceProblem; gridsize=100,
                       title="Objective", observed_states=nothing)
    lo, hi = plot_bounds(mdp)
    xs = range(lo, hi; length=gridsize)
    ys = range(lo, hi; length=gridsize)
    panel = heatmap(
        xs, ys, objective_grid(mdp, problem, xs, ys);
        aspect_ratio=:equal, color=:viridis, title=title, xlabel="x", ylabel="y",
    )
    if !isnothing(observed_states) && !isempty(observed_states)
        points = state_xy.(observed_states)
        plot!(panel, first.(points), last.(points); color=:red, linewidth=2,
              marker=:circle, label="observed")
    end
    panel
end

"""Objective-posterior frame with observed and counterfactual trajectories."""
function plot_particle_filter_frame(result, problem, timestep; true_objective_fn=nothing,
                                    true_mdp=nothing, gridsize=100, n_top=5,
                                    trace_from_current=false,
                                    rollout_horizon=objective_observation_count(result))
    reference_mdp = isnothing(true_mdp) ? hypothesis_mdp(problem, 1) : true_mdp
    lo, hi = plot_bounds(reference_mdp)
    xs = range(lo, hi; length=gridsize)
    ys = range(lo, hi; length=gridsize)
    background = true_grid(true_objective_fn, true_mdp, problem, xs, ys)
    if isnothing(background)
        best = first(top_objective_hypotheses(result, problem, timestep, 1))[:hypothesis]
        background = objective_grid(hypothesis_mdp(problem, best.id), problem, xs, ys)
    end
    field_panel = heatmap(
        xs, ys, background; aspect_ratio=:equal, color=:viridis,
        xlabel="x", ylabel="y", title="Behavioral plans at t=$timestep",
    )
    observed = state_xy.(problem.states[1:timestep])
    plot!(field_panel, first.(observed), last.(observed); color=:red, linewidth=3,
          marker=:circle, label="observed")
    palette = distinguishable_colors(max(n_top, 1), [RGB(1, 1, 1), RGB(0, 0, 0)])
    for (rank, entry) in enumerate(top_objective_hypotheses(result, problem, timestep, n_top))
        points = planned_path(
            problem, entry[:hypothesis], timestep;
            from_current=trace_from_current, horizon=rollout_horizon,
        )
        label = "$(entry[:hypothesis].id) ($(round(entry[:probability]; digits=3)))"
        plot!(field_panel, first.(points), last.(points); color=palette[rank],
              linewidth=2, linestyle=:dash, marker=:diamond, label=label)
    end
    probabilities = objective_probabilities(result, timestep)
    posterior_panel = bar(
        string.(getfield.(problem.hypotheses, :id)), probabilities;
        ylim=(0, 1), legend=false, color=:steelblue,
        xlabel="objective hypothesis", ylabel="posterior probability",
        title="Objective posterior",
    )
    plot(field_panel, posterior_panel; layout=(1, 2), size=(1200, 520))
end

plot_particle_filter_explanation(result, problem; kwargs...) =
    plot_particle_filter_frame(
        result,
        problem,
        objective_observation_count(result);
        kwargs...,
    )

"""Frame containing the true objective and separate top-hypothesis maps."""
function plot_particle_heatmaps_frame(result, problem, timestep; true_objective_fn=nothing,
                                      true_mdp=nothing, gridsize=100, n_top=5)
    reference_mdp = isnothing(true_mdp) ? hypothesis_mdp(problem, 1) : true_mdp
    lo, hi = plot_bounds(reference_mdp)
    xs = range(lo, hi; length=gridsize)
    ys = range(lo, hi; length=gridsize)
    observed = state_xy.(problem.states[1:timestep])
    panels = Any[]
    truth = true_grid(true_objective_fn, true_mdp, problem, xs, ys)
    if !isnothing(truth)
        panel = heatmap(xs, ys, truth; aspect_ratio=:equal, color=:viridis,
                        title="True objective", xlabel="x", ylabel="y")
        plot!(panel, first.(observed), last.(observed); color=:red, linewidth=2,
              marker=:circle, label=false)
        push!(panels, panel)
    end
    for entry in top_objective_hypotheses(result, problem, timestep, n_top)
        mdp = hypothesis_mdp(problem, entry[:hypothesis].id)
        panel = heatmap(
            xs, ys, objective_grid(mdp, problem, xs, ys);
            aspect_ratio=:equal, color=:viridis, xlabel="x", ylabel="y",
            title="$(entry[:hypothesis].id): p=$(round(entry[:probability]; digits=3))",
        )
        plot!(panel, first.(observed), last.(observed); color=:red, linewidth=2,
              marker=:circle, label=false)
        push!(panels, panel)
    end
    columns = ceil(Int, sqrt(length(panels)))
    rows = ceil(Int, length(panels) / columns)
    plot(panels...; layout=(rows, columns), size=(430columns, 390rows),
         plot_title="Objective hypotheses at t=$timestep")
end

make_particle_filter_frame_fn(result, problem; kwargs...) =
    timestep -> plot_particle_filter_frame(result, problem, timestep; kwargs...)
make_particle_heatmaps_frame_fn(result, problem; kwargs...) =
    timestep -> plot_particle_heatmaps_frame(result, problem, timestep; kwargs...)

function animate_particle_filter_from_frames(frames::AbstractVector; fps=2)
    animation = Animation()
    for plot_frame in frames
        frame(animation, plot_frame)
    end
    animation, fps
end

function save_particle_filter_animation(frames::AbstractVector, path::AbstractString; fps=2)
    animation, actual_fps = animate_particle_filter_from_frames(frames; fps=fps)
    mkpath(dirname(abspath(path)))
    gif(animation, path; fps=actual_fps)
end
