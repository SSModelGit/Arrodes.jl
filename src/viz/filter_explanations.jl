function state_xy(state)
    coordinates = hasproperty(state, :x) ? getproperty(state, :x) : state
    values = Float64.(vec(coordinates))
    (values[1], values[2])
end

objective_scalar(value) = value isa Real ? Float64(value) : Float64(first(value))

objective_problem(result::ExactObjectiveResult) = result.problem
objective_problem(result::SMCResult{<:SequentialState{<:ObjectiveInferenceProblem}}) =
    result.state.problem

objective_length(result::ExactObjectiveResult) = size(result.posterior_history, 2)
objective_length(result::SMCResult{<:SequentialState{<:ObjectiveInferenceProblem}}) =
    length(result.state.problem.actions)

objective_probabilities(result::ExactObjectiveResult, timestep) =
    result.posterior_history[:, timestep]

function objective_probabilities(
    result::SMCResult{<:SequentialState{<:ObjectiveInferenceProblem}}, timestep,
)
    summaries = filter(summary -> summary isa ObjectivePosteriorSummary &&
                       summary.stage.observation == timestep, result.state.summaries)
    last(summaries).probabilities
end

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

function planner_context(problem, hypothesis, initial_state, horizon, timestep)
    PlanningContext(
        hypothesis_id=hypothesis.id,
        timestep=1,
        states=Any[initial_state],
        actions=Any[],
        horizon=horizon,
        rng=MersenneTwister(hash((problem.seed, hypothesis.id, :visualization, timestep))),
        metadata=hypothesis.metadata,
    )
end

function planned_path(problem, hypothesis, timestep; from_current, horizon)
    index = hypothesis_index(problem, hypothesis.id)
    mdp = hypothesis_mdp(problem, index)
    observed_index = from_current ? timestep : 1
    observation = problem.states[observed_index]
    initial_state = problem.state_adapter(mdp, observation, observed_index)
    context = planner_context(problem, hypothesis, initial_state, horizon, timestep)
    artifact = prepare_cached!(problem.planning_cache, hypothesis.behavior.planner, mdp, context)
    path = rollout(
        hypothesis.behavior.planner, deepcopy(artifact), mdp, initial_state, horizon, context,
    )
    [state_xy(state) for state in path.states]
end

function top_hypotheses(result, timestep, n_top)
    problem = objective_problem(result)
    probabilities = objective_probabilities(result, timestep)
    indices = sortperm(probabilities; rev=true)[1:min(n_top, length(probabilities))]
    [(index=index, hypothesis=problem.hypotheses[index], probability=probabilities[index])
     for index in indices]
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

"""Objective-posterior frame with observed and candidate-planner trajectories."""
function plot_particle_filter_frame(result, timestep; true_objective_fn=nothing,
                                    true_mdp=nothing, gridsize=100, n_top=5,
                                    trace_from_current=false,
                                    rollout_horizon=objective_length(result))
    problem = objective_problem(result)
    reference_mdp = isnothing(true_mdp) ? hypothesis_mdp(problem, 1) : true_mdp
    lo, hi = plot_bounds(reference_mdp)
    xs = range(lo, hi; length=gridsize)
    ys = range(lo, hi; length=gridsize)
    background = true_grid(true_objective_fn, true_mdp, problem, xs, ys)
    if isnothing(background)
        best = first(top_hypotheses(result, timestep, 1)).hypothesis
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
    for (rank, entry) in enumerate(top_hypotheses(result, timestep, n_top))
        points = planned_path(
            problem, entry.hypothesis, timestep;
            from_current=trace_from_current, horizon=rollout_horizon,
        )
        label = "$(entry.hypothesis.id) ($(round(entry.probability; digits=3)))"
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

plot_particle_filter_explanation(result; kwargs...) =
    plot_particle_filter_frame(result, objective_length(result); kwargs...)

"""Frame containing the true objective and separate top-hypothesis maps."""
function plot_particle_heatmaps_frame(result, timestep; true_objective_fn=nothing,
                                      true_mdp=nothing, gridsize=100, n_top=5)
    problem = objective_problem(result)
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
    for entry in top_hypotheses(result, timestep, n_top)
        mdp = hypothesis_mdp(problem, entry.hypothesis.id)
        panel = heatmap(
            xs, ys, objective_grid(mdp, problem, xs, ys);
            aspect_ratio=:equal, color=:viridis, xlabel="x", ylabel="y",
            title="$(entry.hypothesis.id): p=$(round(entry.probability; digits=3))",
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

make_particle_filter_frame_fn(result; kwargs...) =
    timestep -> plot_particle_filter_frame(result, timestep; kwargs...)
make_particle_heatmaps_frame_fn(result; kwargs...) =
    timestep -> plot_particle_heatmaps_frame(result, timestep; kwargs...)

"""Map mean, map uncertainty, and observed path for a world posterior stage."""
function plot_world_filter_frame(
    result::SMCResult{<:SequentialState{<:WorldInferenceProblem}}, timestep;
    marker_size=8,
)
    state = result.state
    summaries = filter(summary -> summary isa WorldPosteriorSummary &&
                       summary.stage.observation == timestep, state.summaries)
    summary = last(summaries)
    X = state.problem.context.model.params.locations
    path = state.problem.trajectory.states[1:timestep]
    locations = state.problem.evidence isa DirectErgodicEvidence ?
        state.problem.evidence.location.(path) : path
    points = state_xy.(locations)
    mean_panel = scatter(
        X[:, 1], X[:, 2]; marker_z=summary.map_mean, markersize=marker_size,
        color=:viridis, aspect_ratio=:equal, title="Inferred field mean, t=$timestep",
        label=false,
    )
    plot!(mean_panel, first.(points), last.(points); color=:red, linewidth=2,
          marker=:circle, label="observed")
    uncertainty_panel = scatter(
        X[:, 1], X[:, 2]; marker_z=sqrt.(max.(summary.map_variance, 0.0)),
        markersize=marker_size, color=:magma, aspect_ratio=:equal,
        title="Posterior field standard deviation", label=false,
    )
    plot(mean_panel, uncertainty_panel; layout=(1, 2), size=(1100, 480))
end

"""Energy, ESS, CESS, and bridge progression diagnostics."""
function plot_world_diagnostics(result::SMCResult{<:SequentialState{<:WorldInferenceProblem}})
    diagnostics = result.state.diagnostics
    summary_energy = Dict(
        (summary.stage.observation, summary.stage.bridge, summary.stage.λ) => summary.mean_energy
        for summary in result.state.summaries if summary isa WorldPosteriorSummary
    )
    energy = [summary_energy[(diagnostic.stage.observation,
                             diagnostic.stage.bridge,
                             diagnostic.stage.λ)]
              for diagnostic in diagnostics]
    stages = 1:length(diagnostics)
    energy_panel = plot(stages, energy; xlabel="inference stage", ylabel="mean energy",
                        marker=:circle, label=false, title="Behavioral energy")
    population = length(result.state.cloud.particles)
    particle_panel = plot(stages, getfield.(diagnostics, :ess); label="ESS",
                          xlabel="inference stage", ylabel="particle count",
                          ylim=(0, population), marker=:circle)
    plot!(particle_panel, stages, getfield.(diagnostics, :cess); label="CESS", marker=:diamond)
    bridge_panel = plot(stages, [diagnostic.stage.λ for diagnostic in diagnostics];
                        label="λ", marker=:circle, ylim=(0, 1),
                        xlabel="inference stage", ylabel="bridge progress")
    branch_names = sort(unique(reduce(
        vcat, getfield.(result.state.ancestry, :branches); init=Symbol[],
    )); by=string)
    branch_panel = plot(; xlabel="inference stage", ylabel="particle fraction",
                        ylim=(0, 1), title="Paired proposal branches")
    for name in branch_names
        frequencies = [count(==(name), ancestry.branches) / length(ancestry.branches)
                       for ancestry in result.state.ancestry]
        plot!(branch_panel, stages, frequencies; marker=:circle, label=string(name))
    end
    plot(energy_panel, particle_panel, bridge_panel, branch_panel;
         layout=(4, 1), size=(900, 1150))
end

"""Render the fixed SCRIBE EOF basis used by every world particle."""
function plot_world_modes(problem::WorldInferenceProblem; marker_size=8)
    X = problem.context.model.params.locations
    modes = SCRIBE.eof_modes(problem.context.model)
    panels = [scatter(
        X[:, 1], X[:, 2]; marker_z=modes[:, index], markersize=marker_size,
        color=:balance, aspect_ratio=:equal, label=false, title="EOF mode $index",
    ) for index in axes(modes, 2)]
    columns = ceil(Int, sqrt(length(panels)))
    rows = ceil(Int, length(panels) / columns)
    plot(panels...; layout=(rows, columns), size=(420columns, 380rows))
end

"""Show proposal and resampling genealogy without conflating the two ancestries."""
function plot_world_ancestry(result::SMCResult)
    ancestry = result.state.ancestry
    proposal = hcat(getfield.(ancestry, :proposal_parents)...)
    resampling = hcat(getfield.(ancestry, :resampling_parents)...)
    proposal_panel = heatmap(
        proposal; xlabel="inference stage", ylabel="child particle",
        title="Paired-proposal parents", color=:viridis,
    )
    resampling_panel = heatmap(
        resampling; xlabel="inference stage", ylabel="child particle",
        title="Resampling parents", color=:viridis,
    )
    plot(proposal_panel, resampling_panel; layout=(2, 1), size=(900, 800))
end

"""Visualize the inferred Gaussian natural-parameter increment and PSD status."""
function plot_world_deployment(deployment::BehaviorInformationDeployment)
    eigenvalues = eigvals(Symmetric(deployment.ΔY))
    spectrum = bar(
        eachindex(eigenvalues), eigenvalues; legend=false,
        xlabel="coefficient direction", ylabel="ΔY eigenvalue",
        title="Deployment: $(deployment.reason)",
    )
    matrix = heatmap(deployment.ΔY; color=:balance, title="Behavioral ΔY")
    plot(spectrum, matrix; layout=(1, 2), size=(1000, 430))
end

"""Overlay plans induced by the highest-weight candidate SCRIBE worlds."""
function plot_world_particle_plans(
    result::SMCResult{<:SequentialState{<:WorldInferenceProblem}};
    n_top=5,
    horizon=10,
)
    problem = result.state.problem
    evidence = problem.evidence
    evidence isa PlannerWorldEvidence ||
        throw(ArgumentError("particle plans require PlannerWorldEvidence"))
    particles = result.state.cloud.particles
    indices = sortperm(getfield.(particles, :log_weight); rev=true)[1:min(n_top, length(particles))]
    observed = state_xy.(problem.trajectory.states)
    panel = plot(
        first.(observed), last.(observed); color=:red, linewidth=3,
        marker=:circle, label="observed", aspect_ratio=:equal,
        title="Plans under candidate world beliefs",
    )
    for index in indices
        particle = particles[index]
        model = candidate_model(problem.context, particle.value)
        mdp = evidence.mdp_builder(model, evidence.objective)
        state = evidence.state_adapter(mdp, last(problem.trajectory.states),
                                       length(problem.trajectory.states))
        context = PlanningContext(
            hypothesis_id=:world,
            timestep=1,
            states=Any[state],
            actions=Any[],
            horizon=horizon,
            rng=MersenneTwister(hash((particle.lineage, :visualization), UInt(0))),
            metadata=evidence.metadata,
        )
        artifact = prepare(evidence.behavior.planner, mdp, context)
        plan = rollout(evidence.behavior.planner, artifact, mdp, state, horizon, context)
        points = state_xy.(plan.states)
        plot!(panel, first.(points), last.(points); linewidth=2, linestyle=:dash,
              label="lineage $(particle.lineage)")
    end
    panel
end

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
