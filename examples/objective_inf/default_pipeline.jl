using Arrodes
using JSON3
using MuKumari
using Plots
using Random

import GeoInterface as GI

function mission_agent(environment, objective, name)
    exterior = GI.Polygon([[
        (0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0), (0.0, 0.0),
    ]])
    obstacles = [
        GI.Polygon([[(2.0, 2.0), (2.0, 3.0), (3.0, 3.0), (3.0, 2.0), (2.0, 2.0)]]),
        GI.Polygon([[(7.0, 2.0), (7.0, 3.0), (8.0, 3.0), (8.0, 2.0), (7.0, 2.0)]]),
    ]
    world = GI.Polygon([
        GI.getexterior(exterior),
        (GI.getexterior(obstacle) for obstacle in obstacles)...,
    ])
    KAgentPOMDP(
        name=name,
        start=[4.0 4.0],
        dimensions=(0.0, 10.0),
        boxworld=exterior,
        objl=AgentObjectiveLandscape(objectives=Any[], f_types=Any[]),
        obcs=obstacles,
        goals=Any[],
        obj=objective,
        world,
        width=0.1,
        s=1.0,
        w=0.05,
        menv=environment,
        v=0.05,
        γ=0.95,
        digits=3,
    )
end

function main()
    mission_path = joinpath(
        @__DIR__,
        "missions",
        "default_pipeline.json",
    )
    mission = copy(JSON3.read(read(mission_path, String)))
    fields = Dict(
        :sin => x -> sin(x[1]) + cos(x[2]),
        :exp => x -> 100exp(-sum(abs2, x .- [8 8])),
        :lin => x -> x[1]^2 + x[2],
    )
    environment = MuEnv(length(fields), collect(keys(fields)), fields)
    goal = state -> (-hypot(state.x[1, 1] - 9.0, state.x[1, 2] - 9.0), false)
    home = state -> (-hypot(state.x[1, 1] - 1.0, state.x[1, 2] - 1.0), false)
    hypotheses = [
        ObjectiveHypothesis(
            id=:reach_goal,
            objective=goal,
            behavior=BehaviorModel(
                KnownActionPlanner(:ne),
                EpsilonGreedyLikelihood(epsilon=0.08),
            ),
            prior_probability=0.6,
            metadata=Dict(:description => "Travel northeast"),
        ),
        ObjectiveHypothesis(
            id=:return_home,
            objective=home,
            behavior=BehaviorModel(
                KnownActionPlanner(:sw),
                EpsilonGreedyLikelihood(epsilon=0.08),
            ),
            prior_probability=0.4,
            metadata=Dict(:description => "Return southwest"),
        ),
    ]
    states = [Float64.(state) for state in mission[:observed_states]]
    actions = Symbol.(mission[:observed_actions])
    problem = ObjectiveInferenceProblem(
        hypotheses=hypotheses,
        mdp_builder=(objective, hypothesis) ->
            mission_agent(environment, objective, String(hypothesis.id)),
        states=states,
        actions=actions,
        state_adapter=(mdp, state, timestep) ->
            blindstart_KAgentState(mdp, reshape(state, 1, 2)),
    )
    filter = mission[:filter]
    result = infer_objectives(
        problem;
        n_particles=filter[:particles],
        refresh_probability=filter[:refresh_probability],
        ess_threshold=filter[:ess_threshold],
        rejuvenation_steps=filter[:rejuvenation_steps],
        rng=MersenneTwister(42),
    )
    winner = best_hypothesis(result)
    println("Posterior: ", Dict(
        hypothesis.id => probability
        for (hypothesis, probability) in zip(
            hypotheses,
            objective_probabilities(result),
        )
    ))
    println("Best explanation: ", winner[:id])

    true_mdp = Arrodes.ObjectiveInference.hypothesis_mdp(problem, 1)
    plot_options = Dict(
        :true_objective_fn => (x, y) -> -hypot(x - 9.0, y - 9.0),
        :true_mdp => true_mdp,
        :n_top => length(hypotheses),
    )
    start_frame = make_particle_filter_frame_fn(
        result,
        problem;
        plot_options...,
        trace_from_current=false,
    )
    current_frame = make_particle_filter_frame_fn(
        result,
        problem;
        plot_options...,
        trace_from_current=true,
    )
    heatmap_frame = make_particle_heatmaps_frame_fn(result, problem; plot_options...)
    timesteps = eachindex(actions)
    output = normpath(joinpath(@__DIR__, mission[:output]))
    mkpath(output)
    save_particle_filter_animation(
        [start_frame(timestep) for timestep in timesteps],
        joinpath(output, "plans_from_start.gif"),
    )
    save_particle_filter_animation(
        [current_frame(timestep) for timestep in timesteps],
        joinpath(output, "plans_from_current.gif"),
    )
    save_particle_filter_animation(
        [heatmap_frame(timestep) for timestep in timesteps],
        joinpath(output, "objective_heatmaps.gif"),
    )
    savefig(
        plot_particle_filter_explanation(result, problem; plot_options...),
        joinpath(output, "final_filter_explanation.png"),
    )
end

main()
