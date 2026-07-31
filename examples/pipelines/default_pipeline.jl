using Arrodes
using MuKumari
using Plots

import GeoInterface as GI

# These are named, domain-informed objectives. No objective function is sampled.
menv = build_shared_menv(MuEnvSpec())
agent_params = Dict(
    :start => [4.0 4.0],
    :dimensions => (0.0, 10.0),
    :menv => menv,
    :obcs => [
        GI.Polygon([[(2.0, 2.0), (2.0, 3.0), (3.0, 3.0), (3.0, 2.0), (2.0, 2.0)]]),
        # MuKumari's observation encoder requests the two nearest obstacles.
        GI.Polygon([[(7.0, 2.0), (7.0, 3.0), (8.0, 3.0), (8.0, 2.0), (7.0, 2.0)]]),
    ],
)

goal_reward = s -> (-hypot(s.x[1, 1] - 9.0, s.x[1, 2] - 9.0), false)
home_reward = s -> (-hypot(s.x[1, 1] - 1.0, s.x[1, 2] - 1.0), false)

# Custom planner callbacks express known behavior for each objective class.
goal_planner = CallbackPlanner(
    prepare_fn = (mdp, context) -> :ne,
    action_fn = (nominal_action, mdp, state, context) -> nominal_action,
)
home_planner = CallbackPlanner(
    prepare_fn = (mdp, context) -> :sw,
    action_fn = (nominal_action, mdp, state, context) -> nominal_action,
)

hypotheses = [
    ObjectiveHypothesis(
        id = :reach_goal,
        objective = goal_reward,
        behavior = BehaviorModel(goal_planner, EpsilonGreedyLikelihood(epsilon = 0.08)),
        prior_probability = 0.6,
        metadata = (; description = "Travel to the northeast task site"),
    ),
    ObjectiveHypothesis(
        id = :return_home,
        objective = home_reward,
        behavior = BehaviorModel(home_planner, EpsilonGreedyLikelihood(epsilon = 0.08)),
        prior_probability = 0.4,
        metadata = (; description = "Return to the southwest home site"),
    ),
]

config = DiscreteInferenceConfig(
    hypotheses = hypotheses,
    mdp_builder = (objective, hypothesis) ->
        build_kagent_pomdp(agent_params, objective; name = String(hypothesis.id)),
    state_adapter = (mdp, observation, timestep) ->
        blindstart_KAgentState(mdp, reshape(Float64.(observation[1:2]), 1, 2)),
)

observed_states = [4.0 5.0 6.0 7.0; 4.0 5.0 6.0 7.0]
observed_actions = [:ne, :ne, :ne, :ne]

# The scalable default is trace-preserving SMC. Exact enumeration remains useful
# as a small-hypothesis reference calculation via `infer_objectives`.
smc_config = SMCInferenceConfig(
    model = config,
    n_particles = 256,
    ess_threshold = 0.7,
    rejuvenation_steps = 2,
)
result = infer_objectives_smc(smc_config, observed_states, observed_actions)
winner = best_hypothesis(result)

println("Posterior: ", Dict(h.id => p for (h, p) in zip(hypotheses, posterior(result))))
println("Best explanation: ", winner.hypothesis.id, " (", winner.probability, ")")
println("ESS history: ", result.ess_history)
println("Resampling timesteps: ", result.state.resampling_times)

# Restore all three diagnostic animation families:
#   1. each hypothesis planned from the shared initial state;
#   2. each hypothesis replanned from the current observed state;
#   3. the true and candidate objective fields as separate heatmaps.
true_objective_fn = (x, y) -> -hypot(x - 9.0, y - 9.0)
start_frame = make_particle_filter_frame_fn(
    result;
    true_objective_fn = true_objective_fn,
    true_mdp = hypothesis_mdp(result.state, :reach_goal),
    trace_from_current = false,
    n_top = length(hypotheses),
)
current_frame = make_particle_filter_frame_fn(
    result;
    true_objective_fn = true_objective_fn,
    true_mdp = hypothesis_mdp(result.state, :reach_goal),
    trace_from_current = true,
    n_top = length(hypotheses),
)
heatmaps_frame = make_particle_heatmaps_frame_fn(
    result;
    true_objective_fn = true_objective_fn,
    true_mdp = hypothesis_mdp(result.state, :reach_goal),
    n_top = length(hypotheses),
)

timesteps = axes(result.posterior_history, 2)
start_frames = [start_frame(t) for t in timesteps]
current_frames = [current_frame(t) for t in timesteps]
heatmap_frames = [heatmaps_frame(t) for t in timesteps]

output_dir = joinpath(@__DIR__, "res", "default_pipeline")
mkpath(output_dir)
save_particle_filter_animation(start_frames, joinpath(output_dir, "plans_from_start.gif"))
save_particle_filter_animation(current_frames, joinpath(output_dir, "plans_from_current.gif"))
save_particle_filter_animation(heatmap_frames, joinpath(output_dir, "objective_heatmaps.gif"))

final_plot = plot_particle_filter_explanation(
    result;
    true_objective_fn = true_objective_fn,
    true_mdp = hypothesis_mdp(result.state, :reach_goal),
    n_top = length(hypotheses),
)
savefig(final_plot, joinpath(output_dir, "final_filter_explanation.png"))
