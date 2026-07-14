using Arrodes
using VulcanJ

"""Construct the native Arrodes adapter for VulcanJ's one-shot ergodic planner."""
function vulcan_ergodic_behavior(
    gp_factory;
    n_steps::Int,
    epsilon::Float64 = 0.1,
    planner_kwargs...,
)
    planner = VulcanErgodicPlanner(
        gp_factory;
        n_steps = n_steps,
        planner_kwargs...,
    )
    return BehaviorModel(planner, PlanTrackingLikelihood(epsilon = epsilon))
end

"""Construct the native adapter for VulcanJ's risk-bounded information MCTS."""
function vulcan_mcts_behavior(solver_factory; epsilon::Float64 = 0.1)
    return BehaviorModel(
        VulcanMCTSPlanner(solver_factory),
        EpsilonGreedyLikelihood(epsilon = epsilon),
    )
end

# Ergodic usage:
# exploration = ObjectiveHypothesis(
#     id = :explore,
#     objective = exploration_objective,
#     behavior = vulcan_ergodic_behavior(
#         (mdp, state) -> VulcanJ.get_initial_gp(mdp, state);
#         n_steps = 80,
#         optimizer_iters = 150,
#     ),
#     prior_probability = 0.3,
# )
#
# Risk-bounded information-MCTS usage:
# safe_exploration = ObjectiveHypothesis(
#     id = :safe_explore,
#     objective = safe_exploration_objective,
#     behavior = vulcan_mcts_behavior((mdp, context) -> RiskBoundedInfoMCTS(
#         lookahead = 5,
#         time_budget = 3.0,
#         quad_order = 5,
#         risk_budget = 0.05,
#         alpha = 0.0,
#         reference_reward = 1.0,
#         rng = context.rng,
#     )),
#     prior_probability = 0.2,
# )
