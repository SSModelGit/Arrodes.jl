fixed_planner(action) = CallbackPlanner(
    prepare_fn=(mdp, context) -> action,
    action_fn=(prepared, mdp, state, context) -> prepared,
)

function tiny_objective_problem()
    hypotheses = [
        ObjectiveHypothesis(
            :right, :right,
            BehaviorModel(fixed_planner(:right), EpsilonGreedyLikelihood(epsilon=0.1)),
            0.5,
        ),
        ObjectiveHypothesis(
            :left, :left,
            BehaviorModel(fixed_planner(:left), EpsilonGreedyLikelihood(epsilon=0.1)),
            0.5,
        ),
    ]
    ObjectiveInferenceProblem(
        hypotheses=hypotheses,
        mdp_builder=(objective, hypothesis) -> TinyMDP(objective),
    )
end

@testset "Objective inference pipeline" begin
    states = [0, 1, 2, 3]
    actions = fill(:right, length(states))
    exact = infer_objectives_exact(tiny_objective_problem(), states, actions)
    smc = infer_objectives_smc(
        tiny_objective_problem(), states, actions,
        SMCConfig(
            n_particles=128,
            ess_threshold=0.9,
            invariant_move=ObjectiveReplayMove(),
            invariant_steps=2,
            seed=UInt64(5),
        ),
    )
    @test posterior(exact)[1] > 0.99
    @test posterior(smc)[1] > 0.95
    @test maximum(abs.(posterior(exact) - posterior(smc))) < 0.05
    @test length(smc.state.ancestry) == length(actions)
    @test smc.state.cloud.log_normalizer < 0.0
    @test all(length(particle.trace.score_history) == length(actions)
              for particle in posterior_particles(smc))
    @test best_hypothesis(smc).hypothesis.id === :right
    @test hypothesis_mdp(smc.state.problem, :right).objective === :right
end
