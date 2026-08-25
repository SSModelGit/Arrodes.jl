struct ToyObjectiveMDP <: MDP{Int,Symbol}
    objective::Symbol
end

POMDPs.actions(::ToyObjectiveMDP) = [:left, :right]
POMDPs.discount(::ToyObjectiveMDP) = 0.95
POMDPs.isterminal(::ToyObjectiveMDP, state) = false
POMDPs.gen(::ToyObjectiveMDP, state, action, rng) =
    (sp=state + (action == :right ? 1 : -1), r=0.0)

@testset "objective inference from a toy trajectory" begin
    hypotheses = [
        ObjectiveHypothesis(
            id=:right,
            objective=:right,
            behavior=BehaviorModel(
                KnownActionPlanner(:right),
                EpsilonGreedyLikelihood(epsilon=0.1),
            ),
            prior_probability=0.5,
        ),
        ObjectiveHypothesis(
            id=:left,
            objective=:left,
            behavior=BehaviorModel(
                KnownActionPlanner(:left),
                EpsilonGreedyLikelihood(epsilon=0.1),
            ),
            prior_probability=0.5,
        ),
    ]
    problem = ObjectiveInferenceProblem(
        hypotheses=hypotheses,
        mdp_builder=(objective, hypothesis) -> ToyObjectiveMDP(objective),
        states=collect(0:5),
        actions=fill(:right, 6),
    )
    result = infer_objectives(
        problem;
        n_particles=128,
        rejuvenation_steps=0,
        rng=MersenneTwister(4),
    )

    @test best_hypothesis(result)[:id] == :right
    @test objective_probabilities(result)[1] > objective_probabilities(result)[2]
end
