struct TinyMDP <: MDP{Int,Symbol}
    objective::Symbol
end

POMDPs.actions(::TinyMDP) = [:left, :right]
POMDPs.discount(::TinyMDP) = 0.95
POMDPs.isterminal(::TinyMDP, ::Int) = false
POMDPs.gen(::TinyMDP, state::Int, action::Symbol, rng::AbstractRNG) =
    (sp=state + (action === :right ? 1 : -1), r=0.0)

struct FixedSolver <: Solver
    action::Symbol
end
struct FixedPolicy <: Policy
    action::Symbol
end
POMDPs.solve(solver::FixedSolver, ::TinyMDP) = FixedPolicy(solver.action)
POMDPs.action(policy::FixedPolicy, ::Int) = policy.action

@testset "Planning pipeline" begin
    context = PlanningContext(
        hypothesis_id=:right, states=[0], horizon=3, rng=MersenneTwister(2),
    )
    planner = POMDPSolverPlanner(FixedSolver(:right))
    artifact = prepare(planner, TinyMDP(:right), context)
    @test planned_action(planner, artifact, TinyMDP(:right), 0, context) === :right
    @test action_distribution(
        EpsilonGreedyLikelihood(epsilon=0.2), planner, artifact,
        TinyMDP(:right), 0, [:left, :right], context,
    ) ≈ [0.1, 0.9]

    open_loop = OpenLoopPlanner((mdp, context) ->
        (states=[0, 1, 2], actions=[:right, :right]))
    open_artifact = prepare(open_loop, TinyMDP(:right), context)
    @test rollout(open_loop, open_artifact, TinyMDP(:right), 0, 2, context).actions ==
        [:right, :right]

    noisy = MovementNoiseLikelihood(n_transition_samples=8, bandwidth=0.5)
    moving_context = PlanningContext(
        hypothesis_id=:right, timestep=2, states=[0, 1], actions=[:right],
        horizon=2, rng=MersenneTwister(4),
    )
    @test isfinite(observation_loglikelihood(
        noisy, planner, artifact, TinyMDP(:right), 1, :right,
        [:left, :right], moving_context,
    ))
    @test MCTSPlanner() isa AbstractPlanner
    @test SoftQPlanner(n_iterations=1) isa AbstractPlanner
    @test VulcanMCTSPlanner((mdp, context) -> nothing) isa AbstractPlanner
    @test VulcanErgodicPlanner((mdp, state, context) -> nothing; n_steps=2) isa AbstractPlanner
end
