struct VisualMDP <: MDP{Vector{Float64},Symbol}
    objective::Symbol
    dimensions::Tuple{Float64,Float64}
    obj::Function
end

function VisualMDP(objective::Symbol)
    center = objective === :east ? [2.0, 1.0] : [0.0, 1.0]
    VisualMDP(objective, (0.0, 2.0), state -> (-sum(abs2, state .- center), false))
end

POMDPs.actions(::VisualMDP) = [:west, :east]
POMDPs.discount(::VisualMDP) = 0.95
POMDPs.isterminal(::VisualMDP, ::Vector{Float64}) = false
POMDPs.gen(::VisualMDP, state::Vector{Float64}, action::Symbol, rng::AbstractRNG) =
    (sp=state .+ [action === :east ? 0.25 : -0.25, 0.0], r=0.0)

@testset "Complete visualization pipelines" begin
    hypotheses = [
        ObjectiveHypothesis(:east, :east,
            BehaviorModel(fixed_planner(:east), EpsilonGreedyLikelihood(epsilon=0.1)), 0.5),
        ObjectiveHypothesis(:west, :west,
            BehaviorModel(fixed_planner(:west), EpsilonGreedyLikelihood(epsilon=0.1)), 0.5),
    ]
    problem = ObjectiveInferenceProblem(
        hypotheses=hypotheses,
        mdp_builder=(objective, hypothesis) -> VisualMDP(objective),
        state_adapter=(mdp, observation, timestep) -> Float64.(observation),
    )
    states = [[0.5, 1.0], [0.75, 1.0], [1.0, 1.0]]
    result = infer_objectives_exact(problem, states, fill(:east, 3))
    true_fn = (x, y) -> -((x - 2.0)^2 + (y - 1.0)^2)
    true_mdp = hypothesis_mdp(problem, :east)
    @test !isnothing(plot_particle_filter_frame(
        result, 2; true_objective_fn=true_fn, true_mdp=true_mdp,
        trace_from_current=false, gridsize=12,
    ))
    @test !isnothing(plot_particle_filter_frame(
        result, 2; true_objective_fn=true_fn, true_mdp=true_mdp,
        trace_from_current=true, gridsize=12,
    ))
    @test !isnothing(plot_particle_heatmaps_frame(
        result, 2; true_objective_fn=true_fn, true_mdp=true_mdp, gridsize=12,
    ))
    frames = [
        make_particle_filter_frame_fn(
            result; true_objective_fn=true_fn, true_mdp=true_mdp, gridsize=12,
        )(1),
        make_particle_heatmaps_frame_fn(
            result; true_objective_fn=true_fn, true_mdp=true_mdp, gridsize=12,
        )(2),
    ]
    @test animate_particle_filter_from_frames(frames; fps=3)[2] == 3

    world_problem = WorldInferenceProblem(
        context=tiny_scribe_context(), evidence=tiny_ergodic_evidence(),
    )
    world_result = infer_world(
        world_problem,
        [TrajectoryObservation(state=[0.0, 0.0], action=:move)],
        SMCConfig(n_particles=8, kernel=IdentityKernel(), seed=UInt64(13)),
    )
    @test !isnothing(plot_world_filter_frame(world_result, 1))
    @test !isnothing(plot_world_diagnostics(world_result))
    @test !isnothing(plot_world_modes(world_problem))
    @test !isnothing(plot_world_ancestry(world_result))
    deployment = deploy_behavior_information(
        world_problem, world_posterior(world_result); project=true,
    )
    @test !isnothing(plot_world_deployment(deployment))
end
