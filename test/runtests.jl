ENV["GKSwstype"] = "100"
using Arrodes
using POMDPs
using Random
using Test

struct TinyMDP <: MDP{Int,Symbol}
    objective::Symbol
end

POMDPs.actions(::TinyMDP) = [:left, :right]
POMDPs.discount(::TinyMDP) = 0.95
POMDPs.isterminal(::TinyMDP, ::Int) = false
POMDPs.gen(::TinyMDP, s::Int, a::Symbol, rng::AbstractRNG) =
    (sp = s + (a === :right ? 1 : -1), r = 0.0)

@testset "Trace-preserving SMC with resampling and rejuvenation" begin
    right = CallbackPlanner(prepare_fn = (mdp, ctx) -> :right,
        action_fn = (value, mdp, state, ctx) -> value)
    left = CallbackPlanner(prepare_fn = (mdp, ctx) -> :left,
        action_fn = (value, mdp, state, ctx) -> value)
    model = DiscreteInferenceConfig(
        hypotheses = [
            ObjectiveHypothesis(:right, :right,
                BehaviorModel(right, EpsilonGreedyLikelihood(epsilon = 0.1)), 0.5),
            ObjectiveHypothesis(:left, :left,
                BehaviorModel(left, EpsilonGreedyLikelihood(epsilon = 0.1)), 0.5),
        ], mdp_builder = (objective, hypothesis) -> TinyMDP(objective))
    result = infer_objectives_smc(SMCInferenceConfig(model = model, n_particles = 128,
        ess_threshold = 0.95, rejuvenation_steps = 2), [0, 1, 2, 3], fill(:right, 4))
    @test result isa SMCFilterResult
    @test posterior(result)[1] > 0.95
    @test length(result.state.particles) == 128
    @test length(result.ancestry_history) == 4
    @test all(length(trace.score_history) == 4 for trace in result.state.particles)
    @test !isempty(result.state.resampling_times)
    @test result.state.rejuvenation_attempts > 0
    @test all(isfinite, result.ess_history)
end

@testset "Movement-noise plan likelihood" begin
    planner = CallbackPlanner(prepare_fn = (mdp, ctx) -> :right,
        action_fn = (value, mdp, state, ctx) -> value)
    context = PlanningContext(hypothesis_id = :right, timestep = 2,
        states = [0, 1], actions = [:right], horizon = 2, rng = MersenneTwister(4))
    artifact = prepare(planner, TinyMDP(:right), context)
    likelihood = MovementNoiseLikelihood(n_transition_samples = 8, bandwidth = 0.5)
    @test isfinite(observation_loglikelihood(likelihood, planner, artifact, TinyMDP(:right), 1,
        :right, [:left, :right], context))
end

@testset "Deprecated generic objective fields" begin
    rff = RandomFourierField(amplitude_max = 2.0)
    rbf = RadialBasisField(σ = 0.5)
    @test isfinite(make_component(rff, fourier_params_sampler(rff)(MersenneTwister(1)))(1.0, 2.0))
    @test isfinite(make_component(rbf, rbf_params_sampler(rbf)(MersenneTwister(1)))(1.0, 2.0))
end

struct FixedSolver <: Solver
    action::Symbol
end
struct FixedPolicy <: Policy
    action::Symbol
end
POMDPs.solve(solver::FixedSolver, ::TinyMDP) = FixedPolicy(solver.action)
POMDPs.action(policy::FixedPolicy, ::Int) = policy.action

@testset "Arrodes planning interface" begin
    context = PlanningContext(hypothesis_id = :right, states = [0], horizon = 3,
        rng = MersenneTwister(2))
    planner = POMDPSolverPlanner(FixedSolver(:right))
    artifact = prepare(planner, TinyMDP(:right), context)
    @test planned_action(planner, artifact, TinyMDP(:right), 0, context) === :right
    @test action_distribution(EpsilonGreedyLikelihood(epsilon = 0.2), planner, artifact,
        TinyMDP(:right), 0, [:left, :right], context) ≈ [0.1, 0.9]

    open_loop = OpenLoopPlanner((mdp, ctx) ->
        (states = [0, 1, 2], actions = [:right, :right]))
    open_artifact = prepare(open_loop, TinyMDP(:right), context)
    @test planned_action(open_loop, open_artifact, TinyMDP(:right), 0, context) === :right

    callback = CallbackPlanner(
        prepare_fn = (mdp, ctx) -> mdp.objective,
        scores_fn = (value, mdp, state, ctx) -> value === :right ? [0.0, 2.0] : [2.0, 0.0],
    )
    callback_artifact = prepare(callback, TinyMDP(:right), context)
    probabilities = action_distribution(BoltzmannScoreLikelihood(temperature = 1.0),
        callback, callback_artifact, TinyMDP(:right), 0, [:left, :right], context)
    @test probabilities[2] > probabilities[1]
    @test sum(probabilities) ≈ 1.0
    @test MCTSPlanner() isa AbstractPlanner
    @test SoftQPlanner(n_iterations = 1) isa AbstractPlanner
    @test VulcanMCTSPlanner((mdp, context) -> nothing) isa AbstractPlanner
    @test VulcanErgodicPlanner((mdp, state) -> nothing; n_steps = 2) isa AbstractPlanner
end

@testset "Exact discrete objective inference" begin
    preparations = Ref(0)
    right_planner = CallbackPlanner(
        prepare_fn = (mdp, ctx) -> (preparations[] += 1; mdp.objective),
        action_fn = (value, mdp, state, ctx) -> value,
    )
    left_planner = POMDPSolverPlanner(FixedSolver(:left))

    hypotheses = [
        ObjectiveHypothesis(
            id = :move_right,
            objective = :right,
            behavior = BehaviorModel(right_planner, EpsilonGreedyLikelihood(epsilon = 0.1)),
            prior_probability = 0.5,
        ),
        ObjectiveHypothesis(
            id = :move_left,
            objective = :left,
            behavior = BehaviorModel(left_planner, EpsilonGreedyLikelihood(epsilon = 0.1)),
            prior_probability = 0.5,
        ),
    ]

    config = DiscreteInferenceConfig(
        hypotheses = hypotheses,
        mdp_builder = (objective, hypothesis) -> TinyMDP(objective),
        state_adapter = (mdp, observation, timestep) ->
            observation isa AbstractArray ? first(observation) : observation,
    )
    result = infer_objectives(config, [0, 1, 2], [:right, :right, :right])

    @test size(result.posterior_history) == (2, 3)
    @test all(sum(result.posterior_history; dims = 1) .≈ 1.0)
    @test posterior(result)[1] > 0.99
    @test best_hypothesis(result).hypothesis.id === :move_right
    @test preparations[] == 1
    @test hypothesis_mdp(result.state, :move_right).objective === :right
    @test hypothesis_artifact(result.state, :move_right) isa CallbackArtifact

    indexed = infer_objectives(config, reshape([0, 1, 2], 1, :), [2, 2, 2])
    @test best_hypothesis(indexed).hypothesis.id === :move_right

    online = initialize_filter(config)
    update!(online, 0, :right; horizon = 3)
    update!(online, 1, :right; horizon = 3)
    @test online.timestep == 2
    @test posterior(online)[1] > posterior(online)[2]
end

@testset "History-sensitive planning" begin
    preparations = Ref(0)
    planner = CallbackPlanner(
        prepare_fn = (mdp, ctx) -> (preparations[] += 1; ctx.timestep),
        action_fn = (prepared_at, mdp, state, ctx) -> :right,
        scope = :history,
    )
    hypothesis = ObjectiveHypothesis(
        :history,
        :right,
        BehaviorModel(planner, EpsilonGreedyLikelihood(epsilon = 0.1)),
        1.0,
    )
    config = DiscreteInferenceConfig(
        hypotheses = [hypothesis],
        mdp_builder = (objective, hypothesis) -> TinyMDP(objective),
    )
    infer_objectives(config, [0, 1, 2], [:right, :right, :right])
    @test preparations[] == 3
end

@testset "Validation" begin
    planner = POMDPSolverPlanner(FixedSolver(:right))
    behavior = BehaviorModel(planner, EpsilonGreedyLikelihood())
    @test_throws ArgumentError ObjectiveHypothesis(:bad, :right, behavior, 0.0)
    duplicate = [
        ObjectiveHypothesis(:same, :right, behavior, 0.5),
        ObjectiveHypothesis(:same, :left, behavior, 0.5),
    ]
    config = DiscreteInferenceConfig(hypotheses = duplicate,
        mdp_builder = (objective, hypothesis) -> TinyMDP(objective))
    @test_throws ArgumentError infer_objectives(config, [0], [:right])
end

struct VisualMDP <: MDP{Vector{Float64},Symbol}
    objective::Symbol
    dimensions::Tuple{Float64,Float64}
    obj::Function
end

function VisualMDP(objective::Symbol)
    center = objective === :east ? [2.0, 1.0] : [0.0, 1.0]
    reward = state -> (-sum(abs2, state .- center), false)
    return VisualMDP(objective, (0.0, 2.0), reward)
end

POMDPs.actions(::VisualMDP) = [:west, :east]
POMDPs.discount(::VisualMDP) = 0.95
POMDPs.isterminal(::VisualMDP, ::Vector{Float64}) = false
POMDPs.gen(::VisualMDP, state::Vector{Float64}, action::Symbol, rng::AbstractRNG) =
    (sp = state .+ [action === :east ? 0.25 : -0.25, 0.0], r = 0.0)

@testset "Complete visualization infrastructure" begin
    east = CallbackPlanner(
        prepare_fn = (mdp, context) -> :east,
        action_fn = (action, mdp, state, context) -> action,
    )
    west = CallbackPlanner(
        prepare_fn = (mdp, context) -> :west,
        action_fn = (action, mdp, state, context) -> action,
    )
    hypotheses = [
        ObjectiveHypothesis(:east, :east,
            BehaviorModel(east, EpsilonGreedyLikelihood(epsilon = 0.1)), 0.5),
        ObjectiveHypothesis(:west, :west,
            BehaviorModel(west, EpsilonGreedyLikelihood(epsilon = 0.1)), 0.5),
    ]
    config = DiscreteInferenceConfig(
        hypotheses = hypotheses,
        mdp_builder = (objective, hypothesis) -> VisualMDP(objective),
        state_adapter = (mdp, observation, timestep) -> Float64.(observation),
    )
    states = [0.5 0.75 1.0; 1.0 1.0 1.0]
    result = infer_objectives(config, states, [:east, :east, :east])
    true_fn = (x, y) -> -((x - 2.0)^2 + (y - 1.0)^2)
    true_mdp = hypothesis_mdp(result.state, :east)

    from_start = plot_particle_filter_frame(result, 2;
        true_objective_fn = true_fn, true_mdp = true_mdp,
        trace_from_current = false, gridsize = 12)
    from_current = plot_particle_filter_frame(result, 2;
        true_objective_fn = true_fn, true_mdp = true_mdp,
        trace_from_current = true, gridsize = 12)
    heatmaps = plot_particle_heatmaps_frame(result, 2;
        true_objective_fn = true_fn, true_mdp = true_mdp, gridsize = 12)
    final_plot = plot_particle_filter_explanation(result;
        true_objective_fn = true_fn, true_mdp = true_mdp, gridsize = 12)
    @test !isnothing(from_start)
    @test !isnothing(from_current)
    @test !isnothing(heatmaps)
    @test !isnothing(final_plot)

    start_fn = make_particle_filter_frame_fn(result;
        true_objective_fn = true_fn, true_mdp = true_mdp,
        trace_from_current = false, gridsize = 12)
    current_fn = make_particle_filter_frame_fn(result;
        true_objective_fn = true_fn, true_mdp = true_mdp,
        trace_from_current = true, gridsize = 12)
    heatmaps_fn = make_particle_heatmaps_frame_fn(result;
        true_objective_fn = true_fn, true_mdp = true_mdp, gridsize = 12)
    frames = [start_fn(1), current_fn(2), heatmaps_fn(3)]
    animation, fps = animate_particle_filter_from_frames(frames; fps = 3)
    @test !isnothing(animation)
    @test fps == 3
end
