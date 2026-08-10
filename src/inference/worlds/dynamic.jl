export DynamicWorldState, DynamicWorldInferenceProblem, DynamicWorldKernel
export DynamicWorldPosteriorSummary, infer_dynamic_world

@with_kw_noshow struct DynamicWorldState
    initial_time::Int
    coefficients::Vector{Vector{Float64}}
end

@with_kw_noshow mutable struct DynamicWorldInferenceProblem <: AbstractBehaviorInferenceProblem
    context::SCRIBEWorldContext
    evidence::PlannerWorldEvidence
    trajectory::BehaviorTrajectory = BehaviorTrajectory()
    planning_cache::PlanningCache = PlanningCache()
    horizon::Int = 1
end

struct DynamicWorldKernel <: AbstractPairedKernel end

@with_kw_noshow struct DynamicWorldPosteriorSummary
    stage::InferenceStage
    current_mean::Vector{Float64}
    current_covariance::Matrix{Float64}
    path_means::Vector{Vector{Float64}}
    environment_times::Vector{Int}
end

dynamic_index(state::DynamicWorldState, environment_time) =
    environment_time - state.initial_time + 1
dynamic_coefficients(state::DynamicWorldState, environment_time) =
    state.coefficients[dynamic_index(state, environment_time)]
dynamic_last_time(state::DynamicWorldState) =
    state.initial_time + length(state.coefficients) - 1

initial_stage(problem::DynamicWorldInferenceProblem) = InferenceStage(
    environment_time=problem.context.model_time,
)

function initial_proposal(problem::DynamicWorldInferenceProblem, rng, cache)
    context = problem.context
    coefficients = gaussian_sample(rng, context.prior_mean, context.prior_covariance)
    (
        value=DynamicWorldState(
            initial_time=context.model_time,
            coefficients=[coefficients],
        ),
        trace=WorldTrace(),
        logdensity=gaussian_logdensity(
            coefficients, context.prior_mean, context.prior_covariance,
        ),
    )
end

function observe!(problem::DynamicWorldInferenceProblem, observation::TrajectoryObservation)
    previous_time = isempty(problem.trajectory.environment_times) ?
        problem.context.model_time : last(problem.trajectory.environment_times)
    observation.environment_time >= previous_time ||
        throw(ArgumentError("environment time must be monotone"))
    push!(problem.trajectory.states, observation.state)
    push!(problem.trajectory.actions, observation.action)
    push!(problem.trajectory.dwell_times, observation.dwell_time)
    push!(problem.trajectory.environment_times, observation.environment_time)
    problem
end

function stages_for_observation(::OneStagePerObservation,
                                problem::DynamicWorldInferenceProblem, observation,
                                cloud::ParticleCloud, cache)
    [InferenceStage(
        observation=cloud.stage.observation + 1,
        bridge=1,
        λ=1.0,
        environment_time=last(problem.trajectory.environment_times),
    )]
end

function dynamic_planner_loglikelihood(problem::DynamicWorldInferenceProblem,
                                       timestep, state::DynamicWorldState)
    evidence = problem.evidence
    environment_time = problem.trajectory.environment_times[timestep]
    coefficients = dynamic_coefficients(state, environment_time)
    model = candidate_model(problem.context, coefficients)
    mdp = evidence.mdp_builder(model, evidence.objective)
    planner_states = Any[evidence.state_adapter(mdp, problem.trajectory.states[t], t)
                         for t in 1:timestep]
    action_list = collect(POMDPs.actions(mdp))
    prior_actions = Any[resolve_action(problem.trajectory.actions[t], action_list)
                        for t in 1:max(timestep - 1, 0)]
    observed_action = resolve_action(problem.trajectory.actions[timestep], action_list)
    isnothing(observed_action) && return -Inf
    context = PlanningContext(
        hypothesis_id=:dynamic_world,
        timestep=timestep,
        states=planner_states,
        actions=prior_actions,
        horizon=problem.horizon,
        rng=MersenneTwister(hash((timestep, coefficients), UInt(0))),
        metadata=evidence.metadata,
    )
    artifact = prepare(evidence.behavior.planner, mdp, context)
    observation_loglikelihood(
        evidence.behavior.likelihood, evidence.behavior.planner, artifact,
        mdp, planner_states[end], observed_action, action_list, context,
    )
end


function validate_problem(problem::DynamicWorldInferenceProblem)
    gaussian_factor(problem.context.model.params.Q)
    nothing
end

function dynamic_compatibility(problem, t, state)
    sum((dynamic_planner_loglikelihood(problem, timestep, state) for timestep in 1:t);
        init=0.0)
end

function dynamic_logprior(problem, state::DynamicWorldState)
    context = problem.context
    value = gaussian_logdensity(
        first(state.coefficients), context.prior_mean, context.prior_covariance,
    )
    for index in 2:length(state.coefficients)
        value += gaussian_logdensity(
            state.coefficients[index], state.coefficients[index - 1],
            context.model.params.Q,
        )
    end
    value
end

function logtarget(problem::DynamicWorldInferenceProblem, stage::InferenceStage,
                   state::DynamicWorldState, cache)
    current = dynamic_compatibility(problem, stage.observation, state)
    previous = dynamic_compatibility(problem, max(stage.observation - 1, 0), state)
    dynamic_logprior(problem, state) + (1 - stage.λ) * previous + stage.λ * current
end

function propose(::DynamicWorldKernel, problem::DynamicWorldInferenceProblem,
                 old_stage::InferenceStage, new_stage::InferenceStage,
                 particle::WeightedParticle, rng, cache)
    old_state = particle.value
    new_time = new_stage.environment_time
    new_time <= dynamic_last_time(old_state) && return MoveRecord(
        value=old_state,
        log_forward=0.0,
        log_backward=0.0,
        branch=:dynamic_identity,
    )
    coefficients = deepcopy(old_state.coefficients)
    log_forward = 0.0
    for _ in dynamic_last_time(old_state) + 1:new_time
        previous = last(coefficients)
        current = gaussian_sample(rng, previous, problem.context.model.params.Q)
        log_forward += gaussian_logdensity(current, previous, problem.context.model.params.Q)
        push!(coefficients, current)
    end
    MoveRecord(
        value=DynamicWorldState(
            initial_time=old_state.initial_time,
            coefficients=coefficients,
        ),
        log_forward=log_forward,
        log_backward=0.0,
        branch=:physical_transition,
        metadata=(transitions=new_time - dynamic_last_time(old_state),),
    )
end

function update_trace(problem::DynamicWorldInferenceProblem, old_stage, new_stage,
                      particle, move, cache)
    particle.trace
end

function summarize(problem::DynamicWorldInferenceProblem, cloud::ParticleCloud,
                   stage::InferenceStage, cache)
    log_weights = [particle.log_weight for particle in cloud.particles]
    weights = exp.(log_weights .- logsumexp(log_weights))
    path_length = minimum(length(particle.value.coefficients) for particle in cloud.particles)
    path_means = [sum(weight .* particle.value.coefficients[index]
                      for (weight, particle) in zip(weights, cloud.particles))
                  for index in 1:path_length]
    current_values = [last(particle.value.coefficients) for particle in cloud.particles]
    current = weighted_mean_covariance(current_values, log_weights)
    DynamicWorldPosteriorSummary(
        stage=stage,
        current_mean=current.mean,
        current_covariance=current.covariance,
        path_means=path_means,
        environment_times=collect(problem.context.model_time:
                                  problem.context.model_time + path_length - 1),
    )
end

posterior(result::SMCResult{<:SequentialState{<:DynamicWorldInferenceProblem}}) =
    last(result.state.summaries)
function infer_dynamic_world(problem::DynamicWorldInferenceProblem, observations, config::SMCConfig)
    problem.horizon = length(observations)
    run_smc(problem, observations, config)
end
