initial_stage(problem::WorldInferenceProblem) = InferenceStage(
    environment_time=problem.context.model_time,
)

function stages_for_observation(
    ::OneStagePerObservation,
    problem::WorldInferenceProblem,
    observation,
    cloud::ParticleCloud,
    cache,
)
    [InferenceStage(
        observation=cloud.stage.observation + 1,
        bridge=1,
        λ=1.0,
        environment_time=last(problem.trajectory.environment_times),
    )]
end

function observe!(problem::WorldInferenceProblem, observation::TrajectoryObservation)
    observation.environment_time == problem.context.model_time ||
        throw(ArgumentError("static world inference freezes SCRIBE model time"))
    observation.dwell_time > 0 || throw(ArgumentError("dwell time must be positive"))
    push!(problem.trajectory.states, observation.state)
    push!(problem.trajectory.actions, observation.action)
    push!(problem.trajectory.dwell_times, observation.dwell_time)
    push!(problem.trajectory.environment_times, observation.environment_time)
    problem
end

function validate_problem(problem::WorldInferenceProblem)
    context = problem.context
    dimension = length(context.prior_mean)
    size(context.prior_covariance) == (dimension, dimension) ||
        throw(DimensionMismatch("SCRIBE coefficient prior has incompatible dimensions"))
    size(context.quadrature, 1) == length(context.quadrature_weights) ||
        throw(DimensionMismatch("quadrature locations and weights must agree"))
    if problem.evidence isa DirectErgodicEvidence
        config = problem.evidence.energy
        min(config.discrepancy_scale, config.reward_scale,
            problem.evidence.kernel.bandwidth) > 0 ||
            throw(ArgumentError("world energy scales and kernel bandwidth must be positive"))
        config.evaluation in (:combined, :mmd, :reward) ||
            throw(ArgumentError("world evaluation must be :combined, :mmd, or :reward"))
        config.evaluation !== :reward || !isnothing(problem.evidence.reward) ||
            throw(ArgumentError("reward-only evaluation requires a reward function"))
    end
    nothing
end

function bridge_compatibility(problem, stage, coefficients, cache)
    world_compatibility(
        problem, problem.evidence, stage.observation, coefficients, cache,
    )
end

function logtarget(problem::WorldInferenceProblem, stage::InferenceStage, coefficients, cache)
    context = problem.context
    factor = get!(cache, :world_prior_factor) do
        gaussian_factor(context.prior_covariance)
    end
    gaussian_logdensity(coefficients, context.prior_mean, factor) +
        bridge_compatibility(problem, stage, coefficients, cache)
end

function update_trace(problem::WorldInferenceProblem, old_stage, new_stage,
                      particle, move, cache)
    records = copy(particle.trace.moves)
    push!(records, WorldMoveRecord(
        stage=new_stage,
        branch=move.branch,
        transport_fraction=get(move.metadata, :transport_fraction, 1.0),
        log_forward=move.log_forward,
        log_backward=move.log_backward,
    ))
    WorldTrace(moves=records)
end
