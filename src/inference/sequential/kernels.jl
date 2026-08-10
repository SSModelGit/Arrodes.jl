function propose(::IdentityKernel, problem::AbstractBehaviorInferenceProblem,
                 old_stage::InferenceStage, new_stage::InferenceStage,
                 particle::WeightedParticle, rng, cache)
    MoveRecord(
        value=particle.value,
        log_forward=0.0,
        log_backward=0.0,
        branch=:identity,
    )
end

function paired_logweight(problem, old_stage, new_stage, particle, move, cache)
    logtarget(problem, new_stage, move.value, cache) -
    logtarget(problem, old_stage, particle.value, cache) +
    move.log_backward - move.log_forward + move.log_jacobian
end

update_trace(problem, old_stage, new_stage, particle, move, cache) = deepcopy(particle.trace)
prepare_kernel(kernel, problem, old_stage, new_stage, cloud, cache) = kernel

step(::NoInvariantMove, problem, stage, particle, rng, cache) = (particle, false)
