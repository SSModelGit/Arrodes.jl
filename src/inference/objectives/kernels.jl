export ObjectiveReplayMove, ObjectiveDiscreteKernel

@with_kw_noshow struct ObjectiveReplayMove <: AbstractInvariantMove
    proposal::Union{Nothing,Function} = nothing
end

"""A fully corrected discrete paired proposal; identity propagation remains the default."""
@with_kw_noshow struct ObjectiveDiscreteKernel <: AbstractPairedKernel
    proposal::Function
end

function propose(kernel::ObjectiveDiscreteKernel, problem::ObjectiveInferenceProblem,
                 old_stage::InferenceStage, new_stage::InferenceStage,
                 particle::WeightedParticle, rng, cache)
    proposed = kernel.proposal(rng, particle.value, problem, old_stage, new_stage)
    MoveRecord(
        value=proposed.index,
        log_forward=proposed.log_forward,
        log_backward=proposed.log_backward,
        branch=:discrete,
    )
end

function objective_trace(problem, index, timestep, cache)
    scores = [objective_loglikelihood(problem, index, t, cache) for t in 1:timestep]
    ObjectiveTrace(
        score_history=scores,
        hypothesis_history=fill(index, timestep),
        cumulative_score=sum(scores),
    )
end

function update_trace(problem::ObjectiveInferenceProblem, old_stage, new_stage,
                      particle, move, cache)
    objective_trace(problem, move.value, new_stage.observation, cache)
end

function _objective_replay_proposal(move, rng, current, problem, stage)
    if isnothing(move.proposal)
        probabilities = exp.(objective_logpriors(problem))
        candidate = searchsortedfirst(cumsum(probabilities), rand(rng))
        return (
            index=candidate,
            log_forward=log(probabilities[candidate]),
            log_reverse=log(probabilities[current]),
        )
    end
    move.proposal(rng, current, problem, stage)
end

function step(move::ObjectiveReplayMove, problem::ObjectiveInferenceProblem,
              stage::InferenceStage, particle::WeightedParticle, rng, cache)
    proposed = _objective_replay_proposal(move, rng, particle.value, problem, stage)
    log_acceptance = logtarget(problem, stage, proposed.index, cache) -
        logtarget(problem, stage, particle.value, cache) +
        proposed.log_reverse - proposed.log_forward
    log(rand(rng)) >= min(0.0, log_acceptance) && return particle, false
    candidate = WeightedParticle(
        value=proposed.index,
        trace=objective_trace(problem, proposed.index, stage.observation, cache),
        log_weight=particle.log_weight,
        lineage=particle.lineage,
        branch=:replay_mh,
    )
    candidate, true
end
