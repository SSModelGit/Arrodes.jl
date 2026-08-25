struct ObjectiveBehaviorFactor <: Gen.Distribution{Bool} end

const objective_behavior_factor = ObjectiveBehaviorFactor()

function Gen.logpdf(
    ::ObjectiveBehaviorFactor,
    observed::Bool,
    problem::ObjectiveInferenceProblem,
    cache::Dict{Symbol,Any},
    objective_index::Int,
    timestep::Int,
)
    observed || return -Inf
    sum(
        objective_loglikelihood(problem, cache, objective_index, time)
        for time in 1:timestep;
        init=0.0,
    )
end

Gen.random(::ObjectiveBehaviorFactor, problem, cache, objective_index, timestep) = true
Gen.is_discrete(::ObjectiveBehaviorFactor) = true
Gen.has_output_grad(::ObjectiveBehaviorFactor) = false
Gen.has_argument_grads(::ObjectiveBehaviorFactor) = (false, false, false, false)

@gen function objective_model(
    problem::ObjectiveInferenceProblem,
    cache::Dict{Symbol,Any},
    timestep::Int,
)
    objective = {:objective} ~ Gen.categorical(objective_priors(problem))
    {:behavior} ~ objective_behavior_factor(problem, cache, objective, timestep)
    objective
end

function objective_transition(problem::ObjectiveInferenceProblem, refresh_probability)
    priors = objective_priors(problem)
    count = length(priors)
    (1 - refresh_probability) .* Matrix{Float64}(I, count, count) .+
        refresh_probability .* repeat(priors', count, 1)
end

function objective_backward_matrix(transition, priors)
    backward = similar(transition)
    for new in axes(transition, 2)
        backward[:, new] = priors .* transition[:, new]
        backward[:, new] ./= sum(backward[:, new])
    end
    backward
end

@kernel function objective_forward(previous_trace, transition::Matrix{Float64})
    previous = GenTraceKernelDSL.get_undualed(previous_trace, :objective)
    objective ~ Gen.categorical(transition[previous, :])
    return Gen.choicemap((:objective, objective)), Gen.choicemap(
        (:previous_objective, previous_trace[:objective]),
    )
end

@kernel function objective_backward(updated_trace, backward::Matrix{Float64})
    objective = GenTraceKernelDSL.get_undualed(updated_trace, :objective)
    previous_objective ~ Gen.categorical(backward[:, objective])
    return Gen.choicemap((:objective, previous_objective)), Gen.choicemap(
        (:objective, updated_trace[:objective]),
    )
end

@kernel function objective_rejuvenation(trace, transition::Matrix{Float64})
    previous = GenTraceKernelDSL.get_undualed(trace, :objective)
    objective ~ Gen.categorical(transition[previous, :])
    return Gen.choicemap((:objective, objective)), Gen.choicemap(
        (:objective, trace[:objective]),
    )
end

function objective_mh(trace, transition, rng)
    proposed, log_acceptance = GenTraceKernelDSL.run_mcmc_kernel(
        trace,
        objective_rejuvenation,
        (transition,),
    )
    log(rand(rng)) < log_acceptance ? (proposed, true) : (trace, false)
end

function objective_probabilities(problem, state)
    probabilities = zeros(length(problem.hypotheses))
    weights = GenParticleFilters.get_norm_weights(state)
    for (trace, weight) in zip(state.traces, weights)
        probabilities[trace[:objective]] += weight
    end
    probabilities
end

objective_probabilities(result::ObjectiveInferenceResult) =
    result.posterior_history[:, end]

objective_probabilities(result::ObjectiveInferenceResult, timestep::Int) =
    result.posterior_history[:, timestep + 1]

objective_observation_count(result::ObjectiveInferenceResult) =
    size(result.posterior_history, 2) - 1

function top_objective_hypotheses(result, problem, timestep, count)
    probabilities = objective_probabilities(result, timestep)
    indices = sortperm(probabilities; rev=true)[1:min(count, length(probabilities))]
    [
        Dict(
            :index => index,
            :hypothesis => problem.hypotheses[index],
            :probability => probabilities[index],
        )
        for index in indices
    ]
end

function best_hypothesis(result::ObjectiveInferenceResult)
    probabilities = objective_probabilities(result)
    index = argmax(probabilities)
    Dict(
        :index => index,
        :id => result.hypothesis_ids[index],
        :probability => probabilities[index],
    )
end

function infer_objectives(
    problem::ObjectiveInferenceProblem;
    n_particles::Int=256,
    refresh_probability::Float64=0.05,
    ess_threshold::Float64=0.5,
    resampling::Symbol=:residual,
    rejuvenation_steps::Int=1,
    check_inverses::Bool=false,
    rng::AbstractRNG=Random.default_rng(),
)
    cache = Dict{Symbol,Any}(:rng => rng)
    priors = objective_priors(problem)
    transition = objective_transition(problem, refresh_probability)
    backward = objective_backward_matrix(transition, priors)
    constraints = Gen.choicemap((:behavior, true))
    state = GenParticleFilters.pf_initialize(
        objective_model,
        (problem, cache, 0),
        constraints,
        n_particles;
        dynamic=true,
    )
    horizon = length(problem.actions)
    history = Matrix{Float64}(undef, length(problem.hypotheses), horizon + 1)
    history[:, 1] = objective_probabilities(problem, state)
    ess = Vector{Float64}(undef, horizon + 1)
    ess[1] = GenParticleFilters.effective_sample_size(state)
    resampled = falses(horizon)

    update = GenSMCP3.SMCP3Update(
        objective_forward,
        objective_backward,
        (transition,),
        (backward,),
        check_inverses,
    )
    for timestep in 1:horizon
        GenParticleFilters.pf_update!(
            state,
            (problem, cache, timestep),
            (
                Gen.UnknownChange(),
                Gen.UnknownChange(),
                Gen.UnknownChange(),
            ),
            constraints,
            update,
        )
        ess[timestep + 1] = GenParticleFilters.effective_sample_size(state)
        if ess[timestep + 1] < ess_threshold * n_particles
            GenParticleFilters.pf_resample!(state, resampling)
            resampled[timestep] = true
            if rejuvenation_steps > 0
                move = (trace, transition) -> objective_mh(trace, transition, rng)
                GenParticleFilters.pf_rejuvenate!(
                    state,
                    move,
                    (transition,),
                    rejuvenation_steps,
                )
            end
        end
        history[:, timestep + 1] = objective_probabilities(problem, state)
    end

    ObjectiveInferenceResult(
        hypothesis_ids=getfield.(problem.hypotheses, :id),
        posterior_history=history,
        ess_history=ess,
        resampled=resampled,
    )
end

function exact_objective_probabilities(problem::ObjectiveInferenceProblem)
    cache = Dict{Symbol,Any}(:rng => Random.default_rng())
    log_weights = log.(objective_priors(problem))
    for timestep in eachindex(problem.actions)
        log_weights .+= [
            objective_loglikelihood(problem, cache, index, timestep)
            for index in eachindex(problem.hypotheses)
        ]
    end
    log_weights .-= maximum(log_weights)
    weights = exp.(log_weights)
    weights ./ sum(weights)
end
