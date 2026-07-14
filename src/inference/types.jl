"""
    ObjectiveHypothesis

A named, user-specified objective and the behavior model expected to pursue it.
The objective may be any value understood by the configured `mdp_builder`.
"""
struct ObjectiveHypothesis{O,B<:BehaviorModel,M}
    id::Symbol
    objective::O
    behavior::B
    prior_probability::Float64
    metadata::M

    function ObjectiveHypothesis(id::Symbol, objective::O, behavior::B,
                                 prior_probability::Real,
                                 metadata::M = NamedTuple()) where {O,B<:BehaviorModel,M}
        isfinite(prior_probability) && prior_probability > 0 ||
            throw(ArgumentError("prior_probability for $id must be finite and positive"))
        new{O,B,M}(id, objective, behavior, Float64(prior_probability), metadata)
    end
end

function ObjectiveHypothesis(; id::Symbol, objective, behavior::BehaviorModel,
                             prior_probability::Real, metadata = NamedTuple())
    return ObjectiveHypothesis(id, objective, behavior, prior_probability, metadata)
end

"""
Configuration for exact filtering over a finite set of objective hypotheses.

`mdp_builder(objective, hypothesis)` constructs the MuKumari-compatible MDP.
`state_adapter(mdp, observation, timestep)` converts each supplied state observation
to the state representation expected by the MDP and planner.
"""
Base.@kwdef struct DiscreteInferenceConfig{H,F,S,M}
    hypotheses::H
    mdp_builder::F
    state_adapter::S = (mdp, observation, _t) -> observation
    seed::UInt64 = 0x7a6f_6465_7300_0001
    metadata::M = NamedTuple()
end

abstract type AbstractInferenceResult end

mutable struct DiscreteFilterState{C}
    config::C
    cache::PlanningCache
    log_weights::Vector{Float64}
    timestep::Int
    states::Vector{Any}
    actions::Vector{Any}
    log_evidence::Float64
end

struct DiscreteFilterResult{S} <: AbstractInferenceResult
    state::S
    posterior_history::Matrix{Float64}
    log_evidence_history::Vector{Float64}
end

"""One persistent SMC trace, including its current ancestry and score history."""
mutable struct ParticleTrace
    hypothesis_index::Int
    ancestor::Int
    log_weight::Float64
    log_likelihood::Float64
    score_history::Vector{Float64}
    hypothesis_history::Vector{Int}
end

"""
Configuration for sequential Monte Carlo over the named objective hypotheses.

Particles are resampled when their effective sample size falls below
`ess_threshold * n_particles`, then rejuvenated with prior-independent
Metropolis-Hastings moves. This is the resample-move/P3 boundary: users may
replace `proposal` with a domain-informed hypothesis proposal returning an index.
"""
Base.@kwdef struct SMCInferenceConfig{C,P}
    model::C
    n_particles::Int = 256
    ess_threshold::Float64 = 0.5
    rejuvenation_steps::Int = 2
    proposal::P = nothing
    resampling::Symbol = :systematic
end

mutable struct SMCFilterState{C,R}
    config::C
    cache::PlanningCache
    particles::Vector{ParticleTrace}
    timestep::Int
    states::Vector{Any}
    actions::Vector{Any}
    log_evidence::Float64
    rng::R
    resampling_times::Vector{Int}
    rejuvenation_accepts::Int
    rejuvenation_attempts::Int
end

struct SMCFilterResult{S} <: AbstractInferenceResult
    state::S
    posterior_history::Matrix{Float64}
    log_evidence_history::Vector{Float64}
    ess_history::Vector{Float64}
    ancestry_history::Vector{Vector{Int}}
end

posterior(state::DiscreteFilterState) = exp.(state.log_weights)
log_posterior(state::DiscreteFilterState) = copy(state.log_weights)
posterior(result::DiscreteFilterResult) = posterior(result.state)
log_posterior(result::DiscreteFilterResult) = log_posterior(result.state)

function posterior(state::SMCFilterState)
    n = length(state.config.model.hypotheses)
    probabilities = zeros(n)
    weights = exp.([particle.log_weight for particle in state.particles])
    weights ./= sum(weights)
    for (particle, weight) in zip(state.particles, weights)
        probabilities[particle.hypothesis_index] += weight
    end
    return probabilities
end

log_posterior(state::SMCFilterState) = log.(posterior(state))
posterior(result::SMCFilterResult) = posterior(result.state)
log_posterior(result::SMCFilterResult) = log_posterior(result.state)

function hypothesis_index(config::DiscreteInferenceConfig, id::Symbol)
    idx = findfirst(h -> h.id === id, config.hypotheses)
    idx === nothing && throw(KeyError(id))
    return idx
end

function best_hypothesis(state::DiscreteFilterState)
    idx = argmax(state.log_weights)
    return (hypothesis = state.config.hypotheses[idx], probability = exp(state.log_weights[idx]),
        index = idx)
end

best_hypothesis(result::DiscreteFilterResult) = best_hypothesis(result.state)

function best_hypothesis(state::SMCFilterState)
    probabilities = posterior(state)
    idx = argmax(probabilities)
    return (hypothesis = state.config.model.hypotheses[idx], probability = probabilities[idx],
        index = idx)
end

best_hypothesis(result::SMCFilterResult) = best_hypothesis(result.state)

function _validate_config(config::DiscreteInferenceConfig)
    isempty(config.hypotheses) && throw(ArgumentError("at least one objective hypothesis is required"))
    ids = [h.id for h in config.hypotheses]
    length(unique(ids)) == length(ids) || throw(ArgumentError("hypothesis IDs must be unique"))
    return config
end
