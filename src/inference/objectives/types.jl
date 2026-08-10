export ObjectiveHypothesis, BehaviorObservation, ObjectiveInferenceProblem
export ObjectiveTrace, ObjectivePosteriorSummary, ExactObjectiveResult
export hypothesis_index, best_hypothesis, posterior, log_posterior
export hypothesis_mdp, hypothesis_artifact

@with_kw_noshow struct ObjectiveHypothesis
    id::Symbol
    objective
    behavior::BehaviorModel
    prior_probability::Float64
    metadata = NamedTuple()

    function ObjectiveHypothesis(id::Symbol, objective, behavior::BehaviorModel,
                                 prior_probability::Float64, metadata)
        prior_probability > 0 && isfinite(prior_probability) ||
            throw(ArgumentError("objective prior mass must be finite and positive"))
        new(id, objective, behavior, prior_probability, metadata)
    end
end

function ObjectiveHypothesis(id::Symbol, objective, behavior::BehaviorModel,
                             prior_probability::Real)
    prior_probability > 0 && isfinite(prior_probability) ||
        throw(ArgumentError("objective prior mass must be finite and positive"))
    ObjectiveHypothesis(
        id=id,
        objective=objective,
        behavior=behavior,
        prior_probability=Float64(prior_probability),
    )
end

@with_kw struct BehaviorObservation
    state
    action
    horizon::Int
end

@with_kw_noshow mutable struct ObjectiveInferenceProblem <: AbstractBehaviorInferenceProblem
    hypotheses::Vector{ObjectiveHypothesis}
    mdp_builder::Function
    state_adapter::Function = (mdp, observation, timestep) -> observation
    seed::UInt64 = 0x6f62_6a65_6374_6976
    planning_cache::PlanningCache = PlanningCache()
    states::Vector{Any} = Any[]
    actions::Vector{Any} = Any[]
    horizon::Int = 1
end

@with_kw_noshow mutable struct ObjectiveTrace
    score_history::Vector{Float64} = Float64[]
    hypothesis_history::Vector{Int} = Int[]
    cumulative_score::Float64 = 0.0
end

@with_kw_noshow struct ObjectivePosteriorSummary
    stage::InferenceStage
    ids::Vector{Symbol}
    probabilities::Vector{Float64}
    log_probabilities::Vector{Float64}
    best_index::Int
end

@with_kw_noshow struct ExactObjectiveResult
    problem::ObjectiveInferenceProblem
    posterior_history::Matrix{Float64}
    log_normalizer_history::Vector{Float64}
    score_history::Matrix{Float64}
end

function hypothesis_index(problem::ObjectiveInferenceProblem, id::Symbol)
    index = findfirst(hypothesis -> hypothesis.id === id, problem.hypotheses)
    isnothing(index) && throw(KeyError(id))
    index
end

function validate_problem(problem::ObjectiveInferenceProblem)
    isempty(problem.hypotheses) && throw(ArgumentError("at least one objective is required"))
    ids = getfield.(problem.hypotheses, :id)
    length(ids) == length(unique(ids)) ||
        throw(ArgumentError("objective hypothesis IDs must be unique"))
    nothing
end

function hypothesis_mdp(problem::ObjectiveInferenceProblem, index::Int)
    hypothesis = problem.hypotheses[index]
    get!(problem.planning_cache.mdps, hypothesis.id) do
        problem.mdp_builder(hypothesis.objective, hypothesis)
    end
end

hypothesis_mdp(problem::ObjectiveInferenceProblem, id::Symbol) =
    hypothesis_mdp(problem, hypothesis_index(problem, id))

posterior(summary::ObjectivePosteriorSummary) = summary.probabilities
log_posterior(summary::ObjectivePosteriorSummary) = summary.log_probabilities
posterior(result::ExactObjectiveResult) = result.posterior_history[:, end]
log_posterior(result::ExactObjectiveResult) = log.(posterior(result))

function posterior(state::SequentialState{<:ObjectiveInferenceProblem})
    summarize(state.problem, state.cloud, state.cloud.stage, state.cache).probabilities
end

posterior(result::SMCResult{<:SequentialState{<:ObjectiveInferenceProblem}}) =
    posterior(result.state)
log_posterior(result::SMCResult{<:SequentialState{<:ObjectiveInferenceProblem}}) =
    log.(posterior(result))

function best_hypothesis(problem::ObjectiveInferenceProblem, probabilities)
    index = argmax(probabilities)
    (hypothesis=problem.hypotheses[index], probability=probabilities[index], index=index)
end

best_hypothesis(result::ExactObjectiveResult) =
    best_hypothesis(result.problem, posterior(result))
best_hypothesis(result::SMCResult{<:SequentialState{<:ObjectiveInferenceProblem}}) =
    best_hypothesis(result.state.problem, posterior(result))
