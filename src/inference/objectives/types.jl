@with_kw_noshow struct ObjectiveHypothesis
    id::Symbol
    objective
    behavior::BehaviorModel
    prior_probability::Float64
    metadata::Dict{Symbol,Any} = Dict{Symbol,Any}()
end

@with_kw_noshow struct ObjectiveInferenceProblem
    hypotheses::Vector{ObjectiveHypothesis}
    mdp_builder::Function
    states::Vector
    actions::Vector
    state_adapter::Function = (mdp, state, timestep) -> state
end

@with_kw_noshow struct ObjectiveInferenceResult
    hypothesis_ids::Vector{Symbol}
    posterior_history::Matrix{Float64}
    ess_history::Vector{Float64}
    resampled::Vector{Bool}
end

function objective_priors(problem::ObjectiveInferenceProblem)
    priors = [hypothesis.prior_probability for hypothesis in problem.hypotheses]
    isempty(priors) && error("Objective inference requires at least one hypothesis")
    any(probability -> !isfinite(probability) || probability < 0, priors) &&
        error("Objective prior probabilities must be finite and nonnegative")
    sum(priors) > 0 || error("At least one objective must have positive prior mass")
    priors ./ sum(priors)
end

function hypothesis_index(problem::ObjectiveInferenceProblem, id::Symbol)
    findfirst(hypothesis -> hypothesis.id == id, problem.hypotheses)
end

function hypothesis_mdp(problem::ObjectiveInferenceProblem, index::Int)
    hypothesis = problem.hypotheses[index]
    problem.mdp_builder(hypothesis.objective, hypothesis)
end

function hypothesis_mdp(problem, cache, index::Int)
    hypothesis = problem.hypotheses[index]
    mdps = get!(cache, :mdps) do
        Dict{Symbol,Any}()
    end
    get!(mdps, hypothesis.id) do
        problem.mdp_builder(hypothesis.objective, hypothesis)
    end
end

hypothesis_mdp(problem::ObjectiveInferenceProblem, id::Symbol) =
    hypothesis_mdp(problem, hypothesis_index(problem, id))
