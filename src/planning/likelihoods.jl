Base.@kwdef struct BoltzmannScoreLikelihood <: AbstractActionLikelihood
    temperature::Float64 = 1.0
end

function action_distribution(likelihood::BoltzmannScoreLikelihood, planner::AbstractPlanner,
                             artifact::AbstractPlanArtifact, mdp, state, action_list,
                             context::PlanningContext)
    scores = action_scores(planner, artifact, mdp, state, context)
    scores === nothing && throw(ArgumentError(
        "$(typeof(planner)) does not expose action scores required by BoltzmannScoreLikelihood"))
    q = Float64.(vec(scores))
    length(q) == length(action_list) || throw(DimensionMismatch("score count does not match action count"))
    temperature = max(likelihood.temperature, eps(Float64))
    logits = (q .- maximum(q)) ./ temperature
    weights = exp.(logits)
    return _validate_distribution(weights, length(action_list))
end

Base.@kwdef struct EpsilonGreedyLikelihood <: AbstractActionLikelihood
    epsilon::Float64 = 0.05
end

function action_distribution(likelihood::EpsilonGreedyLikelihood, planner::AbstractPlanner,
                             artifact::AbstractPlanArtifact, mdp, state, action_list,
                             context::PlanningContext)
    0 <= likelihood.epsilon <= 1 || throw(ArgumentError("epsilon must lie in [0, 1]"))
    n = length(action_list)
    n > 0 || throw(ArgumentError("MDP action space is empty"))
    selected = planned_action(planner, artifact, mdp, state, context)
    idx = _action_index(action_list, selected)
    probabilities = fill(likelihood.epsilon / n, n)
    probabilities[idx] += 1 - likelihood.epsilon
    return probabilities
end

Base.@kwdef struct PlanTrackingLikelihood <: AbstractActionLikelihood
    epsilon::Float64 = 0.05
end

function action_distribution(likelihood::PlanTrackingLikelihood, planner::AbstractPlanner,
                             artifact::AbstractPlanArtifact, mdp, state, action_list,
                             context::PlanningContext)
    return action_distribution(EpsilonGreedyLikelihood(likelihood.epsilon), planner, artifact,
        mdp, state, action_list, context)
end

"""User-supplied action distribution callback."""
struct CallbackLikelihood{F} <: AbstractActionLikelihood
    distribution_fn::F
end

function action_distribution(likelihood::CallbackLikelihood, planner::AbstractPlanner,
                             artifact::AbstractPlanArtifact, mdp, state, action_list,
                             context::PlanningContext)
    probabilities = likelihood.distribution_fn(planner, artifact, mdp, state, action_list, context)
    return _validate_distribution(probabilities, length(action_list))
end

function _state_vector(state)
    values = hasproperty(state, :x) ? getproperty(state, :x) : state
    values isa AbstractMatrix && return Float64.(vec(values))
    if values isa AbstractVector && !isempty(values) && first(values) isa Tuple
        return Float64.(collect(first(values)))
    end
    return Float64.(vec(collect(values)))
end

_default_state_distance(left, right) = sqrt(sum(abs2, _state_vector(left) .- _state_vector(right)))

function observation_loglikelihood(likelihood::AbstractActionLikelihood, planner::AbstractPlanner,
                                   artifact::AbstractPlanArtifact, mdp, state, observed_action,
                                   action_list, context::PlanningContext)
    probabilities = action_distribution(likelihood, planner, artifact, mdp, state, action_list, context)
    index = findfirst(isequal(observed_action), action_list)
    index === nothing && return -Inf
    return probabilities[index] > 0 ? log(probabilities[index]) : -Inf
end

function observation_loglikelihood(likelihood::MovementNoiseLikelihood, planner::AbstractPlanner,
                                   artifact::AbstractPlanArtifact, mdp, state, observed_action,
                                   action_list, context::PlanningContext)
    likelihood.n_transition_samples > 0 || throw(ArgumentError("n_transition_samples must be positive"))
    likelihood.bandwidth > 0 || throw(ArgumentError("bandwidth must be positive"))
    action_term = observation_loglikelihood(
        EpsilonGreedyLikelihood(epsilon = likelihood.action_epsilon), planner, artifact, mdp,
        state, observed_action, action_list, context)
    length(context.states) < 2 && return action_term
    previous_state = context.states[end - 1]
    previous_context = PlanningContext(
        hypothesis_id = context.hypothesis_id,
        timestep = max(context.timestep - 1, 1),
        states = context.states[1:end-1],
        actions = context.actions,
        horizon = context.horizon,
        rng = context.rng,
        metadata = context.metadata,
    )
    nominal_action = planned_action(planner, artifact, mdp, previous_state, previous_context)
    distance = isnothing(likelihood.state_distance) ? _default_state_distance : likelihood.state_distance
    kernel_sum = 0.0
    for _ in 1:likelihood.n_transition_samples
        transition = POMDPs.gen(mdp, previous_state, nominal_action, context.rng)
        d = distance(transition.sp, state)
        kernel_sum += exp(-0.5 * (d / likelihood.bandwidth)^2)
    end
    return action_term + log(max(kernel_sum / likelihood.n_transition_samples, eps(Float64)))
end
