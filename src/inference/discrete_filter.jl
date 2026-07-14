function _normalize_logweights!(log_weights::Vector{Float64})
    maximum_weight = maximum(log_weights)
    isfinite(maximum_weight) || throw(ArgumentError(
        "all hypotheses assigned zero probability to the observation"))
    log_normalizer = maximum_weight + log(sum(exp.(log_weights .- maximum_weight)))
    log_weights .-= log_normalizer
    return log_normalizer
end

function initialize_filter(config::DiscreteInferenceConfig; cache::PlanningCache = PlanningCache())
    _validate_config(config)
    priors = Float64[h.prior_probability for h in config.hypotheses]
    priors ./= sum(priors)
    return DiscreteFilterState(config, cache, log.(priors), 0, Any[], Any[], 0.0)
end

function hypothesis_mdp(state::DiscreteFilterState, hypothesis::ObjectiveHypothesis)
    return get!(state.cache.mdps, hypothesis.id) do
        state.config.mdp_builder(hypothesis.objective, hypothesis)
    end
end

hypothesis_mdp(state::DiscreteFilterState, id::Symbol) =
    hypothesis_mdp(state, state.config.hypotheses[hypothesis_index(state.config, id)])

function hypothesis_mdp(state::SMCFilterState, hypothesis::ObjectiveHypothesis)
    return get!(state.cache.mdps, hypothesis.id) do
        state.config.model.mdp_builder(hypothesis.objective, hypothesis)
    end
end

hypothesis_mdp(state::SMCFilterState, id::Symbol) = hypothesis_mdp(state,
    state.config.model.hypotheses[hypothesis_index(state.config.model, id)])

function _context(state::DiscreteFilterState, hypothesis::ObjectiveHypothesis, states, actions,
                  timestep::Int, horizon::Int)
    seed = hash((state.config.seed, hypothesis.id, timestep), UInt(0))
    return PlanningContext(
        hypothesis_id = hypothesis.id,
        timestep = timestep,
        states = states,
        actions = actions,
        horizon = horizon,
        rng = MersenneTwister(seed),
        metadata = hypothesis.metadata,
    )
end

function _resolve_action(observed_action, action_list)
    if observed_action isa Integer
        index = Int(observed_action)
        1 <= index <= length(action_list) || throw(BoundsError(action_list, index))
        return (index = index, action = action_list[index])
    end
    index = findfirst(isequal(observed_action), action_list)
    index === nothing && return nothing
    return (index = index, action = action_list[index])
end

function _observation_loglikelihood(config, cache, hypothesis, state_observations, prior_actions,
                                    state_observation, observed_action, timestep, horizon, seed)
    mdp = get!(cache.mdps, hypothesis.id) do
        config.mdp_builder(hypothesis.objective, hypothesis)
    end
    planner_states = Any[config.state_adapter(mdp, observation, t) for
        (t, observation) in enumerate(state_observations)]
    planner_state = config.state_adapter(mdp, state_observation, timestep)
    push!(planner_states, planner_state)
    action_list = collect(POMDPs.actions(mdp))
    planner_actions = Any[]
    for action in prior_actions
        resolved = _resolve_action(action, action_list)
        resolved === nothing && return -Inf
        push!(planner_actions, resolved.action)
    end
    context = PlanningContext(
        hypothesis_id = hypothesis.id,
        timestep = timestep,
        states = planner_states,
        actions = planner_actions,
        horizon = horizon,
        rng = MersenneTwister(hash((seed, hypothesis.id, timestep), UInt(0))),
        metadata = hypothesis.metadata,
    )
    artifact = prepare_cached!(cache, hypothesis.behavior.planner, mdp, context)
    # Policies such as VulcanJ InfoMCTS update internal trees and mission state on
    # query. Keep the cached artifact as a reproducible template so SMC replay and
    # rejuvenation do not depend on which particle happened to query it first.
    query_artifact = deepcopy(artifact)
    resolved = _resolve_action(observed_action, action_list)
    resolved === nothing && return -Inf
    return observation_loglikelihood(hypothesis.behavior.likelihood,
        hypothesis.behavior.planner, query_artifact, mdp, planner_state, resolved.action,
        action_list, context)
end

function hypothesis_artifact(state::DiscreteFilterState, id::Symbol; horizon::Int = max(state.timestep, 1))
    hypothesis = state.config.hypotheses[hypothesis_index(state.config, id)]
    mdp = hypothesis_mdp(state, hypothesis)
    planner_states = Any[
        state.config.state_adapter(mdp, observation, t) for
        (t, observation) in enumerate(state.states)
    ]
    action_list = collect(POMDPs.actions(mdp))
    planner_actions = Any[_resolve_action(action, action_list).action for action in state.actions]
    context = _context(state, hypothesis, planner_states, planner_actions, max(state.timestep, 1), horizon)
    return prepare_cached!(state.cache, hypothesis.behavior.planner, mdp, context)
end

"""Advance an exact finite-hypothesis filter by one observed state/action pair."""
function update!(state::DiscreteFilterState, state_observation, observed_action; horizon::Int = state.timestep + 1)
    timestep = state.timestep + 1
    likelihoods = Vector{Float64}(undef, length(state.config.hypotheses))

    for (i, hypothesis) in enumerate(state.config.hypotheses)
        likelihoods[i] = exp(_observation_loglikelihood(state.config, state.cache, hypothesis,
            state.states, state.actions, state_observation, observed_action, timestep, horizon,
            state.config.seed))
    end

    state.log_weights .+= log.(likelihoods)
    log_increment = _normalize_logweights!(state.log_weights)
    state.log_evidence += log_increment
    state.timestep = timestep
    push!(state.states, state_observation)
    push!(state.actions, observed_action)
    return state
end

function _observations_by_time(state_observations::AbstractMatrix)
    return [view(state_observations, :, t) for t in axes(state_observations, 2)]
end

_observations_by_time(state_observations) = collect(state_observations)

"""
    infer_objectives(config, state_observations, observed_actions; cache=PlanningCache())

Evaluate every named hypothesis at every timestep and return its exact posterior history.
Integer actions are interpreted as one-based indices into `POMDPs.actions(mdp)`.
"""
function infer_objectives(config::DiscreteInferenceConfig, state_observations, observed_actions;
                          cache::PlanningCache = PlanningCache())
    observations = _observations_by_time(state_observations)
    actions = collect(observed_actions)
    length(observations) == length(actions) || throw(DimensionMismatch(
        "received $(length(observations)) states and $(length(actions)) actions"))
    isempty(actions) && throw(ArgumentError("at least one observation is required"))

    state = initialize_filter(config; cache = cache)
    history = Matrix{Float64}(undef, length(config.hypotheses), length(actions))
    evidence_history = Vector{Float64}(undef, length(actions))
    for t in eachindex(actions)
        update!(state, observations[t], actions[t]; horizon = length(actions))
        history[:, t] = posterior(state)
        evidence_history[t] = state.log_evidence
    end
    return DiscreteFilterResult(state, history, evidence_history)
end
