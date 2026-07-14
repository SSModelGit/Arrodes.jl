function _logsumexp(values)
    maximum_value = maximum(values)
    isfinite(maximum_value) || return maximum_value
    return maximum_value + log(sum(exp(value - maximum_value) for value in values))
end

function effective_sample_size(state::SMCFilterState)
    weights = exp.([particle.log_weight for particle in state.particles])
    weights ./= sum(weights)
    return inv(sum(abs2, weights))
end

function _sample_categorical(rng, probabilities)
    threshold = rand(rng) * sum(probabilities)
    cumulative = 0.0
    for (index, probability) in pairs(probabilities)
        cumulative += probability
        cumulative >= threshold && return index
    end
    return lastindex(probabilities)
end

function initialize_smc(config::SMCInferenceConfig; cache::PlanningCache = PlanningCache())
    _validate_config(config.model)
    config.n_particles > 0 || throw(ArgumentError("n_particles must be positive"))
    0 < config.ess_threshold <= 1 || throw(ArgumentError("ess_threshold must lie in (0, 1]"))
    config.rejuvenation_steps >= 0 || throw(ArgumentError("rejuvenation_steps must be nonnegative"))
    config.resampling === :systematic || throw(ArgumentError("only :systematic resampling is supported"))
    rng = MersenneTwister(config.model.seed)
    priors = Float64[h.prior_probability for h in config.model.hypotheses]
    priors ./= sum(priors)
    particles = [ParticleTrace(_sample_categorical(rng, priors), 0,
        -log(config.n_particles), 0.0, Float64[], Int[]) for _ in 1:config.n_particles]
    return SMCFilterState(config, cache, particles, 0, Any[], Any[], 0.0, rng, Int[], 0, 0)
end

function _systematic_resample!(state::SMCFilterState)
    particles = state.particles
    n = length(particles)
    weights = exp.([particle.log_weight for particle in particles])
    weights ./= sum(weights)
    cumulative = cumsum(weights)
    offset = rand(state.rng) / n
    ancestors = Vector{Int}(undef, n)
    cursor = 1
    for child in 1:n
        target = offset + (child - 1) / n
        while cursor < n && cumulative[cursor] < target
            cursor += 1
        end
        ancestors[child] = cursor
    end
    state.particles = [begin
        source = particles[ancestor]
        ParticleTrace(source.hypothesis_index, ancestor, -log(n), source.log_likelihood,
            copy(source.score_history), copy(source.hypothesis_history))
    end for ancestor in ancestors]
    push!(state.resampling_times, state.timestep)
    return ancestors
end

function _trajectory_score(state::SMCFilterState, hypothesis_index::Int, horizon::Int)
    hypothesis = state.config.model.hypotheses[hypothesis_index]
    score = 0.0
    increments = Float64[]
    for timestep in eachindex(state.actions)
        increment = _observation_loglikelihood(state.config.model, state.cache, hypothesis,
            state.states[1:timestep-1], state.actions[1:timestep-1], state.states[timestep],
            state.actions[timestep], timestep, horizon, state.config.model.seed)
        push!(increments, increment)
        score += increment
    end
    return score, increments
end

function _proposal(state::SMCFilterState, current::Int)
    proposal = state.config.proposal
    priors = Float64[h.prior_probability for h in state.config.model.hypotheses]
    priors ./= sum(priors)
    if isnothing(proposal)
        candidate = _sample_categorical(state.rng, priors)
        return (index = candidate, log_forward = log(priors[candidate]),
            log_reverse = log(priors[current]))
    end
    value = proposal(state.rng, current, state)
    value isa Integer && return (index = Int(value), log_forward = 0.0, log_reverse = 0.0)
    all(key -> hasproperty(value, key), (:index, :log_forward, :log_reverse)) ||
        throw(ArgumentError("proposal must return an index or (index, log_forward, log_reverse)"))
    return value
end

function _rejuvenate!(state::SMCFilterState, horizon::Int)
    hypotheses = state.config.model.hypotheses
    # A static-objective trajectory score is shared by every particle proposing
    # that objective. Memoizing it is essential when a score invokes an expensive
    # planner such as InfoMCTS; it also makes rejuvenation O(H), not O(NH).
    trajectory_scores = Dict{Int,Tuple{Float64,Vector{Float64}}}()
    for particle in state.particles
        trajectory_scores[particle.hypothesis_index] =
            (particle.log_likelihood, copy(particle.score_history))
    end
    for particle in state.particles, _ in 1:state.config.rejuvenation_steps
        move = _proposal(state, particle.hypothesis_index)
        1 <= move.index <= length(hypotheses) || throw(BoundsError(hypotheses, move.index))
        state.rejuvenation_attempts += 1
        move.index == particle.hypothesis_index && continue
        candidate_score, candidate_increments = get!(trajectory_scores, move.index) do
            _trajectory_score(state, move.index, horizon)
        end
        current_prior = log(hypotheses[particle.hypothesis_index].prior_probability)
        candidate_prior = log(hypotheses[move.index].prior_probability)
        log_acceptance = candidate_score - particle.log_likelihood + candidate_prior - current_prior +
            move.log_reverse - move.log_forward
        if log(rand(state.rng)) < min(0.0, log_acceptance)
            particle.hypothesis_index = move.index
            particle.log_likelihood = candidate_score
            particle.score_history = copy(candidate_increments)
            particle.hypothesis_history .= move.index
            state.rejuvenation_accepts += 1
        end
    end
    return state
end

"""Condition every particle trace on one additional observed state/action pair."""
function update!(state::SMCFilterState, state_observation, observed_action;
                 horizon::Int = state.timestep + 1)
    timestep = state.timestep + 1
    model = state.config.model
    increments = Dict{Int,Float64}()
    for index in unique(particle.hypothesis_index for particle in state.particles)
        hypothesis = model.hypotheses[index]
        increments[index] = _observation_loglikelihood(model, state.cache, hypothesis,
            state.states, state.actions, state_observation, observed_action, timestep, horizon,
            model.seed)
    end
    unnormalized = Float64[]
    for particle in state.particles
        increment = increments[particle.hypothesis_index]
        particle.log_weight += increment
        particle.log_likelihood += increment
        push!(particle.score_history, increment)
        push!(particle.hypothesis_history, particle.hypothesis_index)
        push!(unnormalized, particle.log_weight)
    end
    log_increment = _logsumexp(unnormalized)
    isfinite(log_increment) || throw(ArgumentError("all particles assigned zero probability to observation"))
    for particle in state.particles
        particle.log_weight -= log_increment
    end
    state.log_evidence += log_increment
    state.timestep = timestep
    push!(state.states, state_observation)
    push!(state.actions, observed_action)
    ancestors = collect(eachindex(state.particles))
    if effective_sample_size(state) < state.config.ess_threshold * length(state.particles)
        ancestors = _systematic_resample!(state)
        _rejuvenate!(state, horizon)
    end
    return ancestors
end

"""
Run a trace-preserving resample-move SMC filter over named objective hypotheses.

The returned ancestry and per-particle score histories retain the successive
conditioning structure needed for diagnostics and backward trace inspection.
"""
function infer_objectives_smc(config::SMCInferenceConfig, state_observations, observed_actions;
                              cache::PlanningCache = PlanningCache())
    observations = _observations_by_time(state_observations)
    actions = collect(observed_actions)
    length(observations) == length(actions) || throw(DimensionMismatch(
        "received $(length(observations)) states and $(length(actions)) actions"))
    isempty(actions) && throw(ArgumentError("at least one observation is required"))
    state = initialize_smc(config; cache = cache)
    n_hypotheses = length(config.model.hypotheses)
    history = Matrix{Float64}(undef, n_hypotheses, length(actions))
    evidence = Vector{Float64}(undef, length(actions))
    ess = Vector{Float64}(undef, length(actions))
    ancestry = Vector{Vector{Int}}(undef, length(actions))
    for timestep in eachindex(actions)
        ancestry[timestep] = update!(state, observations[timestep], actions[timestep];
            horizon = length(actions))
        history[:, timestep] = posterior(state)
        evidence[timestep] = state.log_evidence
        ess[timestep] = effective_sample_size(state)
    end
    return SMCFilterResult(state, history, evidence, ess, ancestry)
end
