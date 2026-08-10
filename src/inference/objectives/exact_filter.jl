export infer_objectives_exact

function objective_logpriors(problem::ObjectiveInferenceProblem)
    values = log.([hypothesis.prior_probability for hypothesis in problem.hypotheses])
    values .- logsumexp(values)
end

function infer_objectives_exact(problem::ObjectiveInferenceProblem, observations)
    validate_problem(problem)
    n_hypotheses = length(problem.hypotheses)
    n_observations = length(observations)
    n_observations > 0 || throw(ArgumentError("at least one behavior observation is required"))
    log_weights = objective_logpriors(problem)
    posterior_history = Matrix{Float64}(undef, n_hypotheses, n_observations)
    score_history = Matrix{Float64}(undef, n_hypotheses, n_observations)
    normalizers = Vector{Float64}(undef, n_observations)
    log_normalizer = 0.0
    cache = Dict{Symbol,Any}()

    for (timestep, observation) in enumerate(observations)
        observe!(problem, observation)
        increments = [objective_loglikelihood(problem, index, timestep, cache)
                      for index in 1:n_hypotheses]
        score_history[:, timestep] = increments
        log_weights .+= increments
        increment = logsumexp(log_weights)
        log_weights .-= increment
        log_normalizer += increment
        posterior_history[:, timestep] = exp.(log_weights)
        normalizers[timestep] = log_normalizer
    end
    ExactObjectiveResult(
        problem=problem,
        posterior_history=posterior_history,
        log_normalizer_history=normalizers,
        score_history=score_history,
    )
end

function infer_objectives_exact(problem::ObjectiveInferenceProblem, states, actions)
    horizon = length(actions)
    problem.horizon = horizon
    observations = [BehaviorObservation(state=states[t], action=actions[t], horizon=horizon)
                    for t in eachindex(actions)]
    infer_objectives_exact(problem, observations)
end

infer_objectives(args...; kwargs...) = infer_objectives_exact(args...; kwargs...)
