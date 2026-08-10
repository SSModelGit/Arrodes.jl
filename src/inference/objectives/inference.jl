export infer_objectives_smc

function observe!(problem::ObjectiveInferenceProblem, observation::BehaviorObservation)
    push!(problem.states, observation.state)
    push!(problem.actions, observation.action)
    problem.horizon = observation.horizon
    problem
end

function logtarget(problem::ObjectiveInferenceProblem, stage::InferenceStage, index::Int, cache)
    objective_logpriors(problem)[index] +
        sum(objective_loglikelihood(problem, index, timestep, cache)
            for timestep in 1:stage.observation; init=0.0)
end

function initial_proposal(problem::ObjectiveInferenceProblem, rng, cache)
    log_priors = objective_logpriors(problem)
    probabilities = exp.(log_priors)
    index = searchsortedfirst(cumsum(probabilities), rand(rng))
    (value=index, trace=ObjectiveTrace(), logdensity=log_priors[index])
end

function summarize(problem::ObjectiveInferenceProblem, cloud::ParticleCloud,
                   stage::InferenceStage, cache)
    probabilities = zeros(length(problem.hypotheses))
    for particle in cloud.particles
        probabilities[particle.value] += exp(particle.log_weight)
    end
    ObjectivePosteriorSummary(
        stage=stage,
        ids=[hypothesis.id for hypothesis in problem.hypotheses],
        probabilities=probabilities,
        log_probabilities=log.(probabilities),
        best_index=argmax(probabilities),
    )
end

function infer_objectives_smc(problem::ObjectiveInferenceProblem, observations, config::SMCConfig)
    run_smc(problem, observations, config)
end

function infer_objectives_smc(problem::ObjectiveInferenceProblem, states, actions,
                              config::SMCConfig)
    horizon = length(actions)
    problem.horizon = horizon
    observations = [BehaviorObservation(state=states[t], action=actions[t], horizon=horizon)
                    for t in eachindex(actions)]
    run_smc(problem, observations, config)
end
