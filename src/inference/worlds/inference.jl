export infer_world, world_posterior, world_energy_history

function gaussian_kl(μ, Σ, prior_mean, prior_covariance)
    dimension = length(μ)
    prior_precision = inv(Symmetric(prior_covariance))
    difference = prior_mean - μ
    0.5 * (
        tr(prior_precision * Σ) +
        dot(difference, prior_precision * difference) -
        dimension +
        logdet(Symmetric(prior_covariance)) - logdet(Symmetric(Σ))
    )
end

function summarize(problem::WorldInferenceProblem, cloud::ParticleCloud,
                   stage::InferenceStage, cache)
    values = [particle.value for particle in cloud.particles]
    log_weights = [particle.log_weight for particle in cloud.particles]
    moments = weighted_mean_covariance(values, log_weights)
    covariance_scale = max(opnorm(problem.context.prior_covariance), 1.0)
    posterior_covariance = regularize_covariance(
        moments.covariance, sqrt(eps(Float64)) * covariance_scale,
    )
    basis = SCRIBE.eof_modes(problem.context.model)
    map_mean, map_variance = if stage.λ == 1.0
        (
            SCRIBE.reconstruct_eof_field(
                problem.context.model; coefficients=moments.mean,
            ),
            vec(sum((basis * posterior_covariance) .* basis; dims=2)) +
                SCRIBE.eof_residual_variance(problem.context.model),
        )
    else
        (Float64[], Float64[])
    end
    energies = if problem.evidence isa DirectErgodicEvidence && stage.observation > 0
        [cached_world_energy(problem, problem.evidence, stage.observation, value, cache).total
         for value in values]
    else
        zeros(length(values))
    end
    prior_covariance = problem.context.prior_covariance
    contraction = 0.5 * (
        logdet(Symmetric(prior_covariance)) -
        logdet(Symmetric(posterior_covariance))
    )
    WorldPosteriorSummary(
        stage=stage,
        coefficient_mean=moments.mean,
        coefficient_covariance=posterior_covariance,
        map_mean=map_mean,
        map_variance=map_variance,
        mean_energy=dot(moments.weights, energies),
        posterior_prior_kl=gaussian_kl(
            moments.mean, posterior_covariance,
            problem.context.prior_mean, prior_covariance,
        ),
        contraction=contraction,
        identifiability=evidence_identifiability(problem.evidence),
    )
end

posterior_coefficient_moments(summary::WorldPosteriorSummary) =
    (mean=summary.coefficient_mean, covariance=summary.coefficient_covariance)

function posterior_coefficient_moments(result::SMCResult{<:SequentialState{<:WorldInferenceProblem}})
    posterior_coefficient_moments(last(result.state.summaries))
end

world_posterior(result::SMCResult{<:SequentialState{<:WorldInferenceProblem}}) =
    last(result.state.summaries)
posterior(result::SMCResult{<:SequentialState{<:WorldInferenceProblem}}) =
    world_posterior(result)

world_energy_history(result::SMCResult{<:SequentialState{<:WorldInferenceProblem}}) =
    [summary.mean_energy for summary in result.state.summaries]

function infer_world(problem::WorldInferenceProblem, observations, config::SMCConfig)
    problem.horizon = length(observations)
    run_smc(problem, observations, config)
end
