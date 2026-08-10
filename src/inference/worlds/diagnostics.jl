export rank_candidate_worlds, world_recovery_diagnostics

function rank_candidate_worlds(
    problem::WorldInferenceProblem,
    candidates::AbstractVector{<:Pair};
    timestep=length(problem.trajectory.states),
)
    evidence = problem.evidence
    evidence isa DirectErgodicEvidence ||
        throw(ArgumentError("candidate energy ranking requires direct ergodic evidence"))
    cache = Dict{Symbol,Any}()
    scores = map(candidates) do candidate
        energy = world_energy(
            problem, evidence, timestep, last(candidate), cache,
        )
        CandidateWorldScore(
            id=Symbol(first(candidate)),
            energy=energy.total,
            discrepancy=energy.discrepancy,
            mean_reward=energy.mean_reward,
        )
    end
    sort(scores; by=score -> score.energy)
end

function target_measure_discrepancy(problem, evidence, left, right, cache)
    left_mass = target_measure(problem, evidence, left)
    right_mass = target_measure(problem, evidence, right)
    kernel = get!(cache, :world_target_kernel) do
        kernel_matrix(
            evidence.kernel,
            problem.context.quadrature,
            problem.context.quadrature,
        )
    end
    difference = left_mass - right_mass
    max(dot(difference, kernel * difference), 0.0)
end

function world_recovery_diagnostics(
    problem::WorldInferenceProblem,
    summary::WorldPosteriorSummary,
    truth::AbstractVector;
    ess=NaN,
    marginal_radius=1.96,
    cache=Dict{Symbol,Any}(),
)
    evidence = problem.evidence
    evidence isa DirectErgodicEvidence ||
        throw(ArgumentError("world recovery diagnostics require direct ergodic evidence"))
    difference = summary.coefficient_mean - truth
    covariance = regularize_covariance(summary.coefficient_covariance, 1e-10)
    posterior_mahalanobis = sqrt(max(dot(difference, covariance \ difference), 0.0))
    truth_map = candidate_field(problem.context, truth)
    map_difference = summary.map_mean - truth_map
    map_correlation = std(summary.map_mean) > 0 && std(truth_map) > 0 ?
        cor(summary.map_mean, truth_map) : NaN
    standard_deviation = sqrt.(max.(diag(covariance), 0.0))
    covered = abs.(difference) .<= marginal_radius .* standard_deviation
    WorldRecoveryDiagnostics(
        timestep=summary.stage.observation,
        coefficient_rmse=sqrt(mean(abs2, difference)),
        posterior_mahalanobis=posterior_mahalanobis,
        map_rmse=sqrt(mean(abs2, map_difference)),
        map_correlation=map_correlation,
        target_discrepancy=target_measure_discrepancy(
            problem, evidence, summary.coefficient_mean, truth, cache,
        ),
        marginal_coverage=mean(covered),
        ess=Float64(ess),
        contraction=summary.contraction,
    )
end

function world_recovery_diagnostics(
    result::SMCResult{<:SequentialState{<:WorldInferenceProblem}},
    truth::AbstractVector;
    marginal_radius=1.96,
)
    world_recovery_diagnostics(
        result.state.problem,
        world_posterior(result),
        truth;
        ess=effective_sample_size(result),
        marginal_radius,
        cache=result.state.cache,
    )
end
