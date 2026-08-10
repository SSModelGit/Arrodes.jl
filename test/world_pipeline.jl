function tiny_scribe_context()
    locations = [0.0 0.0; 1.0 0.0; 0.0 1.0; 1.0 1.0]
    snapshots = [
        0.0 1.0 0.2 1.1 0.1
        0.2 1.2 0.1 1.0 0.3
        1.0 0.0 1.1 0.1 0.9
        1.2 0.2 1.0 0.0 1.1
    ]
    covariance = 0.5 .* Matrix{Float64}(I, 2, 2)
    model = initialize_eof_climate_model(
        snapshots;
        locations=locations,
        rank=2,
        process_covariance=0.02 .* covariance,
        prior_covariance=covariance,
    )
    information = SCRIBE.init_agent_info(model.params; prior_covariance=covariance)
    scribe_world_context(model, information)
end

function tiny_ergodic_evidence(; β_max=0.5)
    DirectErgodicEvidence(
        location=state -> state,
        reward=(model, state, action, context) ->
            only(SCRIBE.predict_SCRIBEModel(model, reshape(state, 1, :))),
        importance=(model, X, values) -> exp.(values),
        kernel=GaussianDiscrepancyKernel(bandwidth=0.8),
        energy=WorldEnergyConfig(
            discrepancy_scale=1.0,
            reward_scale=1.0,
            maturity_half_time=2.0,
            β_max=β_max,
        ),
    )
end

@testset "SCRIBE world-inference pipeline" begin
    context = tiny_scribe_context()
    candidate = [0.3, -0.2]
    @test candidate_field(context, candidate) ≈
        SCRIBE.reconstruct_eof_field(candidate_model(context, candidate))
    problem = WorldInferenceProblem(context=context, evidence=tiny_ergodic_evidence())
    observations = [
        TrajectoryObservation(state=[0.0, 0.0], action=:move),
        TrajectoryObservation(state=[1.0, 0.0], action=:move),
        TrajectoryObservation(state=[1.0, 1.0], action=:move),
    ]
    result = infer_world(
        problem,
        observations,
        SMCConfig(
            n_particles=24,
            scheduler=OneStagePerObservation(),
            kernel=IdentityKernel(),
            seed=UInt64(7),
        ),
    )
    summary = world_posterior(result)
    @test length(summary.coefficient_mean) == context.model.params.nᵩ
    @test length(summary.map_mean) == length(SCRIBE.eof_mean(context.model))
    @test all(isfinite, summary.map_variance)
    @test summary.identifiability === :mean_dependent
    @test all(diagnostic.stage.environment_time == context.model_time
              for diagnostic in result.state.diagnostics)
    @test length(result.state.cache[:world_energy_cache]) <= 2
    @test length(result.state.cache[:world_trajectory_kernels]) <= 2
    @test all(particle.trace isa WorldTrace for particle in posterior_particles(result))

    deployment = deploy_behavior_information(problem, summary)
    @test deployment.reason in (:exact_gaussian, :indefinite_information)
    projected = deploy_behavior_information(problem, summary; project=true)
    @test projected.accepted
    @test all(iszero, projected.information.i)
    @test all(iszero, projected.information.I)
end

@testset "Dynamic SCRIBE time is physical time" begin
    context = tiny_scribe_context()
    planner = fixed_planner(:right)
    evidence = PlannerWorldEvidence(
        objective=:right,
        mdp_builder=(model, objective) -> TinyMDP(objective),
        behavior=BehaviorModel(planner, EpsilonGreedyLikelihood(epsilon=0.1)),
    )
    problem = DynamicWorldInferenceProblem(context=context, evidence=evidence)
    observations = [
        TrajectoryObservation(state=0, action=:right, environment_time=context.model_time),
        TrajectoryObservation(state=1, action=:right, environment_time=context.model_time + 2),
    ]
    result = infer_dynamic_world(
        problem, observations,
        SMCConfig(n_particles=12, kernel=DynamicWorldKernel(), seed=UInt64(9)),
    )
    @test length(first(posterior_particles(result)).value.coefficients) == 3
    @test posterior(result).environment_times == collect(context.model_time:context.model_time + 2)
end

@testset "Paired world transport" begin
    problem = WorldInferenceProblem(
        context=tiny_scribe_context(),
        evidence=tiny_ergodic_evidence(β_max=0.2),
    )
    result = infer_world(
        problem,
        [TrajectoryObservation(state=[0.0, 0.0], action=:move)],
        SMCConfig(
            n_particles=8,
            paired_moves_per_stage=2,
            kernel=WorldKernelMixture(),
            ess_threshold=0.2,
            seed=UInt64(10),
        ),
    )
    @test length(result.state.ancestry) == 2
    @test all(isfinite, getfield.(posterior_particles(result), :log_weight))
    @test all(branch -> branch in (:local, :amortized, :refresh),
              reduce(vcat, getfield.(result.state.ancestry, :branches)))
    @test all(move -> 0 < move.transport_fraction <= 1,
              first(posterior_particles(result)).trace.moves)

    metric = last(first(posterior_particles(result)).trace.moves)
    @test isfinite(metric.log_forward)
    @test isfinite(metric.log_backward)
    prepared = Arrodes.Inference.prepare_kernel(
        WorldKernelMixture(),
        problem,
        result.state.cloud.stage,
        result.state.cloud.stage,
        result.state.cloud,
        result.state.cache,
    )
    local_branch = only(filter(
        branch -> branch.name === :local, prepared.branches,
    ))
    @test minimum(eigvals(Symmetric(local_branch.information))) > 0
    @test 0 < local_branch.transport_fraction <= 1

    pcn_problem = WorldInferenceProblem(
        context=tiny_scribe_context(),
        evidence=tiny_ergodic_evidence(β_max=0.2),
    )
    pcn_result = infer_world(
        pcn_problem,
        [TrajectoryObservation(state=[0.0, 0.0], action=:move)],
        SMCConfig(
            n_particles=8,
            kernel=PriorPCNKernel(ρ=0.9),
            ess_threshold=0.2,
            seed=UInt64(11),
        ),
    )
    @test all(==(:prior_pcn), last(pcn_result.state.ancestry).branches)
    @test all(isfinite, getfield.(posterior_particles(pcn_result), :log_weight))

    ratio_problem = WorldInferenceProblem(
        context=tiny_scribe_context(),
        evidence=tiny_ergodic_evidence(β_max=0.2),
    )
    observe!(ratio_problem, TrajectoryObservation(state=[0.0, 0.0], action=:move))
    old_stage = InferenceStage()
    new_stage = InferenceStage(observation=1, bridge=1, λ=1.0)
    particle = WeightedParticle(
        value=[0.2, -0.1], trace=WorldTrace(), log_weight=0.0, lineage=1,
    )
    kernel = PriorPCNKernel(ρ=0.9)
    cache = Dict{Symbol,Any}()
    move = Arrodes.Inference.propose(
        kernel, ratio_problem, old_stage, new_stage, particle,
        MersenneTwister(12), cache,
    )
    covariance = (1 - kernel.ρ^2) .* ratio_problem.context.prior_covariance
    backward_mean = ratio_problem.context.prior_mean .+
        kernel.ρ .* (move.value .- ratio_problem.context.prior_mean)
    explicit_ratio =
        logtarget(ratio_problem, new_stage, move.value, cache) -
        logtarget(ratio_problem, old_stage, particle.value, cache) +
        Arrodes.Inference.gaussian_logdensity(
            particle.value, backward_mean, covariance,
        ) -
        Arrodes.Inference.gaussian_logdensity(
            move.value, move.metadata.forward_mean, covariance,
        )
    @test Arrodes.Inference.paired_logweight(
        ratio_problem, old_stage, new_stage, particle, move, cache,
    ) ≈ explicit_ratio
end

@testset "World evidence evaluation branches" begin
    context = tiny_scribe_context()
    observation = TrajectoryObservation(state=[1.0, 0.0], action=:move)
    energies = Dict{Symbol,WorldEnergy}()
    for evaluation in (:combined, :mmd, :reward)
        evidence = tiny_ergodic_evidence()
        config = evidence.energy
        configured = DirectErgodicEvidence(
            location=evidence.location,
            reward=evidence.reward,
            importance=evidence.importance,
            kernel=evidence.kernel,
            energy=WorldEnergyConfig(
                discrepancy_scale=config.discrepancy_scale,
                reward_scale=config.reward_scale,
                β_max=config.β_max,
                maturity_half_time=config.maturity_half_time,
                evaluation=evaluation,
            ),
        )
        problem = WorldInferenceProblem(context=context, evidence=configured)
        observe!(problem, observation)
        energies[evaluation] = world_energy(
            problem, configured, 1, [0.2, -0.1], Dict{Symbol,Any}(),
        )
    end
    @test energies[:mmd].reward_weight == 0
    @test energies[:mmd].discrepancy_weight == 1
    @test energies[:reward].reward_weight == 1
    @test energies[:reward].discrepancy_weight == 0
    @test energies[:combined].reward_weight +
        energies[:combined].discrepancy_weight ≈ 1
end

@testset "World evidence ranks a known EOF field" begin
    context = tiny_scribe_context()
    evidence = DirectErgodicEvidence(
        location=state -> state,
        reward=(model, state, action, context) -> 0.0,
        importance=(model, X, values) -> exp.(values .- maximum(values)),
        kernel=GaussianDiscrepancyKernel(bandwidth=0.8),
        energy=WorldEnergyConfig(
            discrepancy_scale=1.0,
            reward_scale=1.0,
            maturity_half_time=1.0,
            β_max=4.0,
        ),
    )
    truth = [0.8, -0.5]
    problem = WorldInferenceProblem(context=context, evidence=evidence)
    masses = Arrodes.Inference.target_measure(problem, evidence, truth)
    counts = floor.(Int, 200 .* masses)
    for index in sortperm(200 .* masses .- counts; rev=true)[1:200 - sum(counts)]
        counts[index] += 1
    end
    for index in eachindex(counts), _ in 1:counts[index]
        observe!(problem, TrajectoryObservation(state=vec(context.quadrature[index, :])))
    end
    candidates = [
        :truth => truth,
        :opposite => -truth,
        :prior => zeros(2),
    ]
    ranking = rank_candidate_worlds(problem, candidates)
    @test first(ranking).id === :truth
    @test first(ranking).discrepancy < last(ranking).discrepancy

    dwell_evidence = DirectErgodicEvidence(
        location=state -> state,
        reward=(model, state, action, context) -> state[1],
        importance=(model, X, values) -> ones(size(X, 1)),
        kernel=GaussianDiscrepancyKernel(bandwidth=1.0),
        energy=WorldEnergyConfig(discrepancy_scale=1.0, reward_scale=1.0),
    )
    dwell_problem = WorldInferenceProblem(context=context, evidence=dwell_evidence)
    observe!(dwell_problem, TrajectoryObservation(state=[0.0, 0.0], dwell_time=1.0))
    observe!(dwell_problem, TrajectoryObservation(state=[1.0, 0.0], dwell_time=3.0))
    @test trajectory_mean_reward(dwell_problem, dwell_evidence, 2, zeros(2)) ≈ 0.75
end

@testset "Behavior deployment uses its declared prior center" begin
    base = tiny_scribe_context()
    Y = copy(base.information.Y)
    shifted_mean = [0.4, -0.3]
    shifted_information = SCRIBE.KFEnvInfo(
        Y * shifted_mean,
        Y,
        zeros(2),
        zeros(2, 2),
    )
    context = scribe_world_context(base.model, shifted_information)
    problem = WorldInferenceProblem(context=context, evidence=tiny_ergodic_evidence())
    map_mean = candidate_field(context, context.prior_mean)
    summary = WorldPosteriorSummary(
        stage=InferenceStage(),
        coefficient_mean=copy(context.prior_mean),
        coefficient_covariance=copy(context.prior_covariance),
        map_mean=map_mean,
        map_variance=zeros(length(map_mean)),
        mean_energy=0.0,
        posterior_prior_kl=0.0,
        contraction=0.0,
        identifiability=:mean_dependent,
    )
    deployment = deploy_behavior_information(problem, summary)
    @test deployment.accepted
    @test deployment.ΔY ≈ zeros(2, 2)
    @test deployment.Δy ≈ zeros(2)

    shifted_summary = WorldPosteriorSummary(
        stage=InferenceStage(observation=20),
        coefficient_mean=shifted_mean .+ [4.0, -3.0],
        coefficient_covariance=copy(context.prior_covariance),
        map_mean=map_mean,
        map_variance=zeros(length(map_mean)),
        mean_energy=0.0,
        posterior_prior_kl=0.0,
        contraction=0.0,
        identifiability=:mean_dependent,
    )
    blended = blended_coefficients(
        problem, shifted_summary; confidence=0.8, trust_radius=0.5,
    )
    inferred_gap = shifted_summary.coefficient_mean - context.prior_mean
    deployed_gap = blended - context.prior_mean
    @test deployed_gap[1] / inferred_gap[1] ≈ deployed_gap[2] / inferred_gap[2]
    @test norm(cholesky(Symmetric(context.prior_covariance)).L \ deployed_gap) <= 0.5
end
