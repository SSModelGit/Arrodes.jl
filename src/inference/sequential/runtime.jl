export initialize_smc, update_smc!, run_smc, posterior_particles

particle_rng(seed, stage, lineage, move) = MersenneTwister(
    hash((seed, stage.observation, stage.bridge, stage.environment_time, lineage, move), UInt(0)),
)

function _copy_particle(source, log_weight, lineage, parent, branch)
    WeightedParticle(
        value=deepcopy(source.value),
        trace=deepcopy(source.trace),
        log_weight=log_weight,
        lineage=lineage,
        branch=branch,
    )
end

function initialize_smc(problem::AbstractBehaviorInferenceProblem, config::SMCConfig)
    validate_problem(problem)
    config.n_particles > 0 || throw(ArgumentError("particle count must be positive"))
    0 < config.ess_threshold <= 1 ||
        throw(ArgumentError("ESS threshold must lie in (0, 1]"))
    config.paired_moves_per_stage > 0 ||
        throw(ArgumentError("each stage needs at least one paired move"))
    config.invariant_steps >= 0 ||
        throw(ArgumentError("invariant move count cannot be negative"))
    cache = Dict{Symbol,Any}()
    stage = initial_stage(problem)
    particles = map(1:config.n_particles) do lineage
        rng = particle_rng(config.seed, stage, lineage, 0)
        proposal = initial_proposal(problem, rng, cache)
        WeightedParticle(
            value=proposal.value,
            trace=proposal.trace,
            log_weight=logtarget(problem, stage, proposal.value, cache) - proposal.logdensity,
            lineage=lineage,
        )
    end
    log_normalizer = normalize_logweights!(particles)
    cloud = ParticleCloud(
        particles=particles,
        stage=stage,
        log_normalizer=log_normalizer - log(config.n_particles),
    )
    state = SequentialState(
        problem=problem,
        cloud=cloud,
        rng=MersenneTwister(config.seed),
        cache=cache,
    )
    push!(state.summaries, summarize(problem, cloud, stage, cache))
    state
end

function _resample!(state, config, stage)
    particles = state.cloud.particles
    weights = exp.([particle.log_weight for particle in particles])
    parents = @match config.resampling begin
        :systematic => systematic_resample(state.rng, weights)
        :residual => residual_resample(state.rng, weights)
        _ => error("unknown resampling method: $(config.resampling)")
    end
    n = length(particles)
    children = map(enumerate(parents)) do (child, parent)
        source = particles[parent]
        lineage = state.cloud.next_lineage
        state.cloud.next_lineage += 1
        _copy_particle(source, -log(n), lineage, parent, source.branch)
    end
    state.cloud.particles = children
    parents
end

function _invariant_moves!(state, config, stage)
    attempts = 0
    accepts = 0
    parents = collect(eachindex(state.cloud.particles))
    for index in eachindex(state.cloud.particles), move_index in 1:config.invariant_steps
        particle = state.cloud.particles[index]
        rng = particle_rng(config.seed, stage, particle.lineage, 10_000 + move_index)
        candidate, accepted = step(
            config.invariant_move, state.problem, stage, particle, rng, state.cache,
        )
        state.cloud.particles[index] = candidate
        attempts += 1
        accepts += accepted
    end
    parents, attempts == 0 ? 0.0 : accepts / attempts
end

function _paired_move!(state, config, old_stage, new_stage, move_index)
    cloud = state.cloud
    kernel = prepare_kernel(
        config.kernel, state.problem, old_stage, new_stage, cloud, state.cache,
    )
    increments = Vector{Float64}(undef, length(cloud.particles))
    branches = Vector{Symbol}(undef, length(cloud.particles))
    proposed = map(enumerate(cloud.particles)) do (index, particle)
        rng = particle_rng(config.seed, new_stage, particle.lineage, move_index)
        move = propose(kernel, state.problem, old_stage, new_stage, particle, rng, state.cache)
        increments[index] = paired_logweight(
            state.problem, old_stage, new_stage, particle, move, state.cache,
        )
        branches[index] = move.branch
        WeightedParticle(
            value=move.value,
            trace=update_trace(
                state.problem, old_stage, new_stage, particle, move, state.cache,
            ),
            log_weight=particle.log_weight + increments[index],
            lineage=particle.lineage,
            branch=move.branch,
        )
    end

    cess = conditional_effective_sample_size(
        [particle.log_weight for particle in cloud.particles], increments,
    )
    log_increment = normalize_logweights!(proposed)
    cloud.log_normalizer += log_increment
    cloud.particles = proposed
    cloud.stage = new_stage

    resampled = effective_sample_size(cloud) < config.ess_threshold * length(proposed)
    resampling_parents = resampled ? _resample!(state, config, new_stage) : collect(eachindex(proposed))
    invariant_parents, acceptance = resampled ?
        _invariant_moves!(state, config, new_stage) :
        (collect(eachindex(proposed)), 0.0)

    push!(state.ancestry, StageAncestry(
        stage=new_stage,
        proposal_parents=collect(eachindex(proposed)),
        resampling_parents=resampling_parents,
        invariant_parents=invariant_parents,
        branches=branches,
    ))
    push!(state.diagnostics, StageDiagnostics(
        stage=new_stage,
        ess=effective_sample_size(cloud),
        cess=cess,
        log_normalizer_increment=log_increment,
        resampled=resampled,
        invariant_acceptance=acceptance,
    ))
end

function update_smc!(state::SequentialState, observation, config::SMCConfig)
    target_observation = state.cloud.stage.observation + 1
    observe!(state.problem, observation)
    while state.cloud.stage.observation < target_observation || state.cloud.stage.λ < 1.0
        stages = stages_for_observation(
            config.scheduler, state.problem, observation, state.cloud, state.cache,
        )
        isempty(stages) && error("stage scheduler did not reach the observation target")
        new_stage = only(stages)
        old_stage = state.cloud.stage
        _paired_move!(state, config, old_stage, new_stage, 1)
        for move_index in 2:config.paired_moves_per_stage
            _paired_move!(state, config, new_stage, new_stage, move_index)
        end
        push!(state.summaries, summarize(
            state.problem, state.cloud, new_stage, state.cache,
        ))
    end
    state
end

function run_smc(problem::AbstractBehaviorInferenceProblem, observations, config::SMCConfig)
    state = initialize_smc(problem, config)
    for observation in observations
        update_smc!(state, observation, config)
    end
    SMCResult(state=state)
end

posterior_particles(state::SequentialState) = state.cloud.particles
posterior_particles(result::SMCResult) = posterior_particles(result.state)
