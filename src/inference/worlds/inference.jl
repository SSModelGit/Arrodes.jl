using GenSMCP3: @kernel

struct WorldBehaviorFactor <: Gen.Distribution{Bool} end

const world_behavior_factor = WorldBehaviorFactor()

function Gen.logpdf(
    ::WorldBehaviorFactor,
    observed::Bool,
    problem::WorldInferenceProblem,
    timestep::Int,
    coefficients::AbstractVector,
    cache::Dict{Symbol,Any},
)
    observed || return -Inf
    world_logscore(problem, timestep, coefficients, cache)
end

Gen.random(::WorldBehaviorFactor, problem, timestep, coefficients, cache) = true
Gen.is_discrete(::WorldBehaviorFactor) = true
Gen.has_output_grad(::WorldBehaviorFactor) = false
Gen.has_argument_grads(::WorldBehaviorFactor) = (false, false, false, false)

@gen function world_model(
    problem::WorldInferenceProblem,
    cache::Dict{Symbol,Any},
    timestep::Int,
)
    coefficients = {:coefficients} ~ Gen.mvnormal(
        problem.context.model.ϕ,
        problem.context.prior_covariance,
    )
    {:behavior} ~ world_behavior_factor(
        problem,
        timestep,
        coefficients,
        cache,
    )
    coefficients
end

function default_world_proposal()
    Dict{Symbol,Any}(
        :mechanism => :random_walk,
        :scale => 1.0,
    )
end

regularized_covariance(covariance) = Matrix(Symmetric(
    covariance + sqrt(eps(Float64)) * max(opnorm(covariance), 1.0) * I,
))

function world_logtarget_gradient(problem, timestep, coefficients, cache)
    prior = problem.context.prior_covariance \ (
        coefficients - problem.context.model.ϕ
    )
    -prior + world_logscore_gradient(
        problem,
        timestep,
        coefficients,
        cache;
        finite_difference=1e-4,
    )
end

function proposal_moments(coefficients, problem, timestep, proposal, cache)
    mechanism = proposal[:mechanism]
    @match mechanism begin
        :random_walk => Dict(
            :mean => coefficients,
            :covariance => regularized_covariance(
                proposal[:scale] .* SCRIBE.eof_process_covariance(
                    problem.context.model,
                ),
            ),
        )
        :pcn => begin
            correlation = proposal[:pcn_correlation]
            Dict(
                :mean => problem.context.model.ϕ + correlation .* (
                    coefficients - problem.context.model.ϕ
                ),
                :covariance => regularized_covariance(
                    (1 - correlation^2) .* problem.context.prior_covariance,
                ),
            )
        end
        :langevin => begin
            step = proposal[:langevin_step]
            preconditioner = regularized_covariance(
                SCRIBE.eof_process_covariance(problem.context.model),
            )
            gradient = world_logtarget_gradient(
                problem,
                timestep,
                coefficients,
                cache,
            )
            Dict(
                :mean => coefficients + 0.5step .* (preconditioner * gradient),
                :covariance => regularized_covariance(step .* preconditioner),
            )
        end
    end
end

@kernel function world_forward(
    previous_trace,
    problem::WorldInferenceProblem,
    cache::Dict{Symbol,Any},
    timestep::Int,
    proposal::Dict{Symbol,Any},
)
    old_coefficients = GenTraceKernelDSL.get_undualed(
        previous_trace,
        :coefficients,
    )
    moments = proposal_moments(
        old_coefficients,
        problem,
        timestep,
        proposal,
        cache,
    )
    coefficients ~ Gen.mvnormal(moments[:mean], moments[:covariance])
    return Gen.choicemap((:coefficients, coefficients)), Gen.choicemap(
        (:coefficients, previous_trace[:coefficients]),
    )
end

@kernel function world_backward(
    updated_trace,
    problem::WorldInferenceProblem,
    cache::Dict{Symbol,Any},
    timestep::Int,
    proposal::Dict{Symbol,Any},
)
    new_coefficients = GenTraceKernelDSL.get_undualed(
        updated_trace,
        :coefficients,
    )
    moments = proposal_moments(
        new_coefficients,
        problem,
        timestep,
        proposal,
        cache,
    )
    coefficients ~ Gen.mvnormal(
        moments[:mean],
        moments[:covariance],
    )
    return Gen.choicemap((:coefficients, coefficients)), Gen.choicemap(
        (:coefficients, updated_trace[:coefficients]),
    )
end

@kernel function world_rejuvenation(
    trace,
    problem::WorldInferenceProblem,
    cache::Dict{Symbol,Any},
    timestep::Int,
    proposal::Dict{Symbol,Any},
)
    old_coefficients = GenTraceKernelDSL.get_undualed(trace, :coefficients)
    moments = proposal_moments(
        old_coefficients,
        problem,
        timestep,
        proposal,
        cache,
    )
    coefficients ~ Gen.mvnormal(moments[:mean], moments[:covariance])
    return Gen.choicemap((:coefficients, coefficients)), Gen.choicemap(
        (:coefficients, trace[:coefficients]),
    )
end

function particle_moments(state)
    particles = hcat((trace[:coefficients] for trace in state.traces)...)
    weights = GenParticleFilters.get_norm_weights(state)
    coefficient_mean = Vector{Float64}(mean(state, :coefficients))
    centered = particles .- coefficient_mean
    covariance = regularized_covariance(centered * Diagonal(weights) * centered')
    Dict(
        :particles => particles,
        :weights => weights,
        :mean => coefficient_mean,
        :covariance => covariance,
    )
end

function world_mh(trace, problem, cache, timestep, proposal, rng)
    proposed, log_acceptance = GenTraceKernelDSL.run_mcmc_kernel(
        trace,
        world_rejuvenation,
        (problem, cache, timestep, proposal),
    )
    log(rand(rng)) < log_acceptance ? (proposed, true) : (trace, false)
end

function infer_world(
    problem::WorldInferenceProblem;
    n_particles::Int=256,
    ess_threshold::Float64=0.5,
    resampling::Symbol=:residual,
    rejuvenation_steps::Int=1,
    proposal::Dict{Symbol,Any}=default_world_proposal(),
    check_inverses::Bool=false,
    rng::AbstractRNG=Random.default_rng(),
)
    cache = Dict{Symbol,Any}()
    constraints = Gen.choicemap((:behavior, true))
    state = GenParticleFilters.pf_initialize(
        world_model,
        (problem, cache, 0),
        constraints,
        n_particles;
        dynamic=true,
    )
    horizon = length(problem.observations)
    dimension = length(problem.context.model.ϕ)
    means = Matrix{Float64}(undef, dimension, horizon + 1)
    covariances = Vector{Matrix{Float64}}(undef, horizon + 1)
    ess = Vector{Float64}(undef, horizon + 1)
    resampled = falses(horizon)
    moments = particle_moments(state)
    initial_particles = copy(moments[:particles])
    means[:, 1] = moments[:mean]
    covariances[1] = moments[:covariance]
    ess[1] = GenParticleFilters.effective_sample_size(state)

    for timestep in 1:horizon
        update = GenSMCP3.SMCP3Update(
            world_forward,
            world_backward,
            (problem, cache, timestep, proposal),
            (problem, cache, timestep, proposal),
            check_inverses,
        )
        GenParticleFilters.pf_update!(
            state,
            (problem, cache, timestep),
            (
                Gen.UnknownChange(),
                Gen.UnknownChange(),
                Gen.UnknownChange(),
            ),
            constraints,
            update,
        )
        ess[timestep + 1] = GenParticleFilters.effective_sample_size(state)
        if timestep < horizon && ess[timestep + 1] < ess_threshold * n_particles
            GenParticleFilters.pf_resample!(state, resampling)
            resampled[timestep] = true
            if rejuvenation_steps > 0
                move = (trace, problem, cache, timestep, proposal) ->
                    world_mh(
                        trace,
                        problem,
                        cache,
                        timestep,
                        proposal,
                        rng,
                    )
                GenParticleFilters.pf_rejuvenate!(
                    state,
                    move,
                    (problem, cache, timestep, proposal),
                    rejuvenation_steps,
                )
            end
        end
        moments = particle_moments(state)
        means[:, timestep + 1] = moments[:mean]
        covariances[timestep + 1] = moments[:covariance]
    end

    final = particle_moments(state)
    WorldInferenceResult(
        model=problem.context.model,
        coefficient_means=means,
        coefficient_covariances=covariances,
        ess_history=ess,
        resampled=resampled,
        initial_particles=initial_particles,
        final_particles=final[:particles],
        final_weights=final[:weights],
    )
end

function world_posterior(result::WorldInferenceResult, timestep::Int=size(
    result.coefficient_means,
    2,
))
    mean = result.coefficient_means[:, timestep]
    covariance = result.coefficient_covariances[timestep]
    Dict(
        :coefficient_mean => mean,
        :coefficient_covariance => covariance,
        :map_mean => SCRIBE.reconstruct_eof_field(
            result.model;
            coefficients=mean,
        ),
    )
end
