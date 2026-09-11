using GenSMCP3: @kernel

struct WorldBehaviorFactor <: Gen.Distribution{Bool} end

const world_behavior_factor = WorldBehaviorFactor()

function Gen.logpdf(
    ::WorldBehaviorFactor,
    observed::Bool,
    problem::WorldInferenceProblem{EOFWorldInferenceContext},
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
    problem::WorldInferenceProblem{EOFWorldInferenceContext},
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

function gauss_newton_moments(problem, timestep, proposal, cache)
    moments = get!(cache, :gauss_newton_moments) do
        Dict{Int,Dict{Symbol,Any}}()
    end
    get!(moments, timestep) do
        prior_mean = problem.context.model.ϕ
        prior_covariance = problem.context.prior_covariance
        timestep == 0 && return Dict(
            :mean => prior_mean,
            :covariance => prior_covariance,
        )

        coefficients = copy(gauss_newton_moments(
            problem,
            timestep - 1,
            proposal,
            cache,
        )[:mean])
        prior_precision = inv(Symmetric(prior_covariance))
        score = problem.score
        query_fraction = query_weight(score, timestep)
        β = score.β_max * maturity(score, timestep)
        kernel = get!(cache, :target_kernel) do
            kernel_matrix(
                score.kernel_bandwidth,
                problem.context.kernel_locations,
                problem.context.kernel_locations,
            )
        end

        for _ in 1:proposal[:optimizer_steps]
            jacobian = target_measure_jacobian(problem, coefficients)
            precision = Matrix(Symmetric(
                prior_precision +
                2β * (1 - query_fraction) / score.discrepancy_scale .* (
                    jacobian' * kernel * jacobian
                ),
            ))
            gradient = world_logtarget_gradient(
                problem,
                timestep,
                coefficients,
                cache,
            )
            direction = precision \ gradient
            current_score = -0.5dot(
                coefficients - prior_mean,
                prior_precision * (coefficients - prior_mean),
            ) + world_logscore(problem, timestep, coefficients, cache)
            step = 1.0
            accepted = false
            candidate = coefficients
            while step > 1 / 128
                candidate = coefficients + step .* direction
                candidate_score = -0.5dot(
                    candidate - prior_mean,
                    prior_precision * (candidate - prior_mean),
                ) + world_logscore(problem, timestep, candidate, cache)
                if candidate_score >= current_score
                    accepted = true
                    break
                end
                step /= 2
            end
            accepted || break
            coefficients .= candidate
            norm(direction) * step < 1e-6 && break
        end

        jacobian = target_measure_jacobian(problem, coefficients)
        precision = Matrix(Symmetric(
            prior_precision +
            2β * (1 - query_fraction) / score.discrepancy_scale .* (
                jacobian' * kernel * jacobian
            ),
        ))
        Dict(
            :mean => coefficients,
            :covariance => regularized_covariance(
                proposal[:covariance_scale] .* inv(Symmetric(precision)),
            ),
        )
    end
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
                :covariance => Matrix(Symmetric(
                    (1 - correlation^2) .* problem.context.prior_covariance,
                )),
            )
        end
        :langevin => begin
            step = proposal[:langevin_step]
            preconditioner = problem.context.prior_covariance
            gradient = world_logtarget_gradient(
                problem,
                timestep,
                coefficients,
                cache,
            )
            Dict(
                :mean => coefficients + 0.5step .* (preconditioner * gradient),
                :covariance => step .* preconditioner,
            )
        end
        :gauss_newton => gauss_newton_moments(
            problem,
            timestep,
            proposal,
            cache,
        )
    end
end

function affine_gaussian_transport(coefficients, old_moments, new_moments)
    old_factor = cholesky(Symmetric(old_moments[:covariance])).L
    new_factor = cholesky(Symmetric(new_moments[:covariance])).L
    new_moments[:mean] + new_factor * (
        old_factor \ (coefficients - old_moments[:mean])
    )
end

@kernel function world_forward(
    previous_trace,
    problem::WorldInferenceProblem{EOFWorldInferenceContext},
    cache::Dict{Symbol,Any},
    timestep::Int,
    proposal::Dict{Symbol,Any},
)
    if proposal[:mechanism] == :gauss_newton
        old_moments = gauss_newton_moments(
            problem,
            timestep - 1,
            proposal,
            cache,
        )
        new_moments = gauss_newton_moments(
            problem,
            timestep,
            proposal,
            cache,
        )
        coefficients = affine_gaussian_transport(
            previous_trace[:coefficients],
            old_moments,
            new_moments,
        )
        return Gen.choicemap((:coefficients, coefficients)), Gen.choicemap()
    end
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
    problem::WorldInferenceProblem{EOFWorldInferenceContext},
    cache::Dict{Symbol,Any},
    timestep::Int,
    proposal::Dict{Symbol,Any},
)
    if proposal[:mechanism] == :gauss_newton
        old_moments = gauss_newton_moments(
            problem,
            timestep - 1,
            proposal,
            cache,
        )
        new_moments = gauss_newton_moments(
            problem,
            timestep,
            proposal,
            cache,
        )
        coefficients = affine_gaussian_transport(
            updated_trace[:coefficients],
            new_moments,
            old_moments,
        )
        return Gen.choicemap((:coefficients, coefficients)), Gen.choicemap()
    end
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
    problem::WorldInferenceProblem{EOFWorldInferenceContext},
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

function run_world_diagnostics!(
    diagnostics::AbstractDict{Symbol},
    event::Symbol,
    state, problem, timestep;
    ess, resampled=false, moments=nothing
)
    configured = get(diagnostics, event, nothing)
    if isnothing(configured); return nothing; end
    callbacks = configured isa Function ? (configured,) : configured
    if isempty(callbacks); return nothing; end

    snapshot = isnothing(moments) ? particle_moments(state) : moments
    let payload = Dict{Symbol, Any}(
        :event => event,
        :timestep => timestep,
        :problem => problem,
        :particles => snapshot[:particles],
        :weights => snapshot[:weights],
        :coefficient_mean => snapshot[:mean],
        :coefficient_covariance => snapshot[:covariance],
        :ess => Float64(ess),
        :resampled => Bool(resampled),
    )
        for callback in callbacks
            callback(payload)
        end
    end
    return nothing
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
    problem::WorldInferenceProblem{EOFWorldInferenceContext};
    n_particles::Int=256,
    ess_threshold::Float64=0.5,
    resampling::Symbol=:residual,
    rejuvenation_steps::Int=1,
    diagnostics::AbstractDict{Symbol}=Dict{Symbol,Any}(),
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
    run_world_diagnostics!(
        diagnostics, :post_initialization, state, problem, 0;
        ess=ess[1], moments=moments,
    )

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
        run_world_diagnostics!(
            diagnostics, :post_update, state, problem, timestep;
            ess=ess[timestep + 1],
        )

        if ess[timestep + 1] < ess_threshold * n_particles
            GenParticleFilters.pf_resample!(state, resampling)
            resampled[timestep] = true

            run_world_diagnostics!(
                diagnostics, :post_resampling, state, problem, timestep;
                ess=GenParticleFilters.effective_sample_size(state),
                resampled=true,
            )

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

                run_world_diagnostics!(
                    diagnostics, :post_rejuvenation,
                    state, problem, timestep;
                    ess=GenParticleFilters.effective_sample_size(state),
                    resampled=true,
                )
            end
        end
        moments = particle_moments(state)
        means[:, timestep + 1] = moments[:mean]
        covariances[timestep + 1] = moments[:covariance]

        run_world_diagnostics!(
            diagnostics, :post_timestep, state, problem, timestep;
            ess=GenParticleFilters.effective_sample_size(state),
            resampled=resampled[timestep],
            moments=moments,
        )

    end

    final = particle_moments(state)

    run_world_diagnostics!(
        diagnostics, :post_inference, state, problem, horizon;
        ess=GenParticleFilters.effective_sample_size(state),
        moments=final,
    )

    EOFWorldInferenceResult(
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

function infer_world(problem::WorldInferenceProblem{SOMWorldInferenceContext})
    prior = problem.context.prior_probabilities
    horizon = length(problem.observations)

    posterior = Matrix{Float64}(undef, length(prior), horizon + 1)
    posterior[:, 1] = prior

    cache = Dict{Symbol,Any}()
    for ts in 1:horizon
        log_probs = log.(prior)
        map!(
            v->log_probs[v]+world_logscore(problem, ts, v, cache),
            log_probs, eachindex(prior)
        )
        log_probs .-= maximum(log_probs)
        probabilities = exp.(log_probs)
        posterior[:, ts + 1] = probabilities ./ sum(probabilities)
    end
    SOMWorldInferenceResult(
        model=problem.context.model,
        posterior_probabilities=posterior,
    )
end

function world_posterior(
    result::EOFWorldInferenceResult,
    observation_count::Int=size(result.coefficient_means, 2) - 1,
)
    column = observation_count + 1
    coefficient_mean = view(result.coefficient_means, :, column)
    Dict(
        :coefficient_mean => Vector{Float64}(coefficient_mean),
        :coefficient_covariance => result.coefficient_covariances[column],
        :model_mean => SCRIBE.reconstruct_eof_field(
            result.model;
            coefficients=coefficient_mean,
        ),
    )
end

function world_posterior(
    result::SOMWorldInferenceResult,
    observation_count::Int=size(result.posterior_probabilities, 2) - 1,
)
    column = observation_count + 1
    probabilities = view(result.posterior_probabilities, :, column)
    vertices = result.model.data[:vertices]
    map_vertex = argmax(probabilities)

    model_mean = mapreduce(
        vertex -> probabilities[vertex] .* vertices[vertex],
        +,
        eachindex(probabilities),
    )
    Dict(
        :model_mean => model_mean,
        :map_model => vertices[map_vertex],
        :map_vertex => map_vertex,
        :map_vertex_id => result.model.data[:vertex_ids][map_vertex],
        :map_probability => probabilities[map_vertex],
        :probabilities => Vector{Float64}(probabilities),
    )
end
