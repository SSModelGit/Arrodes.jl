@with_kw_noshow struct WorldInferenceContext
    model::SCRIBE.EOFClimateModel
    prior_covariance::Matrix{Float64}
    quadrature::Matrix{Float64}
    kernel_locations::Matrix{Float64}
    quadrature_weights::Vector{Float64}
    quadrature_mean::Vector{Float64}
    quadrature_basis::Matrix{Float64}
end

@with_kw_noshow struct TrajectoryObservation{S,A}
    state::S
    action::A = nothing
    dwell_time::Float64 = 1.0
end

@with_kw_noshow struct ErgodicTargetField
    density::Function
    jacobian::Union{Nothing,Function} = nothing
    name::Symbol = :custom
end

@with_kw_noshow struct ErgodicBehaviorScore
    location::Function
    target::ErgodicTargetField
    kernel_bandwidth::Float64
    discrepancy_scale::Float64
    β_max::Float64
    maturity_half_time::Float64
    maturity_power::Float64
    query::Union{Nothing,Function} = nothing
    query_gradient::Union{Nothing,Function} = nothing
    query_scale::Float64 = 1.0
    query_reference::Float64 = 0.0
    query_weights::Dict{Int,Float64} = Dict{Int,Float64}()
end

@with_kw_noshow struct WorldInferenceProblem
    context::WorldInferenceContext
    score::ErgodicBehaviorScore
    observations::Vector{TrajectoryObservation}
end

@with_kw_noshow struct WorldInferenceResult
    model::SCRIBE.EOFClimateModel
    coefficient_means::Matrix{Float64}
    coefficient_covariances::Vector{Matrix{Float64}}
    ess_history::Vector{Float64}
    resampled::Vector{Bool}
    initial_particles::Matrix{Float64}
    final_particles::Matrix{Float64}
    final_weights::Vector{Float64}
end

function world_inference_context(
    model::SCRIBE.EOFClimateModel;
    prior_covariance::AbstractMatrix=SCRIBE.eof_prior_covariance(model),
    quadrature::AbstractMatrix=model.params.locations,
    kernel_locations::AbstractMatrix=quadrature,
    quadrature_weights::AbstractVector=model.params.decomposition.weights,
)
    covariance = Matrix{Float64}(prior_covariance)
    dimension = length(model.ϕ)
    size(covariance) == (dimension, dimension) ||
        error("The world prior covariance must match the EOF coefficient dimension")
    cholesky(Symmetric(covariance); check=true)
    size(quadrature, 1) == length(quadrature_weights) ||
        error("Quadrature locations and weights must have the same length")
    size(kernel_locations) == size(quadrature) ||
        error("Kernel locations must match the quadrature locations")
    weights = Float64.(quadrature_weights)
    all(isfinite, weights) && all(>=(0), weights) && sum(weights) > 0 ||
        error("Quadrature weights must define finite, nonnegative probability mass")
    weights ./= sum(weights)
    WorldInferenceContext(
        model=model,
        prior_covariance=covariance,
        quadrature=Matrix{Float64}(quadrature),
        kernel_locations=Matrix{Float64}(kernel_locations),
        quadrature_weights=weights,
        quadrature_mean=Vector{Float64}(SCRIBE.eof_mean_at(model, quadrature)),
        quadrature_basis=Matrix{Float64}(SCRIBE.eof_basis_at(model, quadrature)),
    )
end

function normalized_density_jacobian(values, derivative, target, weights)
    total = dot(weights, values)
    weighted_derivative = reshape(weights, :, 1) .* derivative
    total_derivative = vec(sum(weighted_derivative; dims=1))
    (weighted_derivative .- reshape(target, :, 1) .* total_derivative') ./ total
end

function eof_target_field(;
    link::Symbol=:softmax,
    scale::Real=1.0,
    floor::Real=sqrt(eps(Float64)),
    name::Symbol=link,
)
    density, jacobian = @match link begin
        :softmax => (
            (model, locations, values, context) -> begin
                logits = values ./ scale
                exp.(logits .- maximum(logits))
            end,
            (model, locations, values, target, context) -> begin
                basis = context.quadrature_basis
                centered = basis .- sum(reshape(target, :, 1) .* basis; dims=1)
                reshape(target, :, 1) .* centered ./ scale
            end,
        )
        :shifted => (
            (model, locations, values, context) ->
                values .- minimum(values) .+ floor,
            (model, locations, values, target, context) -> begin
                minimum_index = argmin(values)
                raw = values .- values[minimum_index] .+ floor
                derivative = context.quadrature_basis .-
                    reshape(context.quadrature_basis[minimum_index, :], 1, :)
                normalized_density_jacobian(
                    raw,
                    derivative,
                    target,
                    context.quadrature_weights,
                )
            end,
        )
        :magnitude => (
            (model, locations, values, context) -> abs.(values) .+ floor,
            (model, locations, values, target, context) ->
                normalized_density_jacobian(
                    abs.(values) .+ floor,
                    reshape(sign.(values), :, 1) .* context.quadrature_basis,
                    target,
                    context.quadrature_weights,
                ),
        )
        :squared => (
            (model, locations, values, context) -> abs2.(values) .+ floor,
            (model, locations, values, target, context) ->
                normalized_density_jacobian(
                    abs2.(values) .+ floor,
                    2 .* reshape(values, :, 1) .* context.quadrature_basis,
                    target,
                    context.quadrature_weights,
                ),
        )
    end
    ErgodicTargetField(density=density, jacobian=jacobian, name=name)
end

function eof_field_score(
    target::ErgodicTargetField;
    kernel_bandwidth::Real,
    discrepancy_scale::Real,
    location::Function=state -> state,
    β_max::Real=7.0,
    maturity_half_time::Real=18.0,
    maturity_power::Real=2.0,
    query::Union{Nothing,Function}=nothing,
    query_gradient::Union{Nothing,Function}=nothing,
    query_scale::Real=1.0,
    query_reference::Real=0.0,
    query_weights::Dict{Int,Float64}=Dict{Int,Float64}(),
)
    ErgodicBehaviorScore(
        location=location,
        target=target,
        kernel_bandwidth=Float64(kernel_bandwidth),
        discrepancy_scale=Float64(discrepancy_scale),
        β_max=Float64(β_max),
        maturity_half_time=Float64(maturity_half_time),
        maturity_power=Float64(maturity_power),
        query=query,
        query_gradient=query_gradient,
        query_scale=Float64(query_scale),
        query_reference=Float64(query_reference),
        query_weights=query_weights,
    )
end
