export scribe_world_context

function scribe_world_context(
    model::SCRIBE.EOFClimateModel,
    information::SCRIBE.KFEnvInfo;
    quadrature::AbstractMatrix=model.params.locations,
    quadrature_weights::AbstractVector=model.params.decomposition.weights,
)
    moments = SCRIBE.posterior_coefficient_moments(information)
    length(model.ϕ) == length(moments.μ) ||
        throw(DimensionMismatch("SCRIBE model and information state use different EOF bases"))
    weights = Float64.(quadrature_weights)
    all(isfinite, weights) && all(>=(0.0), weights) && sum(weights) > 0 ||
        throw(ArgumentError("quadrature weights must be finite, nonnegative, and nonzero"))
    weights ./= sum(weights)
    SCRIBEWorldContext(
        model=model,
        information=information,
        prior_mean=copy(model.ϕ),
        prior_covariance=Matrix(moments.Σ),
        quadrature=Matrix{Float64}(quadrature),
        quadrature_weights=weights,
        quadrature_mean=Vector{Float64}(SCRIBE.eof_mean_at(model, quadrature)),
        quadrature_basis=Matrix{Float64}(SCRIBE.eof_basis_at(model, quadrature)),
        model_time=SCRIBE.get_model_time(model),
    )
end

candidate_model(context::SCRIBEWorldContext, coefficients) =
    SCRIBE.EOFClimateModel(context.model_time, context.model.params, coefficients)

candidate_field(context::SCRIBEWorldContext, coefficients) =
    context.quadrature_mean + context.quadrature_basis * coefficients

candidate_field(context::SCRIBEWorldContext, coefficients, X) =
    SCRIBE.eof_mean_at(context.model, X) + SCRIBE.eof_basis_at(context.model, X) * coefficients

function gaussian_factor(Σ)
    cholesky(Symmetric(Matrix{Float64}(Σ)); check=true)
end

function gaussian_logdensity(x, μ, Σ)
    gaussian_logdensity(x, μ, gaussian_factor(Σ))
end

function gaussian_logdensity(x, μ, factor::LinearAlgebra.Cholesky)
    residual = x - μ
    -0.5 * (length(x) * log(2π) + 2sum(log, diag(factor.L)) + dot(residual, factor \ residual))
end

gaussian_sample(rng, μ, Σ) = μ + gaussian_factor(Σ).L * randn(rng, length(μ))
gaussian_sample(rng, μ, factor::LinearAlgebra.Cholesky) =
    μ + factor.L * randn(rng, length(μ))

function initial_proposal(problem::WorldInferenceProblem, rng, cache)
    context = problem.context
    factor = get!(cache, :world_prior_factor) do
        gaussian_factor(context.prior_covariance)
    end
    value = gaussian_sample(rng, context.prior_mean, factor)
    (
        value=value,
        trace=WorldTrace(),
        logdensity=gaussian_logdensity(value, context.prior_mean, factor),
    )
end
