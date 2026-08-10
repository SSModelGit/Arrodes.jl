export BehaviorInformationDeployment, deploy_behavior_information
export blended_coefficients

@with_kw_noshow struct BehaviorInformationDeployment
    accepted::Bool
    information::Union{Nothing,SCRIBE.KFEnvInfo}
    ΔY::Matrix{Float64}
    Δy::Vector{Float64}
    projection_loss::Float64
    reason::Symbol
end

function deploy_behavior_information(problem::WorldInferenceProblem,
                                     summary::WorldPosteriorSummary;
                                     confidence::Float64=1.0,
                                     project::Bool=false)
    context = problem.context
    posterior_precision = inv(Symmetric(summary.coefficient_covariance))
    prior_precision = inv(Symmetric(context.prior_covariance))
    ΔY = Matrix(Symmetric(posterior_precision - prior_precision))
    Δy = posterior_precision * summary.coefficient_mean -
        prior_precision * context.prior_mean
    eigenvalues, eigenvectors = eigen(Symmetric(ΔY))
    tolerance = sqrt(eps(Float64)) * max(opnorm(ΔY), 1.0)
    exact = minimum(eigenvalues) >= -tolerance
    if !exact && !project
        return BehaviorInformationDeployment(
            accepted=false,
            information=nothing,
            ΔY=ΔY,
            Δy=Δy,
            projection_loss=0.0,
            reason=:indefinite_information,
        )
    end
    deployed_ΔY = exact ? ΔY :
        eigenvectors * Diagonal(max.(eigenvalues, 0.0)) * eigenvectors'
    projection_loss = norm(deployed_ΔY - ΔY)
    κ = clamp(confidence, 0.0, 1.0)
    Y = Matrix(Symmetric(context.information.Y + κ .* deployed_ΔY))
    y = context.information.y + κ .* Δy
    information = SCRIBE.KFEnvInfo(
        Vector{Float64}(y),
        Y,
        zeros(length(y)),
        zeros(length(y), length(y)),
    )
    BehaviorInformationDeployment(
        accepted=true,
        information=information,
        ΔY=ΔY,
        Δy=Δy,
        projection_loss=projection_loss,
        reason=exact ? :exact_gaussian : :projected_gaussian,
    )
end

function blended_coefficients(problem::WorldInferenceProblem,
                              summary::WorldPosteriorSummary;
                              confidence::Float64=1.0,
                              trust_radius::Float64=2.0)
    trust_radius > 0 || throw(ArgumentError("deployment trust radius must be positive"))
    maturity_value = problem.evidence isa DirectErgodicEvidence ?
        maturity(problem.evidence.energy, summary.stage.observation) : 1.0
    difference = summary.coefficient_mean - problem.context.prior_mean
    prior_factor = gaussian_factor(problem.context.prior_covariance)
    distance = norm(prior_factor.L \ difference)
    distance == 0 && return copy(problem.context.prior_mean)
    normalized_confidence = clamp(confidence, 0.0, 1.0)
    step_length = maturity_value * normalized_confidence * trust_radius *
        tanh(distance / trust_radius)
    problem.context.prior_mean + (step_length / distance) .* difference
end
