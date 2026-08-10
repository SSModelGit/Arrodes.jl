export LangevinTransport, AffineAmortizedTransport, PriorRefreshTransport
export PriorPCNKernel, WorldKernelMixture
export gradient_logtarget, whitened_gradient_logtarget, behavior_information

"""A cheap prior-reversible paired proposal for large exploratory runs."""
@with_kw struct PriorPCNKernel <: AbstractPairedKernel
    ρ::Float64 = 0.98
end

function propose(kernel::PriorPCNKernel, problem::WorldInferenceProblem,
                 old_stage::InferenceStage, new_stage::InferenceStage,
                 particle::WeightedParticle, rng, cache)
    0 <= kernel.ρ < 1 || throw(ArgumentError("pCN correlation must lie in [0, 1)"))
    context = problem.context
    covariance = (1 - kernel.ρ^2) .* context.prior_covariance
    factor = get!(cache, :world_pcn_factor) do
        gaussian_factor(covariance)
    end
    forward_mean = context.prior_mean + kernel.ρ .* (
        particle.value - context.prior_mean
    )
    value = gaussian_sample(rng, forward_mean, factor)
    MoveRecord(
        value=value,
        log_forward=0.0,
        log_backward=0.0,
        branch=:prior_pcn,
        metadata=(
            forward_mean=forward_mean,
            prior_reversible=true,
            transport_fraction=1.0,
        ),
    )
end

function paired_logweight(problem::WorldInferenceProblem,
                          old_stage::InferenceStage, new_stage::InferenceStage,
                          particle, move, cache)
    if move.branch === :prior_pcn
        return bridge_compatibility(problem, new_stage, move.value, cache) -
               bridge_compatibility(problem, old_stage, particle.value, cache)
    end
    logtarget(problem, new_stage, move.value, cache) -
    logtarget(problem, old_stage, particle.value, cache) +
    move.log_backward - move.log_forward + move.log_jacobian
end

@with_kw struct LangevinTransport
    step_size::Float64 = 0.12
    information_scale::Float64 = 1.0
    cloud_scale::Float64 = 0.05
    trust_radius::Float64 = 2.5
    finite_difference::Float64 = 1e-4
    regularization::Float64 = 1e-8
end

function default_amortized_transport(problem::WorldInferenceProblem, stage::InferenceStage)
    dimension = length(problem.context.prior_mean)
    (
        A=0.35 .* Matrix{Float64}(I, dimension, dimension),
        b=zeros(dimension),
        covariance=0.65 .* Matrix{Float64}(I, dimension, dimension),
    )
end

@with_kw_noshow struct AffineAmortizedTransport
    parameters::Function = default_amortized_transport
    trust_radius::Float64 = 3.0
end

@with_kw struct PriorRefreshTransport
    inflation::Float64 = 1.5
    trust_radius::Float64 = 4.0
end

@with_kw_noshow struct WorldKernelMixture <: AbstractPairedKernel
    local_transport::LangevinTransport = LangevinTransport()
    amortized::AffineAmortizedTransport = AffineAmortizedTransport()
    refresh::PriorRefreshTransport = PriorRefreshTransport()
    local_min::Float64 = 0.35
    local_max::Float64 = 0.70
    refresh_min::Float64 = 0.10
    refresh_max::Float64 = 0.35
end

@with_kw_noshow struct AffineBranch
    name::Symbol
    probability::Float64
    A::Matrix{Float64}
    b::Vector{Float64}
    covariance::Matrix{Float64}
    covariance_factor
    transport_fraction::Float64
    information::Matrix{Float64}
end

@with_kw_noshow struct BackwardBranch
    predictive_mean::Vector{Float64}
    predictive_factor
    gain::Matrix{Float64}
    conditional_factor
end

@with_kw_noshow struct PreparedWorldKernel <: AbstractPairedKernel
    mixture::WorldKernelMixture
    branches::Vector{AffineBranch}
    backward_branches::Vector{BackwardBranch}
    old_mean::Vector{Float64}
    old_covariance::Matrix{Float64}
    cloud_precision::Matrix{Float64}
    prior_factor::Matrix{Float64}
    logabsdet_prior_factor::Float64
    stage::InferenceStage
end

regularize_covariance(Σ, ε) = Matrix(Symmetric(Σ + ε * I))

function finite_difference_gradient(f, value, step)
    gradient = similar(value, Float64)
    for index in eachindex(value)
        left = copy(value)
        right = copy(value)
        left[index] -= step
        right[index] += step
        gradient[index] = (f(right) - f(left)) / (2step)
    end
    gradient
end

function world_compatibility_gradient(problem, evidence::DirectErgodicEvidence,
                                      t, coefficients, cache, step)
    -world_energy_gradient(
        problem, evidence, t, coefficients, cache; finite_difference=step,
    )
end

function world_compatibility_gradient(problem, evidence::CompositeBehaviorEvidence,
                                      t, coefficients, cache, step)
    sum((world_compatibility_gradient(
        problem, component, t, coefficients, cache, step,
    ) for component in evidence.components); init=zeros(length(coefficients)))
end

function world_compatibility_gradient(problem, evidence::AbstractBehaviorEvidence,
                                      t, coefficients, cache, step)
    finite_difference_gradient(coefficients, step) do candidate
        world_compatibility(problem, evidence, t, candidate, cache)
    end
end

function gradient_logtarget(problem::WorldInferenceProblem, stage::InferenceStage,
                            coefficients, cache, step)
    prior_factor = get!(cache, :world_prior_factor) do
        gaussian_factor(problem.context.prior_covariance)
    end
    prior_gradient = -(prior_factor \ (coefficients - problem.context.prior_mean))
    prior_gradient + world_compatibility_gradient(
        problem,
        problem.evidence,
        stage.observation,
        coefficients,
        cache,
        step,
    )
end

gradient_logtarget(problem, stage, coefficients, cache) =
    gradient_logtarget(problem, stage, coefficients, cache, 1e-4)

function whitened_gradient_logtarget(problem, stage, ξ, prior_factor, cache, step)
    coefficients = problem.context.prior_mean + prior_factor * ξ
    prior_factor' * gradient_logtarget(problem, stage, coefficients, cache, step)
end

function planner_behavior_information(problem, evidence::PlannerWorldEvidence, t,
                                      coefficients, prior_factor, step)
    dimension = length(coefficients)
    information = zeros(dimension, dimension)
    ξ = prior_factor \ (coefficients - problem.context.prior_mean)
    for timestep in 1:t
        score = finite_difference_gradient(ξ, step) do candidate
            planner_world_loglikelihood(
                problem,
                evidence,
                timestep,
                problem.context.prior_mean + prior_factor * candidate,
            )
        end
        information .+= score * score'
    end
    information
end

function behavior_information(problem, evidence::DirectErgodicEvidence, stage,
                              coefficients, prior_factor, cache, step)
    world_information(
        problem,
        evidence,
        stage.observation,
        coefficients,
        prior_factor,
        cache;
        finite_difference=step,
    )
end

function behavior_information(problem, evidence::PlannerWorldEvidence, stage,
                              coefficients, prior_factor, cache, step)
    planner_behavior_information(
        problem, evidence, stage.observation, coefficients, prior_factor, step,
    )
end

function behavior_information(problem, evidence::CompositeBehaviorEvidence, stage,
                              coefficients, prior_factor, cache, step)
    sum((behavior_information(
        problem, component, stage, coefficients, prior_factor, cache, step,
    ) for component in evidence.components); init=zeros(length(coefficients), length(coefficients)))
end

function branch_probabilities(kernel::WorldKernelMixture, problem, stage)
    maturity_value = problem.evidence isa DirectErgodicEvidence ?
        maturity(problem.evidence.energy, stage.observation) :
        stage.observation / (stage.observation + 20)
    local_probability = kernel.local_min + maturity_value * (
        kernel.local_max - kernel.local_min
    )
    refresh = kernel.refresh_min + (1 - maturity_value) * (
        kernel.refresh_max - kernel.refresh_min
    )
    amortized = 1 - local_probability - refresh
    minimum((local_probability, amortized, refresh)) > 0 ||
        throw(ArgumentError("every world proposal branch needs positive probability"))
    (
        local_probability=local_probability,
        amortized=amortized,
        refresh=refresh,
    )
end

function bridged_branch(name, probability, A, b, covariance, trust_radius,
                        reference, metric, regularization)
    dimension = length(reference)
    identity = Matrix{Float64}(I, dimension, dimension)
    displacement = (A - identity) * reference + b
    expected_distance_squared = max(
        dot(displacement, metric * displacement) + tr(metric * covariance),
        0.0,
    )
    expected_distance = sqrt(expected_distance_squared)
    ρ = expected_distance == 0 ? 1.0 : min(1.0, trust_radius / expected_distance)
    realized_covariance = regularize_covariance(
        ρ^2 .* covariance, regularization,
    )
    AffineBranch(
        name=name,
        probability=probability,
        A=identity + ρ .* (A - identity),
        b=ρ .* b,
        covariance=realized_covariance,
        covariance_factor=gaussian_factor(realized_covariance),
        transport_fraction=ρ,
        information=metric,
    )
end

function prepare_kernel(kernel::WorldKernelMixture, problem::WorldInferenceProblem,
                        old_stage, new_stage, cloud, cache)
    local_transport = kernel.local_transport
    prior_cholesky = get!(cache, :world_prior_factor) do
        gaussian_factor(problem.context.prior_covariance)
    end
    prior_factor = Matrix(prior_cholesky.L)
    coefficient_values = [particle.value for particle in cloud.particles]
    log_weights = [particle.log_weight for particle in cloud.particles]
    whitened_values = [prior_factor \ (
        value - problem.context.prior_mean
    ) for value in coefficient_values]
    moments = weighted_mean_covariance(whitened_values, log_weights)
    dimension = length(moments.mean)
    identity = Matrix{Float64}(I, dimension, dimension)
    cloud_covariance = regularize_covariance(
        moments.covariance, local_transport.regularization,
    )
    cloud_precision = inv(Symmetric(cloud_covariance))
    center = problem.context.prior_mean + prior_factor * moments.mean
    information = behavior_information(
        problem,
        problem.evidence,
        new_stage,
        center,
        prior_factor,
        cache,
        local_transport.finite_difference,
    )
    metric = Matrix(Symmetric(
        identity +
        local_transport.information_scale .* information +
        local_transport.cloud_scale .* cloud_precision +
        local_transport.regularization .* identity
    ))
    metric_factor = cholesky(Symmetric(metric); check=true)
    metric_inverse = Matrix(inv(metric_factor))
    score = whitened_gradient_logtarget(
        problem,
        new_stage,
        moments.mean,
        prior_factor,
        cache,
        local_transport.finite_difference,
    )
    curvature = identity + local_transport.information_scale .* information
    score_jacobian = -Matrix(Symmetric(curvature))
    local_A = identity + 0.5local_transport.step_size .*
        metric_inverse * score_jacobian
    local_b = 0.5local_transport.step_size .*
        metric_inverse * (score - score_jacobian * moments.mean)
    local_covariance = local_transport.step_size .* metric_inverse

    amortized = kernel.amortized.parameters(problem, new_stage)
    probabilities = branch_probabilities(kernel, problem, new_stage)
    branches = [
        bridged_branch(
            :local,
            probabilities.local_probability,
            local_A,
            local_b,
            local_covariance,
            local_transport.trust_radius,
            moments.mean,
            metric,
            local_transport.regularization,
        ),
        bridged_branch(
            :amortized,
            probabilities.amortized,
            Matrix{Float64}(amortized.A),
            Vector{Float64}(amortized.b),
            Matrix{Float64}(amortized.covariance),
            kernel.amortized.trust_radius,
            moments.mean,
            identity,
            local_transport.regularization,
        ),
        bridged_branch(
            :refresh,
            probabilities.refresh,
            zeros(dimension, dimension),
            zeros(dimension),
            kernel.refresh.inflation .* identity,
            kernel.refresh.trust_radius,
            moments.mean,
            identity,
            local_transport.regularization,
        ),
    ]
    backward_branches = map(branches) do branch
        predictive_mean = branch.A * moments.mean + branch.b
        predictive_covariance = regularize_covariance(
            branch.A * cloud_covariance * branch.A' + branch.covariance,
            local_transport.regularization,
        )
        gain = cloud_covariance * branch.A' / predictive_covariance
        conditional_covariance = regularize_covariance(
            cloud_covariance - gain * branch.A * cloud_covariance,
            local_transport.regularization,
        )
        BackwardBranch(
            predictive_mean=predictive_mean,
            predictive_factor=gaussian_factor(predictive_covariance),
            gain=gain,
            conditional_factor=gaussian_factor(conditional_covariance),
        )
    end
    PreparedWorldKernel(
        mixture=kernel,
        branches=branches,
        backward_branches=backward_branches,
        old_mean=moments.mean,
        old_covariance=cloud_covariance,
        cloud_precision=Matrix(cloud_precision),
        prior_factor=prior_factor,
        logabsdet_prior_factor=sum(log, diag(prior_factor)),
        stage=new_stage,
    )
end

function select_branch(rng, branches)
    threshold = rand(rng)
    cumulative = 0.0
    for branch in branches
        cumulative += branch.probability
        threshold <= cumulative && return branch
    end
    branches[end]
end

forward_mean(branch::AffineBranch, old_value) = branch.A * old_value + branch.b

function local_forward_parameters(kernel::PreparedWorldKernel,
                                  problem::WorldInferenceProblem, old_value, cache)
    transport = kernel.mixture.local_transport
    dimension = length(old_value)
    identity = Matrix{Float64}(I, dimension, dimension)
    coefficients = problem.context.prior_mean + kernel.prior_factor * old_value
    information = behavior_information(
        problem,
        problem.evidence,
        kernel.stage,
        coefficients,
        kernel.prior_factor,
        cache,
        transport.finite_difference,
    )
    metric = Matrix(Symmetric(
        identity +
        transport.information_scale .* information +
        transport.cloud_scale .* kernel.cloud_precision +
        transport.regularization .* identity
    ))
    metric_inverse = Matrix(inv(cholesky(Symmetric(metric); check=true)))
    score = whitened_gradient_logtarget(
        problem,
        kernel.stage,
        old_value,
        kernel.prior_factor,
        cache,
        transport.finite_difference,
    )
    displacement = 0.5transport.step_size .* metric_inverse * score
    destination_covariance = transport.step_size .* metric_inverse
    expected_distance = sqrt(max(
        dot(displacement, metric * displacement) +
        tr(metric * destination_covariance),
        0.0,
    ))
    ρ = expected_distance == 0 ? 1.0 : min(
        1.0, transport.trust_radius / expected_distance,
    )
    covariance = regularize_covariance(
        ρ^2 .* destination_covariance, transport.regularization,
    )
    (
        mean=old_value + ρ .* displacement,
        covariance_factor=gaussian_factor(covariance),
        transport_fraction=ρ,
        information=metric,
    )
end

function backward_logdensity(kernel::PreparedWorldKernel, selected, old_value, new_value)
    μ = kernel.old_mean
    V = kernel.old_covariance
    scores = map(zip(kernel.branches, kernel.backward_branches)) do pair
        branch, backward = pair
        log(branch.probability) +
            gaussian_logdensity(
                new_value, backward.predictive_mean, backward.predictive_factor,
            )
    end
    branch_index = findfirst(branch -> branch.name === selected.name, kernel.branches)
    branch = kernel.branches[branch_index]
    backward = kernel.backward_branches[branch_index]
    conditional_mean = μ + backward.gain * (
        new_value - branch.A * μ - branch.b
    )
    scores[branch_index] - logsumexp(scores) +
        gaussian_logdensity(
            old_value, conditional_mean, backward.conditional_factor,
        ) -
        kernel.logabsdet_prior_factor
end

function propose(kernel::PreparedWorldKernel, problem::WorldInferenceProblem,
                 old_stage::InferenceStage, new_stage::InferenceStage,
                 particle::WeightedParticle, rng, cache)
    branch = select_branch(rng, kernel.branches)
    old_value = kernel.prior_factor \ (
        particle.value - problem.context.prior_mean
    )
    parameters = branch.name === :local ?
        local_forward_parameters(kernel, problem, old_value, cache) :
        (
            mean=forward_mean(branch, old_value),
            covariance_factor=branch.covariance_factor,
            transport_fraction=branch.transport_fraction,
            information=branch.information,
        )
    new_value = gaussian_sample(rng, parameters.mean, parameters.covariance_factor)
    coefficients = problem.context.prior_mean + kernel.prior_factor * new_value
    log_forward = log(branch.probability) +
        gaussian_logdensity(
            new_value, parameters.mean, parameters.covariance_factor,
        ) -
        kernel.logabsdet_prior_factor
    MoveRecord(
        value=coefficients,
        log_forward=log_forward,
        log_backward=backward_logdensity(
            kernel, branch, old_value, new_value,
        ),
        branch=branch.name,
        metadata=(
            whitened_forward_mean=parameters.mean,
            transport_fraction=parameters.transport_fraction,
            information_metric=parameters.information,
        ),
    )
end
