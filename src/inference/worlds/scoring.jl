candidate_model(context::WorldInferenceContext, coefficients) =
    SCRIBE.EOFClimateModel(
        SCRIBE.get_model_time(context.model),
        context.model.params,
        coefficients,
    )

candidate_field(context::WorldInferenceContext, coefficients) =
    context.quadrature_mean + context.quadrature_basis * coefficients

kernel_value(bandwidth, left, right) =
    exp(-sum(abs2, left - right) / (2bandwidth^2))

function kernel_matrix(bandwidth, left, right)
    [
        kernel_value(bandwidth, view(left, i, :), view(right, j, :))
        for i in axes(left, 1), j in axes(right, 1)
    ]
end

function trajectory_locations(problem::WorldInferenceProblem, timestep)
    reduce(vcat, (
        reshape(Float64.(problem.score.location(observation.state)), 1, :)
        for observation in problem.observations[1:timestep]
    ))
end

function occupation_weights(problem::WorldInferenceProblem, timestep)
    weights = [
        observation.dwell_time for observation in problem.observations[1:timestep]
    ]
    weights ./ sum(weights)
end

function target_measure(problem::WorldInferenceProblem, coefficients)
    context = problem.context
    field = candidate_field(context, coefficients)
    density = problem.score.target.density(
        candidate_model(context, coefficients),
        context.quadrature,
        field,
        context,
    )
    masses = Float64.(density) .* context.quadrature_weights
    all(isfinite, masses) && all(>=(0), masses) && sum(masses) > 0 ||
        error("The target field must define finite, nonnegative probability mass")
    masses ./ sum(masses)
end

function measure_mmd(problem, left, right, cache)
    kernel = get!(cache, :target_kernel) do
        kernel_matrix(
            problem.score.kernel_bandwidth,
            problem.context.kernel_locations,
            problem.context.kernel_locations,
        )
    end
    difference = left - right
    max(dot(difference, kernel * difference), 0.0)
end

"""Squared kernel MMD between two normalized world-induced target measures."""
function target_measure_mmd(
    problem::WorldInferenceProblem,
    left_coefficients,
    right_coefficients,
    cache=Dict{Symbol,Any}(),
)
    measure_mmd(
        problem,
        target_measure(problem, left_coefficients),
        target_measure(problem, right_coefficients),
        cache,
    )
end

function posterior_target_measure(problem, result::WorldInferenceResult)
    posterior = zeros(size(problem.context.quadrature, 1))
    for index in axes(result.final_particles, 2)
        posterior .+= result.final_weights[index] .* target_measure(
            problem,
            view(result.final_particles, :, index),
        )
    end
    posterior
end

function target_measure_mmd(
    problem::WorldInferenceProblem,
    result::WorldInferenceResult,
    coefficients,
    cache=Dict{Symbol,Any}(),
)
    measure_mmd(
        problem,
        posterior_target_measure(problem, result),
        target_measure(problem, coefficients),
        cache,
    )
end

function target_measure_jacobian(problem, coefficients; finite_difference=1e-4)
    context = problem.context
    score = problem.score
    target = target_measure(problem, coefficients)
    if !isnothing(score.target.jacobian)
        return Matrix{Float64}(score.target.jacobian(
            candidate_model(context, coefficients),
            context.quadrature,
            candidate_field(context, coefficients),
            target,
            context,
        ))
    end
    jacobian = Matrix{Float64}(undef, length(target), length(coefficients))
    for index in eachindex(coefficients)
        left = copy(coefficients)
        right = copy(coefficients)
        left[index] -= finite_difference
        right[index] += finite_difference
        jacobian[:, index] = (
            target_measure(problem, right) - target_measure(problem, left)
        ) / (2finite_difference)
    end
    jacobian
end

function trajectory_kernel_statistics(problem, timestep, cache)
    statistics = get!(cache, :trajectory_kernels) do
        Dict{Int,Dict{Symbol,Any}}()
    end
    get!(statistics, timestep) do
        locations = trajectory_locations(problem, timestep)
        weights = occupation_weights(problem, timestep)
        empirical_kernel = kernel_matrix(
            problem.score.kernel_bandwidth,
            locations,
            locations,
        )
        cross_kernel = kernel_matrix(
            problem.score.kernel_bandwidth,
            locations,
            problem.context.kernel_locations,
        )
        Dict(
            :empirical_energy => dot(weights, empirical_kernel, weights),
            :target_cross_mean => cross_kernel' * weights,
        )
    end
end

function measure_discrepancy(problem, timestep, target, cache)
    timestep == 0 && return 0.0
    target_kernel = get!(cache, :target_kernel) do
        kernel_matrix(
            problem.score.kernel_bandwidth,
            problem.context.kernel_locations,
            problem.context.kernel_locations,
        )
    end
    trajectory = trajectory_kernel_statistics(problem, timestep, cache)
    max(
        trajectory[:empirical_energy] -
        2dot(trajectory[:target_cross_mean], target) +
        dot(target, target_kernel, target),
        0.0,
    )
end

function kernel_discrepancy(problem, timestep, coefficients, cache=Dict{Symbol,Any}())
    measure_discrepancy(
        problem,
        timestep,
        target_measure(problem, coefficients),
        cache,
    )
end

function kernel_discrepancy(
    problem,
    timestep,
    result::WorldInferenceResult,
    cache=Dict{Symbol,Any}(),
)
    measure_discrepancy(
        problem,
        timestep,
        posterior_target_measure(problem, result),
        cache,
    )
end

function query_weight(score::ErgodicBehaviorScore, timestep)
    isempty(score.query_weights) && return 0.0
    horizons = sort(collect(keys(score.query_weights)))
    index = searchsortedlast(horizons, timestep)
    score.query_weights[horizons[clamp(index, 1, length(horizons))]]
end

maturity(score::ErgodicBehaviorScore, timestep) = timestep == 0 ? 0.0 :
    timestep^score.maturity_power /
    (timestep^score.maturity_power + score.maturity_half_time^score.maturity_power)

function trajectory_query(problem, timestep, coefficients)
    isnothing(problem.score.query) && return 0.0
    model = candidate_model(problem.context, coefficients)
    values = [
        problem.score.query(model, observation, problem.context)
        for observation in problem.observations[1:timestep]
    ]
    dot(occupation_weights(problem, timestep), values)
end

function trajectory_query_gradient(problem, timestep, coefficients; finite_difference=1e-4)
    isnothing(problem.score.query) && return zeros(length(coefficients))
    if !isnothing(problem.score.query_gradient)
        model = candidate_model(problem.context, coefficients)
        gradients = [
            problem.score.query_gradient(model, observation, problem.context)
            for observation in problem.observations[1:timestep]
        ]
        return sum(
            weight .* gradient
            for (weight, gradient) in zip(
                occupation_weights(problem, timestep),
                gradients,
            )
        )
    end
    gradient = zeros(length(coefficients))
    for index in eachindex(coefficients)
        left = copy(coefficients)
        right = copy(coefficients)
        left[index] -= finite_difference
        right[index] += finite_difference
        gradient[index] = (
            trajectory_query(problem, timestep, right) -
            trajectory_query(problem, timestep, left)
        ) / (2finite_difference)
    end
    gradient
end

function world_score_components(problem, timestep, coefficients, cache=Dict{Symbol,Any}())
    score = problem.score
    weight = query_weight(score, timestep)
    discrepancy = weight < 1 ? kernel_discrepancy(
        problem, timestep, coefficients, cache,
    ) : 0.0
    query = weight > 0 ? trajectory_query(problem, timestep, coefficients) : 0.0
    β = score.β_max * maturity(score, timestep)
    logscore = -β * (
        (1 - weight) * discrepancy / score.discrepancy_scale -
        weight * (query - score.query_reference) / score.query_scale
    )
    Dict(
        :logscore => logscore,
        :mmd => discrepancy,
        :query => query,
        :β => β,
        :query_weight => weight,
    )
end

function world_logscore(problem, timestep, coefficients, cache=Dict{Symbol,Any}())
    score = problem.score
    weight = query_weight(score, timestep)
    discrepancy = weight < 1 ? kernel_discrepancy(
        problem, timestep, coefficients, cache,
    ) : 0.0
    query = weight > 0 ? trajectory_query(problem, timestep, coefficients) : 0.0
    β = score.β_max * maturity(score, timestep)
    -β * (
        (1 - weight) * discrepancy / score.discrepancy_scale -
        weight * (query - score.query_reference) / score.query_scale
    )
end

function world_logscore_gradient(
    problem,
    timestep,
    coefficients,
    cache;
    finite_difference=1e-4,
)
    timestep == 0 && return zeros(length(coefficients))
    score = problem.score
    weight = query_weight(score, timestep)
    discrepancy_gradient = if weight < 1
        target = target_measure(problem, coefficients)
        target_jacobian = target_measure_jacobian(
            problem,
            coefficients;
            finite_difference,
        )
        target_kernel = get!(cache, :target_kernel) do
            kernel_matrix(
                score.kernel_bandwidth,
                problem.context.kernel_locations,
                problem.context.kernel_locations,
            )
        end
        trajectory = trajectory_kernel_statistics(problem, timestep, cache)
        2target_jacobian' * (
            target_kernel * target - trajectory[:target_cross_mean]
        )
    else
        zeros(length(coefficients))
    end
    query_gradient = weight > 0 ? trajectory_query_gradient(
        problem, timestep, coefficients; finite_difference,
    ) : zeros(length(coefficients))
    β = score.β_max * maturity(score, timestep)
    -β .* (
        (1 - weight) / score.discrepancy_scale .* discrepancy_gradient -
        weight / score.query_scale .* query_gradient
    )
end
