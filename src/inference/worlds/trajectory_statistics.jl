export kernel_discrepancy, target_measure, trajectory_mean_reward
export maturity, evidence_mixture, world_energy
export target_measure_jacobian, world_energy_gradient, world_information

kernel_value(kernel::GaussianDiscrepancyKernel, x, y) =
    exp(-sum(abs2, x - y) / (2kernel.bandwidth^2))

function trajectory_locations(problem::WorldInferenceProblem, evidence::DirectErgodicEvidence, t)
    reduce(vcat, (reshape(Float64.(evidence.location(problem.trajectory.states[i])), 1, :)
                  for i in 1:t))
end

function occupation_weights(trajectory::BehaviorTrajectory, t)
    weights = trajectory.dwell_times[1:t]
    weights ./ sum(weights)
end

function target_measure(problem::WorldInferenceProblem, evidence::DirectErgodicEvidence,
                        coefficients)
    context = problem.context
    model = candidate_model(context, coefficients)
    values = candidate_field(context, coefficients)
    masses = Float64.(evidence.importance(model, context.quadrature, values)) .*
        context.quadrature_weights
    all(isfinite, masses) || error("candidate target measure contains non-finite mass")
    all(>=(0.0), masses) || error("candidate target measure contains negative mass")
    total_mass = sum(masses)
    total_mass > 0 || error("candidate target measure has zero mass")
    masses ./ total_mass
end

function target_measure_jacobian(
    problem::WorldInferenceProblem,
    evidence::DirectErgodicEvidence,
    coefficients;
    finite_difference=1e-4,
)
    context = problem.context
    model = candidate_model(context, coefficients)
    values = candidate_field(context, coefficients)
    target = target_measure(problem, evidence, coefficients)
    if !isnothing(evidence.target_jacobian)
        jacobian = Matrix{Float64}(evidence.target_jacobian(
            model,
            context.quadrature,
            values,
            target,
            context,
        ))
        size(jacobian) == (length(target), length(coefficients)) ||
            throw(DimensionMismatch("target Jacobian does not match the EOF model"))
        return jacobian
    end
    jacobian = Matrix{Float64}(undef, length(target), length(coefficients))
    for index in eachindex(coefficients)
        left = copy(coefficients)
        right = copy(coefficients)
        left[index] -= finite_difference
        right[index] += finite_difference
        jacobian[:, index] = (
            target_measure(problem, evidence, right) -
            target_measure(problem, evidence, left)
        ) / (2finite_difference)
    end
    jacobian
end

function kernel_matrix(kernel, left, right)
    [kernel_value(kernel, view(left, i, :), view(right, j, :))
     for i in axes(left, 1), j in axes(right, 1)]
end

function trajectory_kernel_statistics(problem, evidence, t, cache)
    kernel_history = get!(cache, :world_trajectory_kernels) do
        Dict{Int,Any}()
    end
    statistics = get!(kernel_history, t) do
        locations = trajectory_locations(problem, evidence, t)
        empirical = occupation_weights(problem.trajectory, t)
        empirical_kernel = kernel_matrix(evidence.kernel, locations, locations)
        cross_kernel = kernel_matrix(evidence.kernel, locations, problem.context.quadrature)
        (
            empirical_energy=dot(empirical, empirical_kernel, empirical),
            target_cross_mean=cross_kernel' * empirical,
        )
    end
    newest = maximum(keys(kernel_history))
    for old_t in collect(keys(kernel_history))
        old_t < newest - 1 && delete!(kernel_history, old_t)
    end
    statistics
end

function kernel_discrepancy(problem::WorldInferenceProblem,
                            evidence::DirectErgodicEvidence, t, coefficients, cache)
    t == 0 && return 0.0
    target_measure_history = get!(cache, :world_target_measures) do
        Dict{Int,Any}()
    end
    target_measures = get!(target_measure_history, t) do
        Dict{Tuple,Vector{Float64}}()
    end
    target = get!(target_measures, Tuple(coefficients)) do
        target_measure(problem, evidence, coefficients)
    end
    newest_target_time = maximum(keys(target_measure_history))
    for old_t in collect(keys(target_measure_history))
        old_t < newest_target_time - 1 && delete!(target_measure_history, old_t)
    end
    grid = problem.context.quadrature
    target_kernel = get!(cache, :world_target_kernel) do
        kernel_matrix(evidence.kernel, grid, grid)
    end
    trajectory_kernels = trajectory_kernel_statistics(problem, evidence, t, cache)
    max(
        trajectory_kernels.empirical_energy -
        2dot(trajectory_kernels.target_cross_mean, target) +
        dot(target, target_kernel, target),
        0.0,
    )
end

function trajectory_mean_reward(problem::WorldInferenceProblem,
                                evidence::DirectErgodicEvidence, t, coefficients)
    (t == 0 || isnothing(evidence.reward)) && return 0.0
    model = candidate_model(problem.context, coefficients)
    rewards = [evidence.reward(
        model,
        problem.trajectory.states[index],
        problem.trajectory.actions[index],
        problem.context,
    ) for index in 1:t]
    dot(occupation_weights(problem.trajectory, t), rewards)
end


function trajectory_mean_reward_gradient(
    problem::WorldInferenceProblem,
    evidence::DirectErgodicEvidence,
    t,
    coefficients;
    finite_difference=1e-4,
)
    (t == 0 || isnothing(evidence.reward)) && return zeros(length(coefficients))
    if !isnothing(evidence.reward_gradient)
        model = candidate_model(problem.context, coefficients)
        gradients = [Vector{Float64}(evidence.reward_gradient(
            model,
            problem.trajectory.states[index],
            problem.trajectory.actions[index],
            problem.context,
        )) for index in 1:t]
        return sum(
            weight .* gradient
            for (weight, gradient) in zip(occupation_weights(problem.trajectory, t), gradients)
        )
    end
    gradient = similar(coefficients, Float64)
    for index in eachindex(coefficients)
        left = copy(coefficients)
        right = copy(coefficients)
        left[index] -= finite_difference
        right[index] += finite_difference
        gradient[index] = (
            trajectory_mean_reward(problem, evidence, t, right) -
            trajectory_mean_reward(problem, evidence, t, left)
        ) / (2finite_difference)
    end
    gradient
end

maturity(config::WorldEnergyConfig, t) = t == 0 ? 0.0 :
    t^config.maturity_power /
    (t^config.maturity_power + config.maturity_half_time^config.maturity_power)

function evidence_mixture(config::WorldEnergyConfig, t)
    @match config.evaluation begin
        :mmd => (reward=0.0, discrepancy=1.0)
        :reward => (reward=1.0, discrepancy=0.0)
        :combined => begin
            t == 0 && return (reward=1.0, discrepancy=0.0)
            uncertainty = config.ucb_scale * sqrt(log1p(t) / t)
            reference = config.ucb_scale *
                sqrt(log1p(config.mixture_time) / config.mixture_time)
            (
                reward=uncertainty / (uncertainty + reference),
                discrepancy=reference / (uncertainty + reference),
            )
        end
        _ => error("unknown world-evidence evaluation: $(config.evaluation)")
    end
end

function world_energy(problem::WorldInferenceProblem, evidence::DirectErgodicEvidence,
                      t, coefficients, cache)
    mix = evidence_mixture(evidence.energy, t)
    discrepancy = mix.discrepancy == 0 ? 0.0 :
        kernel_discrepancy(problem, evidence, t, coefficients, cache)
    reward = mix.reward == 0 ? 0.0 :
        trajectory_mean_reward(problem, evidence, t, coefficients)
    b = maturity(evidence.energy, t)
    β = evidence.energy.β_max * b
    scaled_discrepancy = discrepancy / evidence.energy.discrepancy_scale
    scaled_reward = (reward - evidence.energy.reward_reference) / evidence.energy.reward_scale
    total = β * (mix.discrepancy * scaled_discrepancy - mix.reward * scaled_reward)
    WorldEnergy(
        total=total,
        discrepancy=discrepancy,
        mean_reward=reward,
        scaled_discrepancy=scaled_discrepancy,
        scaled_reward=scaled_reward,
        maturity=b,
        β=β,
        reward_weight=mix.reward,
        discrepancy_weight=mix.discrepancy,
    )
end

function world_energy_gradient(
    problem::WorldInferenceProblem,
    evidence::DirectErgodicEvidence,
    t,
    coefficients,
    cache;
    finite_difference=1e-4,
)
    t == 0 && return zeros(length(coefficients))
    mix = evidence_mixture(evidence.energy, t)
    discrepancy_gradient = if mix.discrepancy == 0
        zeros(length(coefficients))
    else
        target = target_measure(problem, evidence, coefficients)
        target_jacobian = target_measure_jacobian(
            problem, evidence, coefficients; finite_difference,
        )
        target_kernel = get!(cache, :world_target_kernel) do
            kernel_matrix(
                evidence.kernel,
                problem.context.quadrature,
                problem.context.quadrature,
            )
        end
        trajectory = trajectory_kernel_statistics(problem, evidence, t, cache)
        2target_jacobian' * (
            target_kernel * target - trajectory.target_cross_mean
        )
    end
    reward_gradient = mix.reward == 0 ? zeros(length(coefficients)) :
        trajectory_mean_reward_gradient(
            problem, evidence, t, coefficients; finite_difference,
        )
    β = evidence.energy.β_max * maturity(evidence.energy, t)
    β .* (
        mix.discrepancy / evidence.energy.discrepancy_scale .* discrepancy_gradient -
        mix.reward / evidence.energy.reward_scale .* reward_gradient
    )
end

function world_information(
    problem::WorldInferenceProblem,
    evidence::DirectErgodicEvidence,
    t,
    coefficients,
    prior_factor,
    cache;
    finite_difference=1e-4,
)
    dimension = length(coefficients)
    t == 0 && return zeros(dimension, dimension)
    mix = evidence_mixture(evidence.energy, t)
    discrepancy_information = if mix.discrepancy == 0
        zeros(dimension, dimension)
    else
        target_jacobian = target_measure_jacobian(
            problem, evidence, coefficients; finite_difference,
        ) * prior_factor
        target_kernel = get!(cache, :world_target_kernel) do
            kernel_matrix(
                evidence.kernel,
                problem.context.quadrature,
                problem.context.quadrature,
            )
        end
        2target_jacobian' * target_kernel * target_jacobian
    end
    reward_information = if mix.reward == 0
        zeros(dimension, dimension)
    else
        reward_gradient = prior_factor' * trajectory_mean_reward_gradient(
            problem, evidence, t, coefficients; finite_difference,
        )
        reward_gradient * reward_gradient'
    end
    β = evidence.energy.β_max * maturity(evidence.energy, t)
    information = β .* (
        mix.discrepancy / evidence.energy.discrepancy_scale .*
            discrepancy_information +
        mix.reward / evidence.energy.reward_scale .*
            reward_information
    )
    Matrix(Symmetric(information))
end
