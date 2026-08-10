using Arrodes
using LinearAlgebra
using Parameters: @with_kw_noshow
using Plots
using POMDPs
using Random
using SCRIBE
using Statistics
using VulcanJ

const SCRIBE_ROMS_EXAMPLE = joinpath(
    pkgdir(SCRIBE), "examples", "eof-climate-models",
)
include(joinpath(SCRIBE_ROMS_EXAMPLE, "_roms_data.jl"))

const ROMS_ARCHIVE = normpath(joinpath(
    @__DIR__, "..", "..", "bigdata", "rams_head_model_output",
    "stjohn_hourly_5m_velocity_ramhead_v2.mat",
))
const ROMS_TEMPORAL_STRIDE = 3
const ROMS_TRAINING_FRACTION = 0.8
const EGO_SNAPSHOT = 3030
const OBSERVED_SNAPSHOT = 5302
const TRAJECTORY_SAMPLES = 100
const FILTER_PARTICLES = 192
const OUTPUT_DIR = joinpath(@__DIR__, "res", "scribe_world_inference")

raw_snapshot_index(snapshot) = 1 + (snapshot - 1) * ROMS_TEMPORAL_STRIDE

best_fit_coefficients(params, snapshot) =
    params.decomposition.modes' * (
        params.decomposition.weights .* (snapshot - params.decomposition.mean)
    )

function model_at_coefficients(params, coefficients, role)
    role_metadata = copy(params.metadata)
    role_metadata["arrodes_role"] = String(role)
    role_params = SCRIBE.EOFClimateModelParameters(
        params.decomposition;
        process_covariance=params.Q,
        locations=params.locations,
        ϕ₀=coefficients,
        prior_covariance=params.P₀,
        interpolation=params.interpolation,
        interpolation_neighbors=params.interpolation_neighbors,
        metadata=role_metadata,
    )
    SCRIBE.initialize_SCRIBEModel_from_parameters(role_params)
end

function farthest_point_rows(locations, count)
    count = min(count, size(locations, 1))
    minimums = vec(minimum(locations; dims=1))
    spans = max.(vec(maximum(locations; dims=1)) - minimums, eps(Float64))
    normalized = (locations .- minimums') ./ spans'
    center = vec(mean(normalized; dims=1))
    first_row = argmin([sum(abs2, row - center) for row in eachrow(normalized)])
    selected = [first_row]
    distance = [sum(abs2, row - view(normalized, first_row, :))
                for row in eachrow(normalized)]
    while length(selected) < count
        row = argmax(distance)
        push!(selected, row)
        distance = min.(distance, [
            sum(abs2, point - view(normalized, row, :))
            for point in eachrow(normalized)
        ])
    end
    selected
end

function pairwise_distances(X)
    [hypot(X[i, 1] - X[j, 1], X[i, 2] - X[j, 2])
     for i in axes(X, 1), j in axes(X, 1)]
end

@with_kw_noshow struct ROMSErgodicMDP <: MDP{Matrix{Float64},Symbol}
    sites::Vector{Matrix{Float64}}
    start::Matrix{Float64}
    step::Float64
    lower::Vector{Float64}
    upper::Vector{Float64}
end

@with_kw_noshow struct ROMSTargetModel
    reward_at::Dict{Tuple{Float64,Float64},Float64}
end

POMDPs.actions(::ROMSErgodicMDP) = (:east, :west, :north, :south)
POMDPs.actions(mdp::ROMSErgodicMDP, state) = POMDPs.actions(mdp)
POMDPs.discount(::ROMSErgodicMDP) = 1.0
POMDPs.isterminal(::ROMSErgodicMDP, state) = false

function POMDPs.gen(mdp::ROMSErgodicMDP, state::Matrix{Float64}, action::Symbol, rng)
    delta = action === :east ? [mdp.step, 0.0] :
        action === :west ? [-mdp.step, 0.0] :
        action === :north ? [0.0, mdp.step] : [0.0, -mdp.step]
    location = clamp.(vec(state) + delta, mdp.lower, mdp.upper)
    (sp=reshape(location, 1, :), r=0.0)
end

VulcanJ.cellsites(mdp::ROMSErgodicMDP) = mdp.sites

function VulcanJ.expected_information_gain(
    mdp::ROMSErgodicMDP,
    model::ROMSTargetModel,
    state,
    quadrature_order::Integer,
)
    model.reward_at[Tuple(Float64.(vec(state)))]
end

function nearest_site_indices(points, sites)
    [argmin([sum(abs2, point - site) for site in eachrow(sites)])
     for point in eachrow(points)]
end

function vulcan_ergodic_trajectory(
    locations,
    target_reward;
    horizon=TRAJECTORY_SAMPLES - 1,
    bandwidth,
    rng=MersenneTwister(41),
)
    sites = [reshape(Vector{Float64}(row), 1, :) for row in eachrow(locations)]
    lower = vec(minimum(locations; dims=1))
    upper = vec(maximum(locations; dims=1))
    span = maximum(upper - lower)
    start_row = argmin(vec(sum((locations .- lower') .^ 2; dims=2)))
    mdp = ROMSErgodicMDP(
        sites=sites,
        start=copy(sites[start_row]),
        step=span / 25,
        lower=lower,
        upper=upper,
    )
    model = ROMSTargetModel(reward_at=Dict(
        Tuple(vec(site)) => reward
        for (site, reward) in zip(sites, target_reward)
    ))
    plan = VulcanJ.one_shot_ergodic_planner(
        mdp,
        model,
        horizon;
        initial_state=mdp.start,
        rng,
        backend=:kernel,
        density_bandwidth=bandwidth,
        kernel_bandwidth=bandwidth,
        max_speed=span / 16,
        optimizer_iters=180,
        learning_rate=0.22,
        control_weight=2e-3,
        boundary_weight=20.0,
    )
    continuous = reduce(vcat, plan.states)
    site_indices = nearest_site_indices(continuous, locations)
    observations = map(axes(continuous, 1)) do timestep
        state = vec(continuous[timestep, :])
        previous = timestep == 1 ? state : vec(continuous[timestep - 1, :])
        TrajectoryObservation(
            state=state,
            action=state - previous,
        )
    end
    (; observations, site_indices, plan)
end

function direct_world_evidence(
    context,
    bandwidth,
    temperature,
    discrepancy_scale,
    evaluation,
)
    DirectErgodicEvidence(
        location=state -> state,
        reward=(model, state, action, context) -> only(
            SCRIBE.predict_SCRIBEModel(model, reshape(Float64.(state), 1, :)),
        ),
        reward_gradient=(model, state, action, context) -> vec(
            SCRIBE.eof_basis_at(model, reshape(Float64.(state), 1, :)),
        ),
        importance=(model, X, values) -> begin
            logits = values ./ temperature
            exp.(logits .- maximum(logits))
        end,
        target_jacobian=(model, X, values, target, context) -> begin
            basis = SCRIBE.eof_basis_at(model, X)
            centered_basis = basis .- sum(reshape(target, :, 1) .* basis; dims=1)
            reshape(target, :, 1) .* centered_basis ./ temperature
        end,
        kernel=GaussianDiscrepancyKernel(bandwidth=bandwidth),
        energy=WorldEnergyConfig(
            discrepancy_scale=discrepancy_scale,
            reward_scale=temperature,
            reward_reference=0.0,
            β_max=7.0,
            maturity_half_time=18.0,
            maturity_power=2.0,
            mixture_time=12.0,
            evaluation=evaluation,
        ),
    )
end

function target_mmd_scale(context, evidence, coefficients)
    problem = WorldInferenceProblem(context=context, evidence=evidence)
    masses = [target_measure(problem, evidence, ϕ) for ϕ in coefficients]
    kernel = Arrodes.Inference.kernel_matrix(
        evidence.kernel, context.quadrature, context.quadrature,
    )
    discrepancies = Float64[]
    for i in 2:length(masses)
        for j in 1:(i - 1)
            difference = masses[i] - masses[j]
            push!(discrepancies, dot(difference, kernel * difference))
        end
    end
    max(median(discrepancies), sqrt(eps(Float64)))
end

function completed_summary(state, timestep)
    only(filter(
        summary -> summary isa WorldPosteriorSummary &&
            summary.stage.observation == timestep && summary.stage.λ == 1.0,
        state.summaries,
    ))
end

function completed_ancestry(state, timestep)
    last(filter(
        ancestry -> ancestry.stage.observation == timestep &&
            ancestry.stage.λ == 1.0,
        state.ancestry,
    ))
end

function run_recorded_filter(
    problem,
    observations,
    config,
    observed_field,
    observed_coefficients,
    label,
)
    state = initialize_smc(problem, config)
    basis = SCRIBE.eof_modes(problem.context.model)
    field_metric = (basis' * basis) ./ size(basis, 1)
    history = NamedTuple[]
    for (timestep, observation) in enumerate(observations)
        update_smc!(state, observation, config)
        summary = completed_summary(state, timestep)
        particles = state.cloud.particles
        coefficients = reduce(hcat, (particle.value for particle in particles))
        differences = coefficients .- observed_coefficients
        particle_rmse = vec(sqrt.(max.(
            sum(differences .* (field_metric * differences); dims=1),
            0.0,
        )))
        target_scores = [logtarget(
            problem, state.cloud.stage, particle.value, state.cache,
        ) for particle in particles]
        ancestry = completed_ancestry(state, timestep)
        unique_particles = length(unique(
            Tuple(particle.value) for particle in particles
        ))
        push!(history, (
            timestep,
            summary,
            ess=effective_sample_size(state.cloud),
            unique_particles,
            resampling_parents=length(unique(ancestry.resampling_parents)),
            coefficient_spread=sqrt(tr(summary.coefficient_covariance)),
            posterior_rmse=sqrt(mean(abs2, summary.map_mean - observed_field)),
            highest_target_rmse=particle_rmse[argmax(target_scores)],
            oracle_best_rmse=minimum(particle_rmse),
        ))
        if timestep == 1 || timestep % 25 == 0 || timestep == length(observations)
            record = last(history)
            println(
                "$label inference $timestep/$(length(observations)): ",
                "ESS=$(round(record.ess; digits=1)), ",
                "unique=$(record.unique_particles), ",
                "field RMSE=$(round(record.posterior_rmse; digits=5))",
            )
        end
    end
    SMCResult(state=state), history
end

function field_grid(values, roms)
    grid = fill(NaN, roms.grid_shape...)
    grid[roms.wet_mask] = values
    permutedims(grid)
end

function wet_grid_locations(roms, wet_rows)
    wet_cells = findall(reshape(roms.wet_mask, roms.grid_shape...))
    hcat(
        getindex.(wet_cells[wet_rows], 1),
        getindex.(wet_cells[wet_rows], 2),
    )
end

function save_evaluation_comparison_animation(
    path,
    histories,
    observed_field,
    trajectory_grid,
    roms,
    ego_snapshot,
    observed_snapshot;
    frame_count=80,
    fps=10,
)
    combined_history = histories.combined
    frames = unique(round.(Int, range(
        1, length(combined_history); length=min(frame_count, length(combined_history)),
    )))
    color_limit = max(
        maximum(abs, observed_field),
        maximum(
            maximum(abs, record.summary.map_mean)
            for history in histories for record in history
        ),
    )
    animation = @animate for timestep in frames
        observed_panel = heatmap(
            field_grid(observed_field, roms);
            color=:balance,
            clims=(-color_limit, color_limit),
            colorbar=false,
            aspect_ratio=:equal,
            axis=false,
            title="Observed agent EOF posterior mean\nheld-out ROMS snapshot $observed_snapshot",
        )
        path_prefix = trajectory_grid[1:timestep, :]
        plot!(
            observed_panel,
            path_prefix[:, 1],
            path_prefix[:, 2];
            color=:black,
            linewidth=1.5,
            marker=:circle,
            markersize=2.2,
            markerstrokewidth=0,
            label=false,
        )
        scatter!(
            observed_panel,
            [path_prefix[end, 1]],
            [path_prefix[end, 2]];
            color=:yellow,
            markerstrokecolor=:black,
            markersize=6,
            label=false,
        )
        function inference_panel(record, title)
            heatmap(
                field_grid(record.summary.map_mean, roms);
                color=:balance,
                clims=(-color_limit, color_limit),
                colorbar=false,
                aspect_ratio=:equal,
                axis=false,
                title="$title after $timestep locations\n" *
                    "field RMSE=$(round(record.posterior_rmse; digits=4))",
            )
        end
        combined_panel = inference_panel(
            histories.combined[timestep], "Combined reward + MMD",
        )
        mmd_panel = inference_panel(
            histories.mmd[timestep], "MMD-only",
        )
        reward_panel = inference_panel(
            histories.reward[timestep], "Reward-only",
        )
        plot(
            observed_panel,
            combined_panel,
            mmd_panel,
            reward_panel;
            layout=(2, 2),
            size=(1200, 980),
            plot_title="Ego prior snapshot $ego_snapshot → observed-agent snapshot $observed_snapshot",
        )
    end
    gif(animation, path; fps)
end

function coefficient_colors(values)
    [value >= 0 ? :steelblue : :firebrick for value in values]
end

function save_coefficient_animation(
    path,
    history,
    observed_coefficients;
    frame_count=80,
    fps=10,
)
    frames = unique(round.(Int, range(1, length(history); length=frame_count)))
    rank = length(observed_coefficients)
    limit = maximum(vcat(
        abs.(observed_coefficients),
        [abs.(record.summary.coefficient_mean) for record in history]...,
    ))
    animation = @animate for timestep in frames
        record = history[timestep]
        posterior_mean = record.summary.coefficient_mean
        observed_panel = bar(
            1:rank,
            observed_coefficients;
            color=coefficient_colors(observed_coefficients),
            legend=false,
            ylim=(-1.15limit, 1.15limit),
            ylabel="coefficient value",
            title="Observed agent EOF coefficients",
        )
        inferred_panel = bar(
            1:rank,
            posterior_mean;
            color=coefficient_colors(posterior_mean),
            legend=false,
            ylim=(-1.15limit, 1.15limit),
            xlabel="EOF mode",
            ylabel="coefficient value",
            title="Ego posterior after $timestep locations",
        )
        plot(
            observed_panel,
            inferred_panel;
            layout=(2, 1),
            size=(1200, 760),
            plot_title="Observed versus inferred EOF coefficient profile",
        )
    end
    gif(animation, path; fps)
end

function save_particle_health(
    path,
    result,
    history,
    ego_coefficients,
    observed_coefficients,
)
    timesteps = getfield.(history, :timestep)
    particle_count = length(result.state.cloud.particles)
    activity = plot(
        timesteps,
        getfield.(history, :ess);
        label="ESS",
        linewidth=2,
        ylabel="particle count",
        title="Active particle population",
        ylim=(0, particle_count),
    )
    plot!(activity, timesteps, getfield.(history, :unique_particles);
          label="unique coefficient vectors", linewidth=2)
    plot!(activity, timesteps, getfield.(history, :resampling_parents);
          label="distinct resampling parents", linewidth=2)

    spread = plot(
        timesteps,
        getfield.(history, :coefficient_spread);
        label=false,
        linewidth=2,
        xlabel="observed locations",
        ylabel="sqrt(tr(Σ))",
        title="Posterior coefficient spread",
    )

    rmse = plot(
        timesteps,
        getfield.(history, :posterior_rmse);
        label="posterior mean",
        linewidth=2,
        ylabel="field RMSE",
        title="Reconstruction error to observed-agent model",
    )
    plot!(rmse, timesteps, getfield.(history, :highest_target_rmse);
          label="highest-target particle", linewidth=2)
    plot!(rmse, timesteps, getfield.(history, :oracle_best_rmse);
          label="oracle-best particle", linewidth=2, linestyle=:dash)

    particles = result.state.cloud.particles
    coefficient_matrix = reduce(hcat, (particle.value for particle in particles))
    final_mean = last(history).summary.coefficient_mean
    covariance = last(history).summary.coefficient_covariance
    decomposition = eigen(Symmetric(covariance))
    directions = decomposition.vectors[:, end:-1:end - 1]
    projection = directions' * (coefficient_matrix .- final_mean)
    ego_projection = directions' * (ego_coefficients - final_mean)
    observed_projection = directions' * (observed_coefficients - final_mean)
    target_scores = [logtarget(
        result.state.problem,
        result.state.cloud.stage,
        particle.value,
        result.state.cache,
    ) for particle in particles]
    geometry = scatter(
        projection[1, :],
        projection[2, :];
        marker_z=target_scores,
        color=:viridis,
        markersize=4,
        label="particles",
        xlabel="posterior PC 1",
        ylabel="posterior PC 2",
        title="Final particle spread",
    )
    scatter!(geometry, [ego_projection[1]], [ego_projection[2]];
             marker=:diamond, markersize=9, color=:orange, label="ego prior")
    scatter!(geometry, [observed_projection[1]], [observed_projection[2]];
             marker=:star5, markersize=10, color=:red, label="observed agent")

    sorted_scores = sort(target_scores; rev=true)
    score_panel = plot(
        eachindex(sorted_scores),
        sorted_scores .- maximum(sorted_scores);
        label=false,
        linewidth=2,
        xlabel="particle rank",
        ylabel="log target below best",
        title="Final particle target health",
    )

    coefficient_error = last(history).summary.coefficient_mean - observed_coefficients
    coefficient_panel = bar(
        eachindex(coefficient_error),
        coefficient_error;
        color=coefficient_colors(coefficient_error),
        legend=false,
        xlabel="EOF mode",
        ylabel="posterior mean - truth",
        title="Final coefficient error",
    )

    savefig(plot(
        activity,
        spread,
        rmse,
        score_panel,
        geometry,
        coefficient_panel;
        layout=(3, 2),
        size=(1300, 1350),
        plot_title="World-inference particle health",
    ), path)
end

mkpath(OUTPUT_DIR)
for previous_output in readdir(OUTPUT_DIR; join=true)
    rm(previous_output; recursive=true)
end

println("Loading ROMS velocity snapshots with SCRIBE's eof-climate-models helper ...")
archive_data = read_roms_velocity(:u; path=ROMS_ARCHIVE)
roms = prepare_roms_velocity(
    archive_data;
    temporal_stride=ROMS_TEMPORAL_STRIDE,
)
archive_data = nothing
GC.gc()
n_snapshots = size(roms.data, 2)
n_training = floor(Int, ROMS_TRAINING_FRACTION * n_snapshots)
training_data = roms.data[:, 1:n_training]

println("Fitting the shared SCRIBE EOF space exactly as in construct_roms_eof_model.jl ...")
base_model = initialize_eof_climate_model(
    training_data;
    locations=roms.locations,
    variance_fraction=0.995,
    max_rank=160,
    algorithm=:randomized,
    oversample=16,
    power_iterations=2,
    rng=MersenneTwister(12),
    process_covariance=1e-4,
    interpolation=:nearest,
    metadata=Dict(
        "source" => basename(ROMS_ARCHIVE),
        "component" => "u",
        "temporal_stride" => ROMS_TEMPORAL_STRIDE,
        "training_snapshots" => n_training,
        "process_variance" => 1e-4,
        "grid_shape" => collect(roms.grid_shape),
        "wet_mask" => Int8.(roms.wet_mask),
    ),
)
params = base_model.params
ego_snapshot = EGO_SNAPSHOT
observed_snapshot = OBSERVED_SNAPSHOT
max(ego_snapshot, observed_snapshot) <= n_snapshots ||
    error("the SCRIBE snapshot scenario is unavailable in the prepared ROMS archive")
ego_coefficients = best_fit_coefficients(params, roms.data[:, ego_snapshot])
observed_coefficients = best_fit_coefficients(params, roms.data[:, observed_snapshot])
ego_model = model_at_coefficients(params, ego_coefficients, :ego)
observed_model = model_at_coefficients(params, observed_coefficients, :observed)
ego_information = SCRIBE.init_agent_info(ego_model.params)

quadrature_rows = farthest_point_rows(roms.locations, 196)
quadrature = roms.locations[quadrature_rows, :]
context = scribe_world_context(
    ego_model,
    ego_information;
    quadrature,
    quadrature_weights=params.decomposition.weights[quadrature_rows],
)
distance = pairwise_distances(quadrature)
nearest_distance = [minimum(filter(>(0.0), distance[row, :]))
                    for row in axes(distance, 1)]
kernel_bandwidth = 2.5median(nearest_distance)
field_temperature = std(training_data)

template_evidence = direct_world_evidence(
    context, kernel_bandwidth, field_temperature, 1.0, :mmd,
)
calibration_ids = unique(round.(Int, range(
    n_training + 1,
    n_snapshots;
    length=9,
)))
calibration_coefficients = [
    best_fit_coefficients(params, roms.data[:, snapshot])
    for snapshot in calibration_ids
]
discrepancy_scale = target_mmd_scale(
    context, template_evidence, calibration_coefficients,
)
evidence = (
    combined=direct_world_evidence(
        context, kernel_bandwidth, field_temperature, discrepancy_scale, :combined,
    ),
    mmd=direct_world_evidence(
        context, kernel_bandwidth, field_temperature, discrepancy_scale, :mmd,
    ),
    reward=direct_world_evidence(
        context, kernel_bandwidth, field_temperature, discrepancy_scale, :reward,
    ),
)

observed_quadrature_field = candidate_field(context, observed_coefficients)
target_logits = observed_quadrature_field ./ field_temperature
target_reward = exp.(target_logits .- maximum(target_logits))
ergodic = vulcan_ergodic_trajectory(
    quadrature,
    target_reward;
    horizon=TRAJECTORY_SAMPLES - 1,
    bandwidth=kernel_bandwidth,
)
length(ergodic.observations) == TRAJECTORY_SAMPLES || error(
    "VulcanJ returned $(length(ergodic.observations)) states; expected $TRAJECTORY_SAMPLES",
)

config = SMCConfig(
    n_particles=FILTER_PARTICLES,
    ess_threshold=0.55,
    scheduler=OneStagePerObservation(),
    kernel=PriorPCNKernel(ρ=0.975),
    seed=0x524f_4d53_574f_524c,
)
observed_field = SCRIBE.reconstruct_eof_field(observed_model)
problems = (
    combined=WorldInferenceProblem(context=context, evidence=evidence.combined),
    mmd=WorldInferenceProblem(context=context, evidence=evidence.mmd),
    reward=WorldInferenceProblem(context=context, evidence=evidence.reward),
)
combined_result, combined_history = run_recorded_filter(
    problems.combined,
    ergodic.observations,
    config,
    observed_field,
    observed_coefficients,
    "Combined",
)
mmd_result, mmd_history = run_recorded_filter(
    problems.mmd,
    ergodic.observations,
    config,
    observed_field,
    observed_coefficients,
    "MMD-only",
)
reward_result, reward_history = run_recorded_filter(
    problems.reward,
    ergodic.observations,
    config,
    observed_field,
    observed_coefficients,
    "Reward-only",
)
results = (
    combined=combined_result,
    mmd=mmd_result,
    reward=reward_result,
)
histories = (
    combined=combined_history,
    mmd=mmd_history,
    reward=reward_history,
)

trajectory_wet_rows = quadrature_rows[ergodic.site_indices]
trajectory_grid = wet_grid_locations(roms, trajectory_wet_rows)

field_animation = joinpath(OUTPUT_DIR, "world_evidence_comparison.gif")
coefficient_animation = joinpath(
    OUTPUT_DIR, "observed_and_inferred_coefficients.gif",
)
health_plot = joinpath(OUTPUT_DIR, "particle_health.png")
save_evaluation_comparison_animation(
    field_animation,
    histories,
    observed_field,
    trajectory_grid,
    roms,
    ego_snapshot,
    observed_snapshot,
)
save_coefficient_animation(
    coefficient_animation,
    histories.combined,
    observed_coefficients,
)
save_particle_health(
    health_plot,
    results.combined,
    histories.combined,
    ego_coefficients,
    observed_coefficients,
)

summary_path = joinpath(OUTPUT_DIR, "run_summary.txt")
open(summary_path, "w") do io
    final = last(histories.combined)
    println(io, "ROMS archive: $ROMS_ARCHIVE")
    println(io, "Wet spatial cells: $(size(roms.data, 1))")
    println(io, "Prepared 3-hour snapshots: $n_snapshots")
    println(io, "Training snapshots: $n_training")
    println(io, "Retained EOF modes: $(length(ego_coefficients))")
    println(io, "Explained anomaly variance: $(params.decomposition.explained_variance)")
    println(io, "Ego snapshot: $ego_snapshot (raw index $(raw_snapshot_index(ego_snapshot)))")
    println(io, "Observed snapshot: $observed_snapshot (raw index $(raw_snapshot_index(observed_snapshot)))")
    println(io, "VulcanJ trajectory samples: $(length(ergodic.observations))")
    println(io, "World evidence modes: combined, MMD-only, reward-only")
    println(io, "Initial ego-to-observed field RMSE: $(sqrt(mean(abs2, SCRIBE.reconstruct_eof_field(ego_model) - observed_field)))")
    println(io, "Final combined field RMSE: $(final.posterior_rmse)")
    println(io, "Final MMD-only field RMSE: $(last(histories.mmd).posterior_rmse)")
    println(io, "Final reward-only field RMSE: $(last(histories.reward).posterior_rmse)")
    println(io, "Final highest-target particle RMSE: $(final.highest_target_rmse)")
    println(io, "Final oracle-best particle RMSE: $(final.oracle_best_rmse)")
    println(io, "Final ESS: $(final.ess) / $(config.n_particles)")
    println(io, "Final unique particles: $(final.unique_particles) / $(config.n_particles)")
    println(io, "Final distinct resampling parents: $(final.resampling_parents) / $(config.n_particles)")
    println(io, "VulcanJ final kernel ergodic metric: $(last(ergodic.plan.kernel_metric_history))")
end

println("Retained $(length(ego_coefficients)) EOF modes explaining ",
        round(100params.decomposition.explained_variance; digits=2), "% of anomaly variance.")
println("Ego snapshot $ego_snapshot; observed-agent snapshot $observed_snapshot.")
println("Final combined field RMSE: ", last(histories.combined).posterior_rmse)
println("Final MMD-only field RMSE: ", last(histories.mmd).posterior_rmse)
println("Final reward-only field RMSE: ", last(histories.reward).posterior_rmse)
println("Final active unique particles: ", last(histories.combined).unique_particles,
        "/", config.n_particles)
println("Saved $field_animation")
println("Saved $coefficient_animation")
println("Saved $health_plot")
println("Saved $summary_path")
