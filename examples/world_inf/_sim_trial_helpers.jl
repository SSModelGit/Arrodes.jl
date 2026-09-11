using Arrodes: TrajectoryObservation, WorldInferenceContext,
    kernel_discrepancy, target_measure_mmd, world_result_comparison_panels
using LinearAlgebra: Diagonal, Symmetric, cholesky, dot, norm, tr
using Plots: plot, plot!, twinx
using SCRIBE: reconstruct_eof_field
using SCRIBE.ROMSTools: plot_roms_curl, wet_grid_locations
using Statistics: mean
using UnPack: @pack!, @unpack

import Plots
import VulcanJ

nonnegative_curl_target(field, floor) = abs.(field) .+ floor
nonnegative_curl_target(field, weights, floor) =
    nonnegative_curl_target(field, floor)

weighted_rmse(estimate, truth, weights) = let difference=estimate-truth
    sqrt(dot(difference, Diagonal(weights), difference) / sum(weights))
end

function posterior_target_field(
    model, particles::AbstractMatrix, particle_weights::AbstractVector,
    spatial_weights::AbstractVector, target_floor::Real,
)
    inferred_target = zeros(eltype(particle_weights), length(spatial_weights))
    for pindex in axes(particles, 2)
        pfield = reconstruct_eof_field(
            model; coefficients=view(particles, :, pindex),
        )
        inferred_target .+= particle_weights[pindex] .* nonnegative_curl_target(
            pfield, spatial_weights, target_floor,
        )
    end

    return inferred_target
end

function elapsed_observation_times(obs_locs, agent_speed)
    elapsed = zeros(Float64, size(obs_locs, 1))
    for ts in 2:size(obs_locs, 1)
        elapsed[ts] = elapsed[ts-1] + norm(obs_locs[ts, :] - obs_locs[ts-1, :]) / agent_speed
    end
    return elapsed
end

function simulate_observable_trajectory(
    mission, context::WorldInferenceContext, planner_bandwidth, density;
    behavior_type=:ergodic, start=nothing
)
    if behavior_type != :ergodic
        println("Unknown behavior type: $behavior_type. Defaulting to ergodic.")
    end
    simulate_ergodic_behavior(
        mission, context, planner_bandwidth, density; start,
    )
end

function simulate_ergodic_behavior(
    mission, context::WorldInferenceContext, planner_bandwidth::Real, density;
    start=nothing
)
    @unpack samples, agent_speed, ergodic_iterations, dt, learning_rate,
            momentum, control_weight, boundary_weight,
            line_search_steps, line_search_decay = mission[:trajectory]

    sites = [Tuple(Float64.(row)) for row in eachrow(context.quadrature)]
    if isnothing(start); start = sites[argmin(sum(abs2, row) for row in sites)]; end

    bounds = VulcanJ.coordinate_bounds(sites)
    planned_path, _, _, _ = VulcanJ.kernel_ergodic_trajectory(
        start, sites, density, bounds, samples-1;
        density_bandwidth=planner_bandwidth, kernel_bandwidth=planner_bandwidth,
        dt=dt, max_speed=agent_speed, optimizer_iters=ergodic_iterations,
        learning_rate=learning_rate, momentum=momentum,
        control_weight=control_weight, boundary_weight=boundary_weight,
        line_search_steps=line_search_steps, line_search_decay=line_search_decay
    )

    points = reduce(vcat, reshape(collect(point), 1, :) for point in planned_path)
    extended_points = vcat(points[1:1, :], points)
    return Dict(
        :observations => [
            TrajectoryObservation(
                state=extended_points[t, :],
                action=extended_points[t,:]-extended_points[t-1,:]
            ) for t in 2:size(extended_points, 1)
        ],
        :site_indices => [
            argmin(sum(abs2, point - site) for site in eachrow(context.quadrature))
            for point in eachrow(points)
        ],
        :elapsed_times => elapsed_observation_times(points, agent_speed),
    )
end

function coefficient_recovery(
    posterior, posterior_covariance, particles, weights,
    truth, prior, prior_covariance,
)
    prior_factor = cholesky(Symmetric(prior_covariance)).L
    prior_distance = norm(prior_factor \ (prior - truth))
    posterior_distance = norm(prior_factor \ (posterior - truth))
    posterior_difference = posterior - truth
    posterior_standardized_distance = sqrt(dot(
        posterior_difference,
        posterior_covariance \ posterior_difference,
    ))
    particle_distances = [
        norm(prior_factor \ (view(particles, :, index) - truth))
        for index in axes(particles, 2)
    ]
    representative_index = argmin([
        norm(prior_factor \ (view(particles, :, index) - posterior))
        for index in axes(particles, 2)
    ])
    representative_distance = particle_distances[representative_index]
    coefficient_error = posterior - truth
    relative_mode_error = abs.(coefficient_error) ./ max.(
        abs.(truth),
        abs.(posterior),
        sqrt(eps(Float64)),
    )
    dominant_modes = sortperm(abs.(truth); rev=true)[
        1:min(3, length(truth))
    ]
    Dict(
        :prior_distance => prior_distance,
        :posterior_distance => posterior_distance,
        :posterior_standardized_distance => posterior_standardized_distance,
        :recovery_ratio => posterior_distance /
            max(prior_distance, eps(Float64)),
        :representative_distance => representative_distance,
        :representative_recovery_ratio => representative_distance /
            max(prior_distance, eps(Float64)),
        :nearest_particle_distance => minimum(particle_distances),
        :weighted_particle_distance => dot(weights, particle_distances),
        :mass_within_one_sigma => dot(
            weights,
            particle_distances .<= 1.0,
        ),
        :sign_agreement => mean(sign.(posterior) .== sign.(truth)),
        :coefficient_error => coefficient_error,
        :relative_mode_error => relative_mode_error,
        :dominant_modes => dominant_modes,
        :dominant_mode_relative_error => relative_mode_error[dominant_modes],
    )
end

function construct_world_recovery_diagnostics_cache(
    problem, observed, target_floor, n_particles; top_count=10
)
    @unpack context, observations = problem
    @unpack model, prior_covariance = context
    @unpack field, coefficients = observed

    truth_field = field
    truth_coefficients = coefficients
    spatial_weights = model.params.decomposition.weights
    truth_target_field = nonnegative_curl_target(
        truth_field, spatial_weights, target_floor
    )

    p_count = min(top_count, n_particles)
    horizon = length(observations)

    # rmse/mmd plot diagnostics
    world_rmse = fill(NaN, p_count, horizon)
    target_rmse = fill(NaN, p_count, horizon)
    particle_mmd = fill(NaN, p_count, horizon)

    posterior_world_rmse = fill(NaN, horizon)
    posterior_target_rmse = fill(NaN, horizon)
    posterior_mmd = fill(NaN, horizon)

    # ess/p-health diagnostics
    behavioral_mmd = fill(NaN, horizon)
    ess_hist = fill(NaN, horizon+1)
    coeff_spread = fill(NaN, horizon)
    resampled = falses(horizon)

    mmd_cache = Dict{Symbol,Any}()
    callbacks = Dict{Symbol,Any}()

    diagnostics = Dict{Symbol, Any}(
        :callbacks => callbacks,

        :world_rmse => world_rmse,
        :target_rmse => target_rmse,
        :particle_mmd => particle_mmd,

        :posterior_world_rmse => posterior_world_rmse,
        :posterior_target_rmse => posterior_target_rmse,
        :posterior_mmd => posterior_mmd,

        :behavioral_mmd => behavioral_mmd,
        :ess_history => ess_hist,
        :coefficient_spread => coeff_spread,
        :resampled => resampled,
    )

    callbacks[:post_initialization] = filt_state -> begin
        ess_hist[1] = filt_state[:ess]
    end

    callbacks[:post_update] = filt_state -> begin
        @unpack timestep, particles, weights, coefficient_mean, ess = filt_state

        ess_hist[timestep + 1] = ess

        ranking = partialsortperm(weights, 1:p_count; rev=true)
        ranked_indices = view(ranking, 1:p_count)
        for (rank, pindex) in enumerate(ranked_indices)
            particle_coefficients = view(particles, :, pindex)
            field = reconstruct_eof_field(
                model; coefficients=particle_coefficients,
            )
            target_field = nonnegative_curl_target(
                field, spatial_weights, target_floor,
            )

            world_rmse[rank, timestep] = weighted_rmse(field, truth_field, spatial_weights)
            target_rmse[rank, timestep] = weighted_rmse(target_field, truth_target_field, spatial_weights)            
            particle_mmd[rank, timestep] = target_measure_mmd(
                problem, particle_coefficients, truth_coefficients, mmd_cache
            )
        end

        posterior_field = reconstruct_eof_field(
            model; coefficients=coefficient_mean,
        )
        posterior_world_rmse[timestep] = weighted_rmse(
            posterior_field, truth_field, spatial_weights
        )

        inferred_target = posterior_target_field(
            model, particles, weights, spatial_weights, target_floor
        )
        posterior_target_rmse[timestep] = weighted_rmse(
            inferred_target, truth_target_field, spatial_weights
        )
        posterior_mmd[timestep] = target_measure_mmd(
            problem, particles, weights,
            truth_coefficients, mmd_cache,
        )
    end

    callbacks[:post_resampling] = filt_state -> begin
        resampled[filt_state[:timestep]] = true
    end

    callbacks[:post_timestep] = filt_state -> begin
        @unpack timestep, coefficient_mean, coefficient_covariance = filt_state
        behavioral_mmd[timestep] = target_measure_mmd(
            problem, coefficient_mean, truth_coefficients, mmd_cache,
        )
        coeff_spread[timestep] = sqrt(tr(coefficient_covariance))
    end

    callbacks[:post_inference] = filt_state -> begin
        @unpack timestep, coefficient_mean, coefficient_covariance,
                particles, weights = filt_state

        diagnostics[:coefficient_recovery] = coefficient_recovery(
            coefficient_mean,
            coefficient_covariance,
            particles,
            weights,
            truth_coefficients,
            model.ϕ,
            prior_covariance,
        )

        inferred_coefficients = coefficient_mean
        prior_field = reconstruct_eof_field(
            model; coefficients=model.ϕ,
        )
        inferred_field = reconstruct_eof_field(
            model; coefficients=coefficient_mean,
        )
        prior_target_field = nonnegative_curl_target(
            prior_field, spatial_weights, target_floor
        )
        inferred_target_field = posterior_target_field(
            model, particles, weights, spatial_weights, target_floor
        )

        prior_field_rmse = weighted_rmse(
            prior_field, truth_field, spatial_weights
        )
        posterior_field_rmse = weighted_rmse(
            inferred_field, truth_field, spatial_weights
        )
        prior_target_field_rmse = weighted_rmse(
            prior_target_field, truth_target_field, spatial_weights
        )
        posterior_target_field_rmse = weighted_rmse(
            inferred_target_field, truth_target_field, spatial_weights
        )
        final_discrepancy = target_measure_mmd(
            problem, particles, weights, truth_coefficients, mmd_cache
        )
        posterior_predictive_trajectory_discrepancy = kernel_discrepancy(
            problem, timestep, particles, weights, mmd_cache,
        )

        @pack! diagnostics = inferred_coefficients,
            inferred_field,
            inferred_target_field,
            prior_field_rmse,
            posterior_field_rmse,
            prior_target_field_rmse,
            posterior_target_field_rmse,
            final_discrepancy,
            posterior_predictive_trajectory_discrepancy
    end

    return diagnostics
end

function ranked_series_plot(times, values; title, ylabel, show_legend)
    count = size(values, 1)
    colors = [
        "#000000", "#303030", "#484848", "#606060", "#787878",
        "#909090", "#a0a0a0", "#b0b0b0", "#c0c0c0", "#d0d0d0",
    ]
    panel = plot(; xlabel="Elapsed Time (s)", ylabel, title)

    for rank in count:-1:1
        alpha = count == 1 ? 1.0 : 1.0-0.7*(rank-1) / (count-1)
        plot!(
            panel, times, view(values, rank, :);
            color=colors[min(rank, length(colors))],
            linealpha=alpha,linewidth=(rank==1 ? 2.6 : 1.6),
            marker=:star5, markersize=(rank==1 ? 3.5 : 2.5),
            markeralpha=alpha, markerstrokewidth=0, label="Rank $rank",
            legend=show_legend ? :topright : false
        )
    end
    return panel
end

function ranked_posterior_series_plot(
    times, ranked_history, posterior_history;
    title, ylabel, posterior_label, show_legend
)
    panel = ranked_series_plot(
        times, ranked_history;
        title=title, ylabel=ylabel, show_legend=show_legend
    )
    plot!(
        panel, times, posterior_history;
        color=:red, linewidth=2.8,
        marker=:star5, markersize=5, markerstrokewidth=0,
        label=posterior_label, legend=show_legend ? :topright : false
    )

    return panel
end

function plot_trial_curl_field(
    field,
    title,
    limit,
    trial,
    scenario,
    mission;
    aggregate=false,
)
    plot_roms_curl(
        field,
        trial[:flow_directions],
        scenario[:roms];
        arrow_stride=(aggregate ? 2 : 1) *
            mission[:visualization][:arrow_stride],
        title,
        limit,
        colorbar=!aggregate,
        magnitude=true,
        display_scale=1.0,
        colorbar_title="normalized |curl| shape",
    )
end

trial_curl_plot(trial, scenario, mission; aggregate=false) =
    (field, title, limit) -> plot_trial_curl_field(
        field, title, limit, trial, scenario, mission; aggregate,
    )

function plot_world_method_rmse(trial; show_legend=true)
    @unpack elapsed_times, rmse_histories, prior_distance,
        recovery_diagnostics = trial
    trajectory_mmd = recovery_diagnostics[:behavioral_mmd]
    times = vcat(0.0, elapsed_times)

    panel = plot(
        times, rmse_histories[:EOF];
        color=:firebrick, linewidth=2.8, label="EOF posterior mean",
        xlabel="Elapsed time (s)", ylabel="Spatially weighted field RMSE",
        title="d = $(round(prior_distance, digits=2))",
        legend=show_legend ? :topright : false,
        left_margin=12Plots.mm, right_margin=18Plots.mm,
        top_margin=5Plots.mm, bottom_margin=10Plots.mm,
    )
    plot!(
        panel, times, rmse_histories[:SOM];
        color=:steelblue, linewidth=2.8, label="SOM posterior mean"
    )
    plot!(
        panel, [NaN], [NaN];
        color=:black, linestyle=:dash, linewidth=2.2,
        marker=:star5, markersize=3.5, markerstrokewidth=0,
        label="Behavioral target discrepancy"
    )
    plot!(
        twinx(panel), elapsed_times, trajectory_mmd;
        color=:black, linestyle=:dash, linewidth=2.2,
        marker=:star5, markersize=3.5, markerstrokewidth=0,
        label=false, legend=false, grid=false,
        ylabel="Target-measure MMD²", right_margin=18Plots.mm
    )

    return panel
end

function plot_ten_trial_rmse_histories(trials)
    panels = [
        plot_world_method_rmse(trial; show_legend=index==1)
        for (index, trial) in enumerate(trials)
    ]

    time_limit = maximum(last(trial[:elapsed_times]) for trial in trials)
    rmse_limit = 1.05 * maximum(
        maximum(history) for trial in trials for history in values(trial[:rmse_histories])
    )
    for panel in panels
        plot!(panel; xlims=(0.0, time_limit))
        plot!(panel[1]; ylims=(0.0, rmse_limit))
    end

    columns = min(2, length(trials))
    rows = ceil(Int, length(trials) / columns)

    return plot(
        panels...;
        layout=(rows, columns),
        size=(1000 * columns, 480 * rows + 80),
        plot_title="EOF and SOM field recovery across trials",
        plot_titlefontsize=20, titlefontsize=12, guidefontsize=10, tickfontsize=9,
    )
end

function plot_world_trial_reconstructions(
    trials, scenario, mission
)
    panels = Any[]

    for trial in trials
        timestep = minimum(
            size(fields, 2) - 1
            for fields in values(trial[:field_histories])
        )
        number = lpad(trial[:trial], 2, '0')
        trial_labels = Dict(
            :EOF => "T$number EOF",
            :SOM => "T$number SOM",
        )

        append!(
            panels,
            world_result_comparison_panels(
                trial[:field_histories], trial_labels,
                trial[:truth_field], trial[:trajectory],
                trial_curl_plot(trial, scenario, mission; aggregate=true),
                timestep;
                observed_title="T$number observed"
            )
        )
    end

    columns = length(trials) == 1 ? 3 : 6
    rows = ceil(Int, length(panels) / columns)
    source = first(trials)[:source] == :som_vertices ?
        "SOM-vertex generating worlds" : "ROMS snapshot generating worlds"

    return plot(
        panels...;
        layout=(rows, columns),
        size=(600 * columns, 400 * rows + 100),
        plot_title="$source: observed / EOF / SOM",
        plot_titlefontsize=20, titlefontsize=12,
        left_margin=5Plots.mm, right_margin=5Plots.mm,
        top_margin=4Plots.mm, bottom_margin=6Plots.mm
    )
end

function plot_world_recovery_over_time(trial)
    elapsed_times = trial[:elapsed_times]
    diagnostics = trial[:recovery_diagnostics]

    world_panel = ranked_posterior_series_plot(
        elapsed_times, diagnostics[:world_rmse],
        diagnostics[:posterior_world_rmse];
        title="World-model average RMSE",
        ylabel="Spatially weighted average field RMSE",
        posterior_label="Posterior expected field",
        show_legend=true
    )
    target_panel = ranked_posterior_series_plot(
        elapsed_times, diagnostics[:target_rmse],
        diagnostics[:posterior_target_rmse];
        title="Inferred target-field weighted RMSE",
        ylabel="Spatially weighted target field RMSE",
        posterior_label="Posterior expected target",
        show_legend=true
    )

    behavioral_mmd_panel = plot(
        elapsed_times, diagnostics[:behavioral_mmd];
        color=:black, linewidth=2.6, marker=:star5, markersize=3.5,
        markerstrokewidth=0, label=false, xlabel="Elapsed Time (s)",
        ylabel="target-measure MMD²",
        title="Target behavior's kernel discrepancy measure over time"
    )
    particle_mmd_panel = ranked_posterior_series_plot(
        elapsed_times,
        diagnostics[:particle_mmd], diagnostics[:posterior_mmd];
        title="Kernel discrepancy measure of inferred behavior over time",
        ylabel="Target-measure MMD²",
        posterior_label="Posterior mixture",
        show_legend=true
    )

    plot(world_panel, target_panel,
         behavioral_mmd_panel, particle_mmd_panel;
         layout=(2, 2), size=(1700, 1150), titlefontsize=16,
         left_margin=12Plots.mm, right_margin=5Plots.mm,
         top_margin=5Plots.mm, bottom_margin=8Plots.mm,
         plot_title="Trial $(trial[:trial]) world-model recovery over elapsed time"
   )
end
