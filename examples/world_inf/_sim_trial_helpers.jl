using Arrodes: TrajectoryObservation, kernel_discrepancy
using Arrodes: plot_world_trial_particles, plot_world_trial_recovery
using Arrodes: save_world_inference_visualizations
using Arrodes: target_measure, target_measure_mmd
import SCRIBE
import VulcanJ
using SCRIBE.ROMSTools: plot_roms_curl, wet_grid_locations

using LinearAlgebra: Diagonal, Symmetric, cholesky, dot, norm, tr
using Plots
using Statistics: mean

using Match: @match
using UnPack: @pack!, @unpack

function normalized_curl_target(field, weights, floor)
    density = abs.(field) .+ floor
    density ./ dot(weights, density)
end

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
        pfield = SCRIBE.reconstruct_eof_field(
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

function simulate_observable_trajectory(mission, scenario, score, coefficients; behavior_type=:ergodic)
    @match behavior_type begin
        :ergodic => begin
            return simulate_ergodic_behavior(mission, scenario, score, coefficients)
        end
        _ => begin 
            println("Unknown behavior type: $behavior_type. Defaulting to ergodic.")
            return simulate_ergodic_behavior(mission, scenario, score, coefficients)
        end
    end
end

function simulate_ergodic_behavior(mission, scenario, score, coefficients)
    @unpack context, planner_bandwidth = scenario
    @unpack samples, agent_speed, ergodic_iterations, dt, learning_rate,
            momentum, control_weight, boundary_weight,
            line_search_steps, line_search_decay = mission[:trajectory]

    density = target_measure(context, score.target, coefficients)
    sites = [Tuple(Float64.(row)) for row in eachrow(context.quadrature)]
    start = sites[argmin(sum(abs2, row) for row in eachrow(context.quadrature))]
    bounds = VulcanJ.coordinate_bounds(sites)
    planned_path, _, _, _ = VulcanJ.kernel_ergodic_trajectory(
        start, sites, density, bounds, samples-1;
        density_bandwidth=planner_bandwidth, kernel_bandwidth=planner_bandwidth,
        dt=dt, max_speed=agent_speed, optimizer_iters=ergodic_iterations,
        learning_rate=learning_rate, momentum=momentum, control_weight=control_weight, boundary_weight=boundary_weight,
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
        :elapsed_times => elapsed_observation_times(points, agent_speed)
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
            field = SCRIBE.reconstruct_eof_field(
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

        posterior_field = SCRIBE.reconstruct_eof_field(
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
        prior_field = SCRIBE.reconstruct_eof_field(
            model; coefficients=model.ϕ,
        )
        inferred_field = SCRIBE.reconstruct_eof_field(
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
         plot_title="Trial $(trial[:trial]) world-model recovery over elapsed time"
   )
end

function plot_ten_trial_rmse_histories(trials)
    ordered_trials = sort(trials; by=trial -> trial[:trial])
    panels = Any[]

    for trial in ordered_trials
        elapsed_times = trial[:elapsed_times]
        diagnostics = trial[:recovery_diagnostics]
        trial_number = trial[:trial]

        show_legend = trial_number == 1

        world_panel = ranked_posterior_series_plot(
            elapsed_times, diagnostics[:world_rmse],
            diagnostics[:posterior_world_rmse];
            title="Trial $trial_number: inferred field RMSE",
            ylabel="Weighted field RMSE",
            posterior_label="Posterior expected field",
            show_legend=show_legend
        )

        target_panel = ranked_posterior_series_plot(
            elapsed_times, diagnostics[:target_rmse],
            diagnostics[:posterior_target_rmse];
            title="Trial $trial_number: inferred target field RMSE",
            ylabel="Weighted target field RMSE",
            posterior_label="Posterior expected target",
            show_legend=show_legend
        )

        push!(panels, world_panel, target_panel)
    end

    return plot(
        panels...;
        layout=(5,4),
        size=(3600, 2500),
        titlefontsize=11,
        plot_titlefontsize=20,
        plot_title=(
            "Spatially weighted field and target-field recovery " *
            "across ten trials"
        )
    )
end

function save_world_trial_reconstructions(path, trials, roms, arrow_stride)
    panels = Any[]
    for trial in trials
        limit = max(
            maximum(trial[:truth_target_field]),
            maximum(trial[:inferred_target_field]),
            eps(Float64),
        )
        truth = plot_roms_curl(
            trial[:truth_target_field],
            trial[:flow_directions],
            roms;
            arrow_stride=2arrow_stride,
            title="T$(lpad(trial[:trial], 2, '0')) observed target density",
            limit,
            colorbar=false,
            magnitude=true,
            display_scale=1.0,
            colorbar_title="normalized target density",
        )
        plot!(
            truth,
            trial[:trajectory][:, 1],
            trial[:trajectory][:, 2];
            color=:red,
            linewidth=2.0,
            label=false,
        )
        push!(panels, truth)
        push!(panels, plot_roms_curl(
            trial[:inferred_target_field],
            trial[:flow_directions],
            roms;
            arrow_stride=2arrow_stride,
            title="T$(lpad(trial[:trial], 2, '0')) posterior predictive target",
            limit,
            colorbar=false,
            magnitude=true,
            display_scale=1.0,
            colorbar_title="normalized target density",
        ))
    end
    savefig(
        plot(
            panels...;
            layout=(5, 4),
            size=(3600, 2100),
            plot_title="Observed and posterior-predictive normalized target densities (equal-length flow directions)",
            plot_titlefontsize=20,
            titlefontsize=12,
        ),
        path,
    )
end

function save_results(mission, scenario, trials)
    output = normpath(joinpath(@__DIR__, mission[:rel_output_path]))
    mkpath(output)
    savefig(
        plot_world_trial_recovery(trials),
        joinpath(output, "recovery_across_ten_worlds.png"),
    )
    savefig(
        plot_world_trial_particles(
            trials,
            scenario[:context].model.ϕ,
            scenario[:context].prior_covariance,
        ),
        joinpath(output, "final_particle_locations.png"),
    )
    save_world_trial_reconstructions(
        joinpath(output, "ten_world_reconstructions.png"),
        trials,
        scenario[:roms],
        mission[:visualization][:arrow_stride],
    )
    savefig(
        plot_ten_trial_rmse_histories(trials),
        joinpath(output, "weighted_rmse_over_time_across_ten_trials.png"),
    )
    for trial in trials
        trial_output = joinpath(
            output,
            "trial_$(lpad(trial[:trial], 2, '0'))",
        )
        field_plot = (field, title, limit) -> plot_roms_curl(
            field,
            trial[:flow_directions],
            scenario[:roms];
            arrow_stride=mission[:visualization][:arrow_stride],
            title,
            limit,
            magnitude=true,
            display_scale=1.0,
            colorbar_title="normalized |curl| shape",
        )
        save_world_inference_visualizations(
            trial_output,
            trial[:problem],
            trial[:result],
            trial[:truth_coefficients],
            trial[:truth_field],
            trial[:trajectory],
            scenario[:context].model.ϕ,
            scenario[:context].prior_covariance,
            field_plot;
            diagnostics=trial[:recovery_diagnostics],
            frame_count=mission[:visualization][:animation_frames],
            fps=mission[:visualization][:fps],
        )
        savefig(
            plot_world_recovery_over_time(trial),
            joinpath(trial_output, "recovery_over_time.png"),
        )
        target_limit = max(
            maximum(trial[:truth_target_field]),
            maximum(trial[:inferred_target_field]),
            eps(Float64),
        )
        target_plot = (field, title) -> plot_roms_curl(
            field,
            trial[:flow_directions],
            scenario[:roms];
            arrow_stride=mission[:visualization][:arrow_stride],
            title,
            limit=target_limit,
            magnitude=true,
            display_scale=1.0,
            colorbar_title="normalized target density",
        )
        observed_target = target_plot(
            trial[:truth_target_field],
            "Observed normalized target density",
        )
        plot!(
            observed_target,
            trial[:trajectory][:, 1],
            trial[:trajectory][:, 2];
            color=:red,
            linewidth=2.5,
            label=false,
        )
        inferred_target = target_plot(
            trial[:inferred_target_field],
            "Posterior predictive target density",
        )
        savefig(
            plot(
                observed_target,
                inferred_target;
                layout=(1, 2),
                size=(2200, 620),
                titlefontsize=15,
            ),
            joinpath(trial_output, "target_posterior_comparison.png"),
        )
    end
    output
end
