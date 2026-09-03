function world_inference_history(problem, result, truth)
    cache = Dict{Symbol,Any}()
    [Dict(
        :timestep => timestep,
        :covariance => result.coefficient_covariances[timestep + 1],
        :discrepancy => target_measure_mmd(
            problem,
            view(result.coefficient_means, :, timestep + 1),
            truth,
            cache,
        ),
        :ess => result.ess_history[timestep + 1],
    ) for timestep in 1:(size(result.coefficient_means, 2) - 1)]
end

coefficient_colors(values) =
    [value >= 0 ? :steelblue : :firebrick for value in values]

function coefficient_limit(result, truth, prior_mean)
    1.15max(
        maximum(abs, truth),
        maximum(abs, prior_mean),
        maximum(abs, view(result.coefficient_means, :, size(
            result.coefficient_means,
            2,
        ))),
        eps(Float64),
    )
end

function plot_world_coefficient_comparison(
    result,
    truth,
    prior_mean,
    timestep;
    limit=coefficient_limit(result, truth, prior_mean),
)
    estimate = result.coefficient_means[:, timestep + 1]
    observed = bar(
        eachindex(truth),
        truth;
        color=coefficient_colors(truth),
        legend=false,
        ylim=(-limit, limit),
        title="Observed-agent EOF coefficients",
    )
    inferred = bar(
        eachindex(estimate),
        estimate;
        color=coefficient_colors(estimate),
        legend=false,
        ylim=(-limit, limit),
        xlabel="EOF mode",
        title="Inferred coefficients after $timestep locations",
    )
    plot(observed, inferred; layout=(2, 1), size=(1300, 850))
end

function save_world_coefficient_animation(
    path,
    result,
    truth,
    prior_mean;
    frame_count=80,
    fps=8,
)
    horizon = size(result.coefficient_means, 2) - 1
    timesteps = unique(round.(Int, range(
        1,
        horizon;
        length=min(frame_count, horizon),
    )))
    limit = coefficient_limit(result, truth, prior_mean)
    animation = @animate for timestep in timesteps
        plot_world_coefficient_comparison(
            result,
            truth,
            prior_mean,
            timestep;
            limit,
        )
    end
    mkpath(dirname(abspath(path)))
    gif(animation, path; fps)
end

function world_field_limit(result, truth_field)
    prior_field = SCRIBE.reconstruct_eof_field(
        result.model;
        coefficients=view(result.coefficient_means, :, 1),
    )
    final_field = SCRIBE.reconstruct_eof_field(
        result.model;
        coefficients=view(result.coefficient_means, :, size(
            result.coefficient_means,
            2,
        )),
    )
    max(
        maximum(abs, truth_field),
        maximum(abs, prior_field),
        maximum(abs, final_field),
        eps(Float64),
    )
end

function plot_world_posterior_comparison(
    result,
    truth_field,
    trajectory,
    field_plot,
    timestep;
    limit=world_field_limit(result, truth_field),
)
    observed = field_plot(truth_field, "Observed-agent posterior mean", limit)
    path_end = min(timestep, size(trajectory, 1))
    plot!(
        observed,
        trajectory[1:path_end, 1],
        trajectory[1:path_end, 2];
        color=:red,
        linewidth=2.5,
        label=false,
    )
    inferred_field = SCRIBE.reconstruct_eof_field(
        result.model;
        coefficients=view(result.coefficient_means, :, timestep + 1),
    )
    inferred = field_plot(
        inferred_field,
        "Ego inference after $timestep locations",
        limit,
    )
    plot(
        observed,
        inferred;
        layout=(1, 2),
        size=(2200, 620),
        titlefontsize=15,
    )
end

function save_world_posterior_animation(
    path,
    result,
    truth_field,
    trajectory,
    field_plot;
    frame_count=80,
    fps=8,
)
    horizon = size(result.coefficient_means, 2) - 1
    timesteps = unique(round.(Int, range(
        1,
        horizon;
        length=min(frame_count, horizon),
    )))
    limit = world_field_limit(result, truth_field)
    animation = @animate for timestep in timesteps
        plot_world_posterior_comparison(
            result,
            truth_field,
            trajectory,
            field_plot,
            timestep;
            limit,
        )
    end
    mkpath(dirname(abspath(path)))
    gif(animation, path; fps)
end

function particle_projection(result, truth, prior_mean, prior_covariance)
    factor = cholesky(Symmetric(prior_covariance)).L
    whiten(vector) = factor \ (vector - prior_mean)
    whiten(particles::AbstractMatrix) = factor \ (
        particles .- reshape(prior_mean, :, 1)
    )
    truth_whitened = whiten(truth)
    d₁ = truth_whitened ./ max(norm(truth_whitened), eps(Float64))
    initial = whiten(result.initial_particles)
    final = whiten(result.final_particles)
    posterior = final * result.final_weights
    centered = final .- reshape(posterior, :, 1)
    covariance = centered * Diagonal(result.final_weights) * centered'
    projector = I - d₁ * d₁'
    orthogonal_covariance = Symmetric(projector * covariance * projector)
    d₂ = eigen(orthogonal_covariance).vectors[:, end]
    if norm(projector * d₂) <= sqrt(eps(Float64))
        basis = zeros(length(d₁))
        basis[argmin(abs.(d₁))] = 1.0
        d₂ = projector * basis
    end
    d₂ ./= max(norm(d₂), eps(Float64))
    dot(d₂, posterior) < 0 && (d₂ .*= -1)
    coordinates(particles) = hcat(vec(d₁' * particles), vec(d₂' * particles))
    Dict(
        :initial => coordinates(initial),
        :final => coordinates(final),
        :truth => [dot(d₁, truth_whitened), dot(d₂, truth_whitened)],
        :posterior => [dot(d₁, posterior), dot(d₂, posterior)],
    )
end

function plot_world_particle_distribution(
    result,
    truth,
    prior_mean,
    prior_covariance,
)
    projection = particle_projection(
        result,
        truth,
        prior_mean,
        prior_covariance,
    )
    initial = projection[:initial]
    final = projection[:final]
    panel = scatter(
        initial[:, 1],
        initial[:, 2];
        color=:gray,
        markersize=2.5,
        alpha=0.25,
        label="initial particles",
        xlabel="ego→observed coordinate (prior σ)",
        ylabel="dominant orthogonal coordinate (prior σ)",
        title="EOF particles in a prior-whitened linear plane",
        size=(1200, 800),
    )
    scatter!(
        panel,
        final[:, 1],
        final[:, 2];
        marker_z=result.final_weights,
        color=:viridis,
        colorbar=false,
        markersize=3,
        alpha=0.55,
        label="final particles",
    )
    scatter!(panel, [0.0], [0.0]; marker=:diamond, color=:orange,
             markersize=8, label="ego prior")
    scatter!(panel, [projection[:truth][1]], [projection[:truth][2]];
             marker=:star5, color=:red, markersize=9, label="observed world")
    scatter!(panel, [projection[:posterior][1]], [projection[:posterior][2]];
             marker=:circle, color=:black, markersize=7,
             label="posterior particle mean")
    plot!(panel; legend=:topright, legendfontsize=7)
    panel
end

function plot_world_particle_health(
    problem,
    result,
    truth,
    prior_mean,
    prior_covariance,
    diagnostics,
)
    horizon = length(diagnostics[:behavioral_mmd])
    timesteps = 1:horizon

    ess = plot(
        timesteps,
        view(diagnostics[:ess_history], 2:(horizon + 1));
        linewidth=2,
        label=false,
        xlabel="observed locations",
        ylabel="ESS",
        title="Effective particle count",
    )
    spread = plot(
        timesteps,
        diagnostics[:coefficient_spread];
        linewidth=2,
        label=false,
        xlabel="observed locations",
        ylabel="sqrt(tr(Σ))",
        title="Posterior coefficient spread",
    )
    recovery = plot(
        timesteps,
        diagnostics[:behavioral_mmd];
        linewidth=2,
        label=false,
        xlabel="observed locations",
        ylabel="target-measure MMD²",
        title="Behavioral target discrepancy",
    )
    particles = plot_world_particle_distribution(
        result,
        truth,
        prior_mean,
        prior_covariance,
    )
    plot(ess, spread, recovery, particles; layout=(2, 2), size=(1600, 1100))
end

function save_world_inference_visualizations(
    output,
    problem,
    result,
    truth,
    truth_field,
    trajectory,
    prior_mean,
    prior_covariance,
    field_plot;
    diagnostics,
    frame_count=80,
    fps=8,
)
    mkpath(output)
    horizon = size(result.coefficient_means, 2) - 1
    savefig(
        plot_world_posterior_comparison(
            result,
            truth_field,
            trajectory,
            field_plot,
            horizon,
        ),
        joinpath(output, "posterior_comparison.png"),
    )
    savefig(
        plot_world_coefficient_comparison(
            result,
            truth,
            prior_mean,
            horizon,
        ),
        joinpath(output, "coefficient_comparison.png"),
    )
    savefig(
        plot_world_particle_distribution(
            result,
            truth,
            prior_mean,
            prior_covariance,
        ),
        joinpath(output, "particle_distribution.png"),
    )
    savefig(
        plot_world_particle_health(
            problem,
            result,
            truth,
            prior_mean,
            prior_covariance,
            diagnostics,
        ),
        joinpath(output, "particle_health.png"),
    )
    save_world_posterior_animation(
        joinpath(output, "posterior_recovery.gif"),
        result,
        truth_field,
        trajectory,
        field_plot;
        frame_count,
        fps,
    )
    save_world_coefficient_animation(
        joinpath(output, "coefficient_recovery.gif"),
        result,
        truth,
        prior_mean;
        frame_count,
        fps,
    )
    output
end

function plot_world_trial_recovery(trials)
    let ordered = trials[sortperm(getindex.(trials, :prior_distance))],
        distance = getindex.(ordered, :prior_distance),
        trajectory_discrepancies = getindex.(ordered, :trajectory_discrepancy),
        recovery_diagnostics = getindex.(ordered, :recovery_diagnostics),
        prior_occupancy_mmd = getindex.(trajectory_discrepancies, :prior),
        posterior_occupancy_mmd = getindex.(trajectory_discrepancies, :posterior_predictive),
        prior_field_rmse = getindex.(recovery_diagnostics, :prior_field_rmse),
        posterior_field_rmse = getindex.(recovery_diagnostics, :posterior_field_rmse),
        prior_target_field_rmse = getindex.(recovery_diagnostics, :prior_target_field_rmse),
        posterior_target_field_rmse = getindex.(recovery_diagnostics, :posterior_target_field_rmse)

        function comparison_panel(prior_values, posterior_values; title, ylabel, prior_label, posterior_label)
            panel = plot(distance, prior_values;
                         color=:black, marker=:star5, markersize=7, markerstrokewidth=0, linewidth=2.8,
                         label=prior_label, xlabel="Prior-whitened Mahalanobis distance", ylabel=ylabel, title=title,
                         legend=:topright, titlefontsize=18, guidefontsize=16, tickfontsize=14, legendfontsize=13)
            plot!(panel, distance, posterior_values;
                  color=:red, marker=:star5, markersize=7, markerstrokewidth=0, linewidth=2.8, label=posterior_label)
            return panel
        end

        occupancy_panel = comparison_panel(prior_occupancy_mmd, posterior_occupancy_mmd;
                                          title="Observed-agent occupancy fit", ylabel="Ergodic occupancy metric: MMD²",
                                          prior_label="Ego-agent prior", posterior_label="Posterior-predictive fit")
        field_panel = comparison_panel(prior_field_rmse, posterior_field_rmse;
                                       title="True field recovery", ylabel="Spatially weighted field RMSE",
                                       prior_label="Ego-agent prior belief field",
                                       posterior_label="Recovered posterior field")
        target_panel = comparison_panel(prior_target_field_rmse, posterior_target_field_rmse;
                                       title="Target field recovery", ylabel="Spatially weighted target RMSE",
                                       prior_label="Ego-agent prior on target field",
                                       posterior_label="Recovered posterior on target field")
        return plot(occupancy_panel, field_panel, target_panel;
                    layout=(1, 3), size=(3600, 1200), dpi=180,
                    left_margin=24Plots.mm, right_margin=8Plots.mm,
                    top_margin=10Plots.mm, bottom_margin=24Plots.mm,
                    plot_title="Inference recovery across trials of increasing Mahalanobis distance " *
                               "between world belief of ego-agent and observed agent belief",
                    plot_titlefontsize=24)
    end
end

function plot_world_trial_particles(trials, prior_mean, prior_covariance)
    panels = [begin
        panel = plot_world_particle_distribution(
            trial[:result],
            trial[:truth_coefficients],
            prior_mean,
            prior_covariance,
        )
        plot!(
            panel;
            title="trial $(trial[:trial])",
            legend=trial[:trial] == 1 ? :topright : false,
            xlabel=trial[:trial] > 5 ? "ego→observed (prior σ)" : "",
            ylabel=trial[:trial] in (1, 6) ? "orthogonal (prior σ)" : "",
            titlefontsize=10,
            guidefontsize=8,
            tickfontsize=7,
        )
        panel
    end for trial in trials]
    plot(
        panels...;
        layout=(2, 5),
        size=(3000, 1300),
        plot_title="Initial and final particles in each trial's prior-whitened plane",
        plot_titlefontsize=14,
    )
end

function plot_world_result_comparison(
    results,
    ordering,
    truth_field,
    trajectory,
    field_plot,
    timestep;
    limit=nothing,
)
    inferred_fields = [SCRIBE.reconstruct_eof_field(
        results[name].model;
        coefficients=view(results[name].coefficient_means, :, timestep + 1),
    ) for name in ordering]
    limit = isnothing(limit) ? max(
        maximum(abs, truth_field),
        (maximum(abs, field) for field in inferred_fields)...,
        eps(Float64),
    ) : limit
    observed = field_plot(truth_field, "Observed-agent world and trajectory", limit)
    path_end = min(timestep, size(trajectory, 1))
    plot!(
        observed,
        trajectory[1:path_end, 1],
        trajectory[1:path_end, 2];
        color=:red,
        linewidth=2.5,
        label=false,
    )
    panels = [field_plot(field, String(name), limit)
              for (name, field) in zip(ordering, inferred_fields)]
    columns = 2
    rows = ceil(Int, (length(panels) + 1) / columns)
    plot(observed, panels...; layout=(rows, columns), size=(2200, 540rows))
end

function save_world_result_comparison_animation(
    path,
    results,
    ordering,
    truth_field,
    trajectory,
    field_plot;
    frame_count=80,
    fps=8,
)
    horizon = minimum(size(result.coefficient_means, 2) - 1
                      for result in values(results))
    final_fields = [SCRIBE.reconstruct_eof_field(
        results[name].model;
        coefficients=view(results[name].coefficient_means, :, horizon + 1),
    ) for name in ordering]
    limit = max(
        maximum(abs, truth_field),
        (maximum(abs, field) for field in final_fields)...,
        eps(Float64),
    )
    timesteps = unique(round.(Int, range(
        1,
        horizon;
        length=min(frame_count, horizon),
    )))
    animation = @animate for timestep in timesteps
        plot_world_result_comparison(
            results,
            ordering,
            truth_field,
            trajectory,
            field_plot,
            timestep,
            limit=limit,
        )
    end
    mkpath(dirname(abspath(path)))
    gif(animation, path; fps)
end
