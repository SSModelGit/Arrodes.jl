function world_inference_history(
    problem,
    result::EOFWorldInferenceResult,
    truth,
)
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

function coefficient_limit(
    result::EOFWorldInferenceResult,
    truth,
    prior_mean,
)
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
    result::EOFWorldInferenceResult,
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
    result::EOFWorldInferenceResult,
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

function world_field_limit(field_history, truth_field)
    max(
        maximum(abs, truth_field),
        maximum(abs, field_history),
        eps(Float64),
    )
end

function plot_world_field_comparison(
    field_history,
    truth_field,
    trajectory,
    field_plot,
    observation_count;
    limit=world_field_limit(field_history, truth_field),
)
    observed = field_plot(truth_field, "Observed-agent posterior mean", limit)
    path_end = min(observation_count, size(trajectory, 1))
    plot!(
        observed,
        trajectory[1:path_end, 1],
        trajectory[1:path_end, 2];
        color=:red,
        linewidth=2.5,
        label=false,
    )
    inferred_field = view(field_history, :, observation_count + 1)
    inferred = field_plot(
        inferred_field,
        "Ego inference after $observation_count locations",
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

function plot_world_posterior_comparison(
    result::EOFWorldInferenceResult,
    truth_field,
    trajectory,
    field_plot,
    observation_count;
    limit=nothing,
)
    field_history = SCRIBE.reconstruct_eof_field(
        result.model; coefficients=result.coefficient_means,
    )
    plot_world_field_comparison(
        field_history, truth_field, trajectory, field_plot, observation_count;
        limit=isnothing(limit) ? world_field_limit(field_history, truth_field) : limit,
    )
end

function plot_world_posterior_comparison(
    result::SOMWorldInferenceResult,
    field_history,
    truth_field,
    trajectory,
    field_plot,
    observation_count;
    limit=nothing,
)
    plot_world_field_comparison(
        field_history, truth_field, trajectory, field_plot, observation_count;
        limit=isnothing(limit) ? world_field_limit(field_history, truth_field) : limit,
    )
end

function save_world_field_animation(
    path,
    field_history,
    truth_field,
    trajectory,
    field_plot;
    frame_count=80,
    fps=8,
)
    horizon = size(field_history, 2) - 1
    timesteps = unique(round.(Int, range(
        1,
        horizon;
        length=min(frame_count, horizon),
    )))
    limit = world_field_limit(field_history, truth_field)
    animation = @animate for timestep in timesteps
        plot_world_field_comparison(
            field_history,
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

function save_world_posterior_animation(
    path,
    result::EOFWorldInferenceResult,
    truth_field,
    trajectory,
    field_plot;
    frame_count=80,
    fps=8,
)
    field_history = SCRIBE.reconstruct_eof_field(
        result.model; coefficients=result.coefficient_means,
    )
    save_world_field_animation(
        path, field_history, truth_field, trajectory, field_plot;
        frame_count, fps,
    )
end


function save_world_posterior_animation(
    path,
    result::SOMWorldInferenceResult,
    field_history,
    truth_field,
    trajectory,
    field_plot;
    frame_count=80,
    fps=8,
)
    save_world_field_animation(
        path, field_history, truth_field, trajectory, field_plot;
        frame_count, fps,
    )
end

function particle_projection(
    result::EOFWorldInferenceResult,
    truth,
    prior_mean,
    prior_covariance,
    som_coefficients=nothing,
)
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
    projection = Dict(
        :initial => coordinates(initial),
        :final => coordinates(final),
        :truth => [dot(d₁, truth_whitened), dot(d₂, truth_whitened)],
        :posterior => [dot(d₁, posterior), dot(d₂, posterior)],
    )
    if !isnothing(som_coefficients)
        projection[:som_vertices] = coordinates(whiten(som_coefficients))
    end
    projection
end

function plot_world_particle_distribution(
    result::EOFWorldInferenceResult,
    truth,
    prior_mean,
    prior_covariance,
    som_coefficients=nothing,
    som_probabilities=nothing,
)
    projection = particle_projection(
        result,
        truth,
        prior_mean,
        prior_covariance,
        som_coefficients,
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
        color=:purple,
        colorbar=false,
        markersize=3,
        alpha=0.55,
        label="final particles",
    )
    if haskey(projection, :som_vertices)
        vertices = projection[:som_vertices]
        vertex_sizes = isnothing(som_probabilities) ? 6 :
            4 .+ 8 .* sqrt.(som_probabilities ./ max(
                maximum(som_probabilities), eps(Float64),
            ))
        scatter!(
            panel,
            vertices[:, 1],
            vertices[:, 2];
            marker=:utriangle,
            color=:steelblue,
            markersize=vertex_sizes,
            alpha=0.9,
            markerstrokecolor=:navy,
            markerstrokewidth=1.0,
            label="SOM vertices",
        )
    end
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
    result::EOFWorldInferenceResult,
    truth,
    prior_mean,
    prior_covariance,
    diagnostics,
    som_coefficients=nothing,
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
        som_coefficients,
    )
    plot(ess, spread, recovery, particles; layout=(2, 2), size=(1600, 1100))
end

function save_world_inference_visualizations(
    output,
    problem,
    result::EOFWorldInferenceResult,
    truth,
    truth_field,
    trajectory,
    prior_mean,
    prior_covariance,
    field_plot;
    diagnostics,
    som_coefficients=nothing,
    frame_count=80,
    fps=8,
    animate=true
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
            som_coefficients,
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
            som_coefficients,
        ),
        joinpath(output, "particle_health.png"),
    )

    if animate
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
    end
    output
end

function plot_som_probabilities(
    result::SOMWorldInferenceResult,
    observation_count,
    truth_vertex=nothing,
)
    probabilities = view(
        result.posterior_probabilities, :, observation_count + 1,
    )
    panel = bar(
        result.model.data[:vertex_ids], probabilities;
        color=:steelblue,
        label="",
        legend=false,
        ylim=(0.0, 1.05maximum(result.posterior_probabilities)),
        xlabel="SOM vertex ID",
        ylabel="posterior probability",
        title="SOM posterior after $observation_count locations",
        size=(1300, 650),
        left_margin=8Plots.mm,
        bottom_margin=7Plots.mm,
    )
    if !isnothing(truth_vertex)
        vertex_id = result.model.data[:vertex_ids][truth_vertex]
        vline!(
            panel, [vertex_id];
            color=:firebrick,
            linewidth=2.5,
            label="generating vertex",
            legend=:topright,
        )
    end
    panel
end

function save_som_probability_animation(
    path,
    result::SOMWorldInferenceResult,
    truth_vertex=nothing;
    frame_count=80,
    fps=8,
)
    horizon = size(result.posterior_probabilities, 2) - 1
    observation_counts = unique(round.(Int, range(
        0, horizon; length=min(frame_count, horizon + 1),
    )))
    animation = @animate for observation_count in observation_counts
        plot_som_probabilities(result, observation_count, truth_vertex)
    end
    mkpath(dirname(abspath(path)))
    gif(animation, path; fps)
end

function save_world_inference_visualizations(
    output,
    result::SOMWorldInferenceResult,
    field_history,
    truth_field,
    trajectory,
    field_plot;
    truth_vertex=nothing,
    frame_count=80,
    fps=8,
    animate=true,
)
    mkpath(output)
    horizon = size(result.posterior_probabilities, 2) - 1

    savefig(
        plot_world_posterior_comparison(
            result, field_history, truth_field, trajectory, field_plot, horizon,
        ),
        joinpath(output, "posterior_comparison.png"),
    )
    savefig(
        plot_som_probabilities(result, horizon, truth_vertex),
        joinpath(output, "vertex_probabilities.png"),
    )

    if animate
        save_world_posterior_animation(
            joinpath(output, "posterior_recovery.gif"),
            result, field_history, truth_field, trajectory, field_plot;
            frame_count, fps,
        )
        save_som_probability_animation(
            joinpath(output, "vertex_probabilities.gif"),
            result, truth_vertex; frame_count, fps,
        )
    end
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

function plot_world_trial_particles(
    trials, prior_mean, prior_covariance, som_coefficients=nothing,
)
    panels = [begin
        panel = plot_world_particle_distribution(
            trial[:result],
            trial[:truth_coefficients],
            prior_mean,
            prior_covariance,
            som_coefficients,
            haskey(trial, :som_result) ? view(
                trial[:som_result].posterior_probabilities, :, size(
                    trial[:som_result].posterior_probabilities, 2,
                ),
            ) : nothing,
        )
        title = ""
        if !isnothing(som_coefficients)
            title = "δSOM = $(round(trial[:som_hull_distance]; digits=2))"
        end
        plot!(
            panel;
            title,
            legend=false,
            xlabel=index > length(trials) - min(5, length(trials)) ?
                "Prior → belief" : "",
            ylabel=(index - 1) % 5 == 0 ?
                "Orthogonal direction" : "",
            titlefontsize=19,
            guidefontsize=17,
            tickfontsize=16,
            xticks=2ceil(Int, xlims(panel)[1] / 2):2:
                2floor(Int, xlims(panel)[2] / 2),
            legendfontsize=16,
            left_margin=(index - 1) % 5 == 0 ? 10Plots.mm : 4Plots.mm,
            right_margin=5Plots.mm,
            top_margin=5Plots.mm,
            bottom_margin=index > length(trials) - min(5, length(trials)) ?
                15Plots.mm : 4Plots.mm,
        )
        panel
    end for (index, trial) in enumerate(trials)]
    columns = min(5, length(panels))
    rows = ceil(Int, length(panels) / columns)
    legend_panel = plot(; framestyle=:none, axis=false, grid=false,
        legend=:top, legend_columns=3, legendfontsize=17)
    for (label, color, marker) in (
        ("EOF prior particles", :gray, :circle),
        ("EOF posterior particles", :purple, :circle),
        ("SOM states", :steelblue, :utriangle),
        ("Prior mean", :orange, :diamond),
        ("Generating belief", :red, :star5),
        ("EOF posterior mean", :black, :circle),
    )
        label == "SOM states" && isnothing(som_coefficients) && continue
        scatter!(legend_panel, [NaN], [NaN]; label, color, marker, markersize=5)
    end
    plot(
        legend_panel, panels...;
        layout=@layout([a{0.13h}; grid(rows, columns)]),
        size=(1600, 760), dpi=180,
    )
end

function world_result_comparison_panels(
    field_histories,
    labels,
    truth_field,
    trajectory,
    field_plot,
    timestep;
    limit=nothing,
    observed_title="Observed-agent world and trajectory",
)
    methods = sort(collect(keys(labels)); by=String)
    inferred_fields = [
        view(field_histories[name], :, timestep + 1)
        for name in methods
    ]
    limit = isnothing(limit) ? max(
        maximum(abs, truth_field),
        (maximum(abs, field) for field in inferred_fields)...,
        eps(Float64),
    ) : limit

    observed = field_plot(truth_field, observed_title, limit)
    path_end = min(timestep, size(trajectory, 1))
    plot!(
        observed,
        trajectory[1:path_end, 1],
        trajectory[1:path_end, 2];
        color=:red,
        linewidth=2.5,
        label=false,
    )

    inferred = [
        field_plot(field, labels[name], limit)
        for (name, field) in zip(methods, inferred_fields)
    ]

    return [observed; inferred]
end

function plot_world_result_comparison(
    field_histories,
    labels,
    truth_field,
    trajectory,
    field_plot,
    timestep;
    limit=nothing,
    observed_title="Observed-agent world and trajectory",
)
    panels = world_result_comparison_panels(
        field_histories, labels, truth_field, trajectory, field_plot, timestep;
        limit=limit, observed_title=observed_title
    )
    panel_count = length(panels)
    columns = panel_count <= 3 ? panel_count : ceil(Int, sqrt(panel_count))
    rows = ceil(Int, panel_count / columns)

    plot(
        panels...;
        layout=(rows, columns),
        size=(1100 * columns, 620 * rows),
        titlefontsize=15,
    )
end

function save_world_result_comparison_animation(
    path,
    field_histories,
    labels,
    truth_field,
    trajectory,
    field_plot;
    frame_count=80,
    fps=8,
    observed_title="Observed-agent world and trajectory",
)
    horizon = minimum(
        size(fields, 2) - 1 for fields in values(field_histories)
    )
    limit = max(
        maximum(abs, truth_field),
        (maximum(abs, field_histories[name]) for name in keys(labels))...,
        eps(Float64),
    )
    timesteps = unique(round.(Int, range(
        1,
        horizon;
        length=min(frame_count, horizon),
    )))
    animation = @animate for timestep in timesteps
        plot_world_result_comparison(
            field_histories,
            labels,
            truth_field,
            trajectory,
            field_plot,
            timestep,
            limit=limit,
            observed_title=observed_title,
        )
    end
    mkpath(dirname(abspath(path)))
    gif(animation, path; fps)
end
