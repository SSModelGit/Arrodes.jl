using Arrodes: WorldInferenceProblem, calibrate_discrepancy_scale,
    eof_field_score, eof_target_field, infer_world,
    plot_world_result_comparison, plot_world_trial_particles,
    save_world_inference_visualizations, save_world_result_comparison_animation,
    world_inference_context
using JSON: parsefile
using LinearAlgebra: BLAS, Symmetric, cholesky, norm
using Match: @match
using Plots: plot, plot!, savefig
using Random: MersenneTwister, randperm
using SCRIBE: SOMModel, eof_coefficients, eof_model_at_coefficients,
    eof_prior_covariance, reconstruct_eof_field
using SCRIBE.ROMSTools: fit_roms_eof, prepare_roms_curl_shape,
    read_roms_flow_directions, velocity_curl, wet_grid_locations
using Statistics: mean, std
using UnPack: @unpack

import Plots

include("_sim_trial_helpers.jl")

BLAS.set_num_threads(1)

function som_vertex_world(vertex, model::SOMModel, roms)
    @unpack x, y = model.data
    u = view(vertex, :, :, 1)
    v = view(vertex, :, :, 2)
    curl = velocity_curl(u, v, x, y)
    wet = roms[:wet_mask]

    field = abs.(vec(curl)[wet])
    field ./= mean(field)

    u_wet = vec(u)[wet]
    v_wet = vec(v)[wet]
    speed = hypot.(u_wet, v_wet)
    directions = hcat(u_wet, v_wet)
    moving = speed .> eps(Float64)
    directions[moving, :] ./= reshape(speed[moving], :, 1)
    directions[.!moving, :] .= 0.0

    Dict(
        :field => field,
        :flow_directions => directions,
    )
end

som_candidate_worlds(model::SOMModel, roms) = [
    som_vertex_world(vertex, model, roms)
    for vertex in model.data[:vertices]
]

function select_quadrature_rows(locations, count)
    center = mean(locations; dims=1)
    selected = [argmin(vec(sum(abs2, locations .- center; dims=2)))]
    distances = vec(sum(
        abs2,
        locations .- locations[first(selected), :]';
        dims=2,
    ))
    while length(selected) < min(count, size(locations, 1))
        row = argmax(distances)
        push!(selected, row)
        distances = min.(distances, vec(sum(
            abs2,
            locations .- locations[row, :]';
            dims=2,
        )))
    end
    selected
end

function gaussian_vertex_prior(coefficients, mean, covariance)
    factor = cholesky(Symmetric(covariance)).L
    whitened = factor \ (coefficients .- reshape(mean, :, 1))
    log_probabilities = -0.5 .* vec(sum(abs2, whitened; dims=1))
    probabilities = exp.(log_probabilities .- maximum(log_probabilities))
    probabilities ./ sum(probabilities)
end

function world_score(target, mission, scenario, discrepancy_scale)
    eof_field_score(
        target;
        kernel_bandwidth=scenario[:kernel_bandwidth],
        discrepancy_scale,
        β_max=mission[:filter][:beta_max],
        maturity_half_time=mission[:filter][:maturity_half_time],
        maturity_power=mission[:filter][:maturity_power],
        location=state -> (
            Float64.(state) .- scenario[:metric_minima]
        ) ./ scenario[:metric_spans],
    )
end

function prepare_mission(mission)
    println("Preparing $(mission[:name]) from ROMS and SOM world models ...")
    @unpack temporal_stride, spatial_stride, training_fraction,
        eof_rank, eof_oversample, eof_power_iterations,
        quadrature_count, calibration_worlds = mission[:roms]

    archive = normpath(joinpath(@__DIR__, mission[:roms_archive]))
    roms = prepare_roms_curl_shape(
        archive;
        temporal_stride,
        spatial_stride,
    )
    fitted = fit_roms_eof(
        roms;
        training_fraction,
        rank=eof_rank,
        oversample=eof_oversample,
        power_iterations=eof_power_iterations,
    )
    params = fitted[:model].params
    ego_coefficients = zeros(length(params.decomposition.eigenvalues))
    eof_model = eof_model_at_coefficients(params, ego_coefficients)
    prior_covariance =
        mission[:observed_world_prior][:archive_covariance_multiplier] .*
        eof_prior_covariance(eof_model)

    rows = select_quadrature_rows(roms[:locations], quadrature_count)
    quadrature = roms[:locations][rows, :]
    minima = vec(minimum(quadrature; dims=1))
    spans = max.(
        vec(maximum(quadrature; dims=1)) - minima,
        eps(Float64),
    )
    kernel_locations = (quadrature .- minima') ./ spans'
    spatial_weights = params.decomposition.weights
    quadrature_weights = spatial_weights[rows]
    eof_context = world_inference_context(
        eof_model;
        prior_covariance,
        quadrature,
        kernel_locations,
        quadrature_weights,
    )

    som_path = normpath(joinpath(@__DIR__, mission[:som_model_data]))
    som_model = SOMModel(som_path)
    som_worlds = som_candidate_worlds(som_model, roms)
    som_fields = hcat(getindex.(som_worlds, :field)...)
    som_coefficients = eof_coefficients(params, som_fields)
    som_prior = gaussian_vertex_prior(
        som_coefficients,
        ego_coefficients,
        prior_covariance,
    )
    som_context = world_inference_context(
        som_model;
        prior_probabilities=som_prior,
        quadrature,
        kernel_locations,
        quadrature_weights,
        fields=som_fields[rows, :],
    )

    kernel_bandwidth = Float64(mission[:target][:kernel_bandwidth])
    target_floor = mission[:target][:floor_fraction] * fitted[:field_scale]
    scenario = Dict(
        :archive => archive,
        :roms => roms,
        :eof_model => eof_model,
        :eof_context => eof_context,
        :som_model => som_model,
        :som_context => som_context,
        :som_worlds => som_worlds,
        :som_fields => som_fields,
        :som_coefficients => som_coefficients,
        :validation_start => fitted[:validation_start],
        :quadrature_rows => rows,
        :kernel_bandwidth => kernel_bandwidth,
        :planner_bandwidth => kernel_bandwidth * maximum(spans),
        :metric_minima => minima,
        :metric_spans => spans,
        :target_floor => target_floor,
        :spatial_weights => spatial_weights,
        :prior_covariance => prior_covariance,
    )

    target = eof_target_field(
        link=Symbol(mission[:target][:link]),
        floor=target_floor,
        name=Symbol(mission[:target][:name]),
    )
    template = world_score(target, mission, scenario, 1.0)
    calibration_ids = unique(round.(Int, range(
        fitted[:n_training] + 1,
        fitted[:calibration_end];
        length=calibration_worlds,
    )))
    calibration_coefficients = [
        eof_coefficients(params, view(roms[:data], :, snapshot))
        for snapshot in calibration_ids
    ]
    eof_scale = calibrate_discrepancy_scale(
        eof_context,
        template,
        calibration_coefficients,
    )
    som_scale = calibrate_discrepancy_scale(som_context, template)
    scores = Dict(
        :EOF => world_score(target, mission, scenario, eof_scale),
        :SOM => world_score(target, mission, scenario, som_scale),
    )
    println(
        "Offline MMD² units: EOF=$(round(eof_scale; sigdigits=4)), " *
        "SOM=$(round(som_scale; sigdigits=4))",
    )

    settings = mission[:filter][:proposal]
    proposal = @match settings[:mechanism] begin
        "gauss_newton" => Dict{Symbol,Any}(
            :mechanism => :gauss_newton,
            :covariance_scale => Float64(settings[:covariance_scale]),
            :optimizer_steps => Int(settings[:optimizer_steps]),
        )
        "random_walk" => Dict{Symbol,Any}(
            :mechanism => :random_walk,
            :scale => Float64(settings[:scale]),
        )
    end

    Dict(
        :mission => mission,
        :scenario => scenario,
        :scores => scores,
        :proposal => proposal,
    )
end

function distinguish_worlds_by_distance(mission, scenario, candidates)
    @unpack worlds_per_distance, snapshot_gap,
        prior_sigma_distances = mission[:trials]
    factor = cholesky(Symmetric(scenario[:prior_covariance])).L
    prior = scenario[:eof_model].ϕ
    distances = [
        norm(factor \ (candidate[:coefficients] - prior))
        for candidate in candidates
    ]
    available = trues(length(candidates))
    worlds = Dict{Symbol,Any}[]

    for target in prior_sigma_distances
        for _ in 1:worlds_per_distance
            remaining = findall(available)
            selected = remaining[
                argmin(abs.(distances[remaining] .- target))
            ]
            push!(worlds, merge(candidates[selected], Dict(
                :requested_distance => Float64(target),
                :actual_distance => distances[selected],
            )))
            available[selected] = false

            if haskey(candidates[selected], :snapshot)
                snapshot = candidates[selected][:snapshot]
                for index in remaining
                    if abs(candidates[index][:snapshot] - snapshot) < snapshot_gap
                        available[index] = false
                    end
                end
            end
        end
    end

    worlds
end

function som_trial_worlds(mission, scenario)
    model = scenario[:som_model]
    candidates = [
        Dict(
            :source => :som_vertices,
            :vertex => vertex,
            :field => scenario[:som_worlds][vertex][:field],
            :flow_directions => scenario[:som_worlds][vertex][:flow_directions],
            :coefficients => Vector{Float64}(
                view(scenario[:som_coefficients], :, vertex),
            ),
        )
        for vertex in eachindex(model.data[:vertices])
    ]
    distinguish_worlds_by_distance(mission, scenario, candidates)
end

function roms_trial_worlds(mission, scenario)
    model = scenario[:eof_model]
    roms = scenario[:roms]
    snapshots = collect(scenario[:validation_start]:size(roms[:data], 2))
    coefficients = eof_coefficients(
        model.params,
        view(roms[:data], :, snapshots),
    )
    candidates = [
        Dict(
            :source => :roms_snapshots,
            :snapshot => snapshots[index],
            :field => Vector{Float64}(view(roms[:data], :, snapshots[index])),
            :coefficients => Vector{Float64}(view(coefficients, :, index)),
        )
        for index in eachindex(snapshots)
    ]
    selected = distinguish_worlds_by_distance(mission, scenario, candidates)
    directions = read_roms_flow_directions(
        scenario[:archive],
        roms,
        getindex.(selected, :snapshot),
    )
    [
        merge(world, Dict(
            :flow_directions => directions[world[:snapshot]],
        ))
        for world in selected
    ]
end

function field_rmse_history(field_history, truth, weights)
    [
        weighted_rmse(view(field_history, :, column), truth, weights)
        for column in axes(field_history, 2)
    ]
end

function run_trial(
    mission, scenario, scores, proposal, world, trial, start;
    keep_history=false,
)
    context = scenario[:eof_context]
    target = scores[:EOF].target
    density = target.density(
        nothing, context.quadrature, world[:field][scenario[:quadrature_rows]], context
    )
    density = Float64.(density) .* context.quadrature_weights
    density ./= sum(density)
    trajectory = simulate_observable_trajectory(
        mission, context, scenario[:planner_bandwidth], density;
        start,
    )
    observations = trajectory[:observations]
    eof_problem = WorldInferenceProblem(
        context=scenario[:eof_context],
        score=scores[:EOF],
        observations=observations,
    )
    som_problem = WorldInferenceProblem(
        context=scenario[:som_context],
        score=scores[:SOM],
        observations=observations,
    )
    observed = Dict(
        :coefficients => world[:coefficients],
        :field => world[:field],
    )
    diagnostics = keep_history ? construct_world_recovery_diagnostics_cache(
        eof_problem, observed, scenario[:target_floor],
        mission[:filter][:particles]; top_count=10,
    ) : nothing
    seed = UInt64(mission[:trials][:seed]) + UInt64(2trial)
    eof_result = infer_world(
        eof_problem;
        n_particles=mission[:filter][:particles],
        ess_threshold=mission[:filter][:ess_threshold],
        rejuvenation_steps=mission[:filter][:rejuvenation_steps],
        diagnostics=keep_history ? diagnostics[:callbacks] : Dict{Symbol,Any}(),
        proposal,
        rng=MersenneTwister(seed),
    )
    som_result = infer_world(som_problem)

    eof_final_field = reconstruct_eof_field(
        eof_result.model;
        coefficients=view(eof_result.coefficient_means, :, size(
            eof_result.coefficient_means, 2,
        )),
    )
    som_final_field = scenario[:som_fields] * view(
        som_result.posterior_probabilities, :,
        size(som_result.posterior_probabilities, 2),
    )
    summary = Dict(
        :prior_distance => world[:actual_distance],
        :requested_distance => world[:requested_distance],
        :eof_final_rmse => weighted_rmse(
            eof_final_field, world[:field], scenario[:spatial_weights],
        ),
        :som_final_rmse => weighted_rmse(
            som_final_field, world[:field], scenario[:spatial_weights],
        ),
    )
    !keep_history && return summary

    field_histories = Dict(
        :EOF => reconstruct_eof_field(
            eof_result.model;
            coefficients=eof_result.coefficient_means,
        ),
        :SOM => scenario[:som_fields] * som_result.posterior_probabilities,
    )
    rmse_histories = Dict(
        name => field_rmse_history(
            fields,
            world[:field],
            scenario[:spatial_weights],
        )
        for (name, fields) in field_histories
    )
    merge(summary, Dict(
        :trial => trial,
        :source => world[:source],
        :truth_field => world[:field],
        :truth_coefficients => world[:coefficients],
        :flow_directions => world[:flow_directions],
        :trajectory => wet_grid_locations(
            scenario[:roms],
            scenario[:quadrature_rows][trajectory[:site_indices]],
        ),
        :elapsed_times => trajectory[:elapsed_times],
        :problem => eof_problem,
        :result => eof_result,
        :som_result => som_result,
        :field_histories => field_histories,
        :rmse_histories => rmse_histories,
        :recovery_diagnostics => diagnostics,
        :truth_vertex => get(world, :vertex, nothing),
    ))
end

function plot_final_rmse(trials)
    targets = sort(unique(getindex.(trials, :requested_distance)))
    groups = [
        filter(trial -> trial[:requested_distance] == target, trials)
        for target in targets
    ]
    distances = [
        mean(getindex.(group, :prior_distance))
        for group in groups
    ]
    eof_rmse = [
        getindex.(group, :eof_final_rmse)
        for group in groups
    ]
    som_rmse = [
        getindex.(group, :som_final_rmse)
        for group in groups
    ]

    panel = plot(
        distances, mean.(eof_rmse); yerror=std.(eof_rmse),
        color=:firebrick, marker=:star5, markersize=7,
        markerstrokecolor=:firebrick, markerstrokewidth=0, linewidth=2.8,
        label="EOF posterior mean",
        xlabel="Actual prior-whitened distance from mean world",
        ylabel="Final spatially weighted field RMSE",
        title="Final recovery (mean ± one standard deviation)",
        size=(1600, 900), legend=:topright,
        left_margin=18Plots.mm, right_margin=8Plots.mm,
        top_margin=6Plots.mm, bottom_margin=14Plots.mm,
        titlefontsize=20, guidefontsize=16, tickfontsize=13, legendfontsize=13,
    )

    plot!(
        panel, distances, mean.(som_rmse);
        yerror=std.(som_rmse), color=:steelblue, marker=:star5, markersize=7,
        markerstrokecolor=:steelblue, markerstrokewidth=0, linewidth=2.8,
        label="SOM posterior mean",
    )

    return panel
end

function save_trial_results(output, mission, scenario, trial)
    mkpath(output)

    field_plot = trial_curl_plot(trial, scenario, mission)
    horizon = size(trial[:field_histories][:EOF], 2) - 1
    frame_count = mission[:visualization][:animation_frames]
    fps = mission[:visualization][:fps]
    animate = mission[:visualization][:animate]
    labels = Dict(
        :EOF => "EOF posterior mean",
        :SOM => "SOM posterior mean",
    )
    observed_title = trial[:source] == :som_vertices ?
        "Observed SOM-vertex world and trajectory" :
        "Observed ROMS-snapshot world and trajectory"

    comparison = plot_world_result_comparison(
        trial[:field_histories],
        labels,
        trial[:truth_field],
        trial[:trajectory],
        field_plot,
        horizon;
        observed_title,
    )
    plot!(
        comparison;
        left_margin=5Plots.mm,
        right_margin=5Plots.mm,
        top_margin=5Plots.mm,
        bottom_margin=7Plots.mm,
    )
    savefig(
        comparison,
        joinpath(output, "eof_som_posterior_comparison.png"),
    )

    if animate
        save_world_result_comparison_animation(
            joinpath(output, "eof_som_posterior_comparison.gif"),
            trial[:field_histories],
            labels,
            trial[:truth_field],
            trial[:trajectory],
            field_plot;
            frame_count,
            fps,
            observed_title,
        )
    end

    rmse = plot_world_method_rmse(trial)
    plot!(
        rmse;
        size=(1400, 800),
        titlefontsize=18,
        guidefontsize=15,
        tickfontsize=12,
        legendfontsize=12,
    )
    savefig(rmse, joinpath(output, "eof_som_rmse.png"))

    save_world_inference_visualizations(
        joinpath(output, "eof"),
        trial[:problem],
        trial[:result],
        trial[:truth_coefficients],
        trial[:truth_field],
        trial[:trajectory],
        scenario[:eof_model].ϕ,
        scenario[:prior_covariance],
        field_plot;
        diagnostics=trial[:recovery_diagnostics],
        frame_count,
        fps,
        animate,
    )
    savefig(
        plot_world_recovery_over_time(trial),
        joinpath(output, "eof", "recovery_over_time.png"),
    )

    save_world_inference_visualizations(
        joinpath(output, "som"),
        trial[:som_result],
        trial[:field_histories][:SOM],
        trial[:truth_field],
        trial[:trajectory],
        field_plot;
        truth_vertex=trial[:truth_vertex],
        frame_count,
        fps,
        animate,
    )
end

function save_source_results(
    output, mission, scenario, source, trials, displayed,
)
    source_output = joinpath(output, source)
    mkpath(source_output)

    savefig(
        plot_world_trial_reconstructions(displayed, scenario, mission),
        joinpath(source_output, "eof_som_world_reconstructions.png"),
    )
    savefig(
        plot_ten_trial_rmse_histories(displayed),
        joinpath(source_output, "rmse_over_time.png"),
    )
    savefig(
        plot_final_rmse(trials),
        joinpath(source_output, "final_rmse_by_prior_distance.png"),
    )
    savefig(
        plot_world_trial_particles(
            displayed,
            scenario[:eof_model].ϕ,
            scenario[:prior_covariance],
        ),
        joinpath(source_output, "eof_particle_locations.png"),
    )

    for trial in displayed
        save_trial_results(
            joinpath(
                source_output,
                "trial_$(lpad(trial[:trial], 2, '0'))",
            ),
            mission,
            scenario,
            trial,
        )
    end
end

function run_source(
    source,
    output,
    mission,
    scenario,
    scores,
    proposal,
    sites,
    rng,
)
    worlds = @match source begin
        "roms_snapshots" => roms_trial_worlds(mission, scenario)
        "som_vertices" => som_trial_worlds(mission, scenario)
    end
    worlds_per_distance = mission[:trials][:worlds_per_distance]
    start_count = mission[:trials][:trajectories_per_world]
    trials = Dict{Symbol,Any}[]
    displayed = Dict{Symbol,Any}[]
    trial = 0

    for (world_index, world) in enumerate(worlds)
        starts = sites[
            randperm(rng, length(sites))[1:start_count]
        ]

        for (start_index, start) in enumerate(starts)
            trial += 1
            println(
                "Running $source trial $trial/" *
                "$(length(worlds) * start_count)"
            )
            keep_history = (world_index - 1) % worlds_per_distance == 0 &&
                start_index == 1
            result = run_trial(
                mission,
                scenario,
                scores,
                proposal,
                world,
                trial,
                start,
                keep_history=keep_history,
            )
            push!(trials, Dict(
                :requested_distance => result[:requested_distance],
                :prior_distance => result[:prior_distance],
                :eof_final_rmse => result[:eof_final_rmse],
                :som_final_rmse => result[:som_final_rmse],
            ))
            if keep_history
                result[:trial] = length(displayed) + 1
                push!(displayed, result)
            end
        end
    end

    save_source_results(
        output,
        mission,
        scenario,
        source,
        trials,
        displayed,
    )
end

function main(mission_path=nothing)
    if isnothing(mission_path)
        mission_path = joinpath(
            @__DIR__,
            "missions",
            "som_comparison_multi_trial.json",
        )
    end

    mission = parsefile(mission_path; dicttype=Dict{Symbol,Any})
    @unpack run_roms, run_som, seed = mission[:trials]

    if !run_roms && !run_som; return nothing; end

    prepared = prepare_mission(mission)
    @unpack mission, scenario, scores, proposal = prepared

    output = normpath(joinpath(@__DIR__, mission[:rel_output_path]))
    mkpath(output)

    sites = [
        Tuple(Float64.(row))
        for row in eachrow(scenario[:eof_context].quadrature)
    ]
    rng = MersenneTwister(seed)

    sources = [
        source for (source, enabled) in (
            ("roms_snapshots", run_roms),
            ("som_vertices", run_som),
        ) if enabled
    ]
    for source in sources
        run_source(
            source,
            output,
            mission,
            scenario,
            scores,
            proposal,
            sites,
            rng,
        )
    end

    println("Saved mission results to $output")
    output
end
