using Arrodes
using JSON3
using LinearAlgebra
using Plots
using Random
using SCRIBE
using SCRIBE.ROMSTools
using Statistics
using VulcanJ

BLAS.set_num_threads(1)

function normalized_curl_target(field, weights, floor)
    density = abs.(field) .+ floor
    density ./ dot(weights, density)
end

function coefficient_recovery(result, truth, prior, prior_covariance)
    prior_factor = cholesky(Symmetric(prior_covariance)).L
    prior_distance = norm(prior_factor \ (prior - truth))
    posterior = result.coefficient_means[:, end]
    posterior_distance = norm(prior_factor \ (posterior - truth))
    posterior_difference = posterior - truth
    posterior_standardized_distance = sqrt(dot(
        posterior_difference,
        result.coefficient_covariances[end] \ posterior_difference,
    ))
    particle_distances = [
        norm(prior_factor \ (view(result.final_particles, :, index) - truth))
        for index in axes(result.final_particles, 2)
    ]
    representative_index = argmin([
        norm(prior_factor \ (
            view(result.final_particles, :, index) - posterior
        ))
        for index in axes(result.final_particles, 2)
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
        :recovery_ratio => posterior_distance / max(prior_distance, eps(Float64)),
        :representative_distance => representative_distance,
        :representative_recovery_ratio => representative_distance /
            max(prior_distance, eps(Float64)),
        :nearest_particle_distance => minimum(particle_distances),
        :weighted_particle_distance => dot(
            result.final_weights,
            particle_distances,
        ),
        :mass_within_one_sigma => dot(
            result.final_weights,
            particle_distances .<= 1.0,
        ),
        :sign_agreement => mean(sign.(posterior) .== sign.(truth)),
        :coefficient_error => coefficient_error,
        :relative_mode_error => relative_mode_error,
        :dominant_modes => dominant_modes,
        :dominant_mode_relative_error => relative_mode_error[dominant_modes],
    )
end

function trial_worlds(mission, scenario)
    factor = cholesky(Symmetric(scenario[:context].prior_covariance)).L
    snapshots = collect(
        scenario[:validation_start]:size(scenario[:roms][:data], 2),
    )
    coefficients = SCRIBE.eof_coefficients(
        scenario[:model],
        view(scenario[:roms][:data], :, snapshots),
    )
    distances = [
        norm(factor \ (
            view(coefficients, :, index) - scenario[:context].model.ϕ
        ))
        for index in axes(coefficients, 2)
    ]
    available = trues(length(snapshots))
    [begin
        candidates = findall(available)
        selected = candidates[argmin(abs.(distances[candidates] .- target))]
        available[selected] = false
        Dict(
            :snapshot => snapshots[selected],
            :coefficients => Vector{Float64}(view(coefficients, :, selected)),
            :requested_distance => Float64(target),
            :actual_distance => distances[selected],
        )
    end for target in mission[:trials][:prior_sigma_distances]]
end

function ergodic_trajectory(mission, scenario, score, coefficients)
    target_problem = WorldInferenceProblem(
        context=scenario[:context],
        score=score,
        observations=TrajectoryObservation[],
    )
    density = target_measure(target_problem, coefficients)
    sites = [
        Tuple(Float64.(row))
        for row in eachrow(scenario[:context].quadrature)
    ]
    start = sites[argmin(sum(
        abs2,
        row,
    ) for row in eachrow(scenario[:context].quadrature))]
    bounds = VulcanJ.coordinate_bounds(sites)
    planned_path, _, _, _ = VulcanJ.kernel_ergodic_trajectory(
        start,
        sites,
        density,
        bounds,
        mission[:trajectory][:samples] - 1;
        density_bandwidth=scenario[:planner_bandwidth],
        kernel_bandwidth=scenario[:planner_bandwidth],
        dt=1.0,
        optimizer_iters=mission[:trajectory][:ergodic_iterations],
        learning_rate=0.22,
        momentum=0.85,
        control_weight=2e-3,
        boundary_weight=20.0,
        max_speed=1 / 16,
        line_search_steps=8,
        line_search_decay=0.5,
    )
    points = reduce(
        vcat,
        (reshape(collect(point), 1, :) for point in planned_path),
    )
    Dict(
        :observations => [begin
            state = vec(points[timestep, :])
            previous = timestep == 1 ?
                state :
                vec(points[timestep - 1, :])
            TrajectoryObservation(state=state, action=state - previous)
        end for timestep in axes(points, 1)],
        :site_indices => [
            argmin(sum(abs2, point - site)
                   for site in eachrow(scenario[:context].quadrature))
            for point in eachrow(points)
        ],
    )
end

function run_trial(
    mission,
    scenario,
    score,
    proposal,
    flow_directions,
    world,
    trial,
)
    snapshot = world[:snapshot]
    coefficients = world[:coefficients]
    observed = Dict(
        :coefficients => coefficients,
        :field => SCRIBE.reconstruct_eof_field(
            scenario[:model];
            coefficients,
        ),
    )
    raw_field = Vector{Float64}(view(
        scenario[:roms][:data],
        :,
        snapshot,
    ))
    reconstruction_error = observed[:field] - raw_field
    raw_rms = sqrt(mean(abs2, raw_field))
    eof_rms = sqrt(mean(abs2, observed[:field]))
    curl_diagnostics = Dict(
        :raw_min => minimum(raw_field),
        :raw_max => maximum(raw_field),
        :raw_rms => raw_rms,
        :eof_min => minimum(observed[:field]),
        :eof_max => maximum(observed[:field]),
        :eof_rms => eof_rms,
        :rms_retention => eof_rms / max(raw_rms, eps(Float64)),
        :relative_reconstruction_error => norm(reconstruction_error) /
            max(norm(raw_field), eps(Float64)),
    )
    println(
        "  shape audit: raw RMS=$(round(curl_diagnostics[:raw_rms]; digits=3)), " *
        "rank-$(mission[:roms][:eof_rank]) RMS=$(round(curl_diagnostics[:eof_rms]; digits=3)), " *
        "relative EOF error=$(round(curl_diagnostics[:relative_reconstruction_error]; digits=3))",
    )
    seed = UInt64(mission[:trials][:seed]) + UInt64(2trial)
    trajectory = ergodic_trajectory(
        mission,
        scenario,
        score,
        observed[:coefficients],
    )
    problem = WorldInferenceProblem(
        context=scenario[:context],
        score=score,
        observations=trajectory[:observations],
    )
    result = infer_world(
        problem;
        n_particles=mission[:filter][:particles],
        ess_threshold=mission[:filter][:ess_threshold],
        rejuvenation_steps=mission[:filter][:rejuvenation_steps],
        proposal,
        rng=MersenneTwister(seed + 1),
    )
    recovery = coefficient_recovery(
        result,
        observed[:coefficients],
        scenario[:context].model.ϕ,
        scenario[:context].prior_covariance,
    )
    design = world
    inferred_coefficients = result.coefficient_means[:, end]
    inferred_field = SCRIBE.reconstruct_eof_field(
        result.model;
        coefficients=inferred_coefficients,
    )
    full_weights = result.model.params.decomposition.weights
    truth_target_field = normalized_curl_target(
        observed[:field],
        full_weights,
        scenario[:target_floor],
    )
    inferred_target_field = zeros(length(inferred_field))
    for index in axes(result.final_particles, 2)
        particle_field = SCRIBE.reconstruct_eof_field(
            result.model;
            coefficients=view(result.final_particles, :, index),
        )
        inferred_target_field .+= result.final_weights[index] .* normalized_curl_target(
            particle_field,
            full_weights,
            scenario[:target_floor],
        )
    end
    discrepancy_cache = Dict{Symbol,Any}()
    horizon = length(problem.observations)
    compared_worlds = Dict(
        :prior => scenario[:context].model.ϕ,
        :observed => observed[:coefficients],
        :posterior_predictive => result,
    )
    trajectory_discrepancy = Dict(
        name => kernel_discrepancy(
            problem,
            horizon,
            coefficients,
            discrepancy_cache,
        )
        for (name, coefficients) in compared_worlds
    )
    println(
        "  final trajectory MMD²: prior=$(round(trajectory_discrepancy[:prior]; sigdigits=4)), " *
        "observed=$(round(trajectory_discrepancy[:observed]; sigdigits=4)), " *
        "posterior predictive=$(round(trajectory_discrepancy[:posterior_predictive]; sigdigits=4)); " *
        "minimum ESS=$(round(minimum(result.ess_history); digits=1))",
    )
    println(
        "  coefficient recovery: requested=$(design[:requested_distance])σ, " *
        "actual=$(round(recovery[:prior_distance]; digits=2))σ, " *
        "posterior=$(round(recovery[:posterior_distance]; digits=2))σ, " *
        "Rϕ=$(round(recovery[:recovery_ratio]; digits=3)), " *
        "posterior-z=$(round(recovery[:posterior_standardized_distance]; digits=2)), " *
        "nearest=$(round(recovery[:nearest_particle_distance]; digits=2))σ, " *
        "mass≤1σ=$(round(100recovery[:mass_within_one_sigma]; digits=1))%, " *
        "dominant-mode error=$(round(mean(
            recovery[:dominant_mode_relative_error],
        ); digits=2))",
    )
    Dict(
        :trial => trial,
        :snapshot => snapshot,
        :prior_distance => recovery[:prior_distance],
        :requested_distance => design[:requested_distance],
        :coefficient_recovery => recovery,
        :initial_discrepancy => target_measure_mmd(
            problem,
            scenario[:context].model.ϕ,
            observed[:coefficients],
            discrepancy_cache,
        ),
        :final_discrepancy => target_measure_mmd(
            problem,
            result,
            observed[:coefficients],
            discrepancy_cache,
        ),
        :truth_field => observed[:field],
        :truth_target_field => truth_target_field,
        :inferred_field => inferred_field,
        :inferred_target_field => inferred_target_field,
        :truth_coefficients => observed[:coefficients],
        :curl_diagnostics => curl_diagnostics,
        :trajectory_discrepancy => trajectory_discrepancy,
        :flow_directions => flow_directions,
        :problem => problem,
        :result => result,
        :trajectory => wet_grid_locations(
            scenario[:roms],
            scenario[:quadrature_rows][trajectory[:site_indices]],
        ),
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
    diagnostics = Dict(
        "representation" => "normalized absolute-curl shape",
        "field_units" => "dimensionless",
        "definition" => "vertical vorticity dv/dx - du/dy",
        "velocity_assumption" => "u eastward and v northward on the collocated lon/lat grid",
        "eof_rank" => mission[:roms][:eof_rank],
        "proposal" => Dict(
            String(key) => value
            for (key, value) in pairs(mission[:filter][:proposal])
        ),
        "discrepancy_scale" => trials[1][:problem].score.discrepancy_scale,
        "kernel_bandwidth" => trials[1][:problem].score.kernel_bandwidth,
        "trials" => [merge(
            Dict(
                "trial" => trial[:trial],
                "snapshot" => trial[:snapshot],
                "requested_prior_sigma_distance" => trial[:requested_distance],
                "actual_prior_sigma_distance" => trial[:prior_distance],
                "prior_target_mmd2" => trial[:initial_discrepancy],
                "inferred_target_mmd2" => trial[:final_discrepancy],
                "prior_trajectory_mmd2" => trial[:trajectory_discrepancy][:prior],
                "observed_trajectory_mmd2" => trial[:trajectory_discrepancy][:observed],
                "posterior_predictive_trajectory_mmd2" => trial[:trajectory_discrepancy][:posterior_predictive],
                "minimum_ess" => minimum(trial[:result].ess_history),
                "median_ess" => median(trial[:result].ess_history),
                "resampling_steps" => count(trial[:result].resampled),
            ),
            Dict(
                String(key) => value
                for (key, value) in trial[:curl_diagnostics]
            ),
            Dict(
                String(key) => value
                for (key, value) in trial[:coefficient_recovery]
            ),
        ) for trial in trials],
    )
    open(joinpath(output, "curl_reconstruction_diagnostics.json"), "w") do io
        JSON3.pretty(io, diagnostics)
        write(io, '\n')
    end
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
            frame_count=mission[:visualization][:animation_frames],
            fps=mission[:visualization][:fps],
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

function prepare_mission(mission_path)
    mission = copy(JSON3.read(read(mission_path, String)))
    println("Preparing $(mission[:name]) from ROMS curl snapshots ...")
    settings = mission[:roms]
    archive = normpath(joinpath(@__DIR__, mission[:roms_archive]))
    roms = prepare_roms_curl_shape(
        archive;
        temporal_stride=settings[:temporal_stride],
        spatial_stride=settings[:spatial_stride],
    )
    fitted = fit_roms_eof(
        roms;
        training_fraction=settings[:training_fraction],
        rank=settings[:eof_rank],
        oversample=settings[:eof_oversample],
        power_iterations=settings[:eof_power_iterations],
    )
    params = fitted[:model].params
    ego_snapshot = round(Int, mission[:ego][:snapshot_fraction] * fitted[:n_training])
    ego_coefficients = SCRIBE.eof_coefficients(
        params,
        roms[:data][:, ego_snapshot],
    )
    model = SCRIBE.eof_model_at_coefficients(params, ego_coefficients)
    prior_covariance =
        mission[:observed_world_prior][:archive_covariance_multiplier] .*
        SCRIBE.eof_prior_covariance(model)
    quadrature_count = min(settings[:quadrature_count], size(roms[:locations], 1))
    center = mean(roms[:locations]; dims=1)
    quadrature_rows = [argmin(vec(sum(
        abs2,
        roms[:locations] .- center;
        dims=2,
    )))]
    distances = vec(sum(
        abs2,
        roms[:locations] .- roms[:locations][first(quadrature_rows), :]';
        dims=2,
    ))
    while length(quadrature_rows) < quadrature_count
        row = argmax(distances)
        push!(quadrature_rows, row)
        distances = min.(distances, vec(sum(
            abs2,
            roms[:locations] .- roms[:locations][row, :]';
            dims=2,
        )))
    end
    quadrature = roms[:locations][quadrature_rows, :]
    minima = vec(minimum(quadrature; dims=1))
    spans = max.(
        vec(maximum(quadrature; dims=1)) - minima,
        eps(Float64),
    )
    kernel_locations = (quadrature .- minima') ./ spans'
    context = world_inference_context(
        model;
        quadrature,
        kernel_locations,
        quadrature_weights=params.decomposition.weights[quadrature_rows],
        prior_covariance,
    )
    calibration_ids = unique(round.(Int, range(
        fitted[:n_training] + 1,
        fitted[:calibration_end];
        length=settings[:calibration_worlds],
    )))
    kernel_bandwidth = Float64(mission[:target][:kernel_bandwidth])
    scenario = Dict(
        :roms => roms,
        :model => model,
        :context => context,
        :ego_snapshot => ego_snapshot,
        :validation_start => fitted[:validation_start],
        :quadrature_rows => quadrature_rows,
        :kernel_bandwidth => kernel_bandwidth,
        :planner_bandwidth => kernel_bandwidth * maximum(spans),
        :metric_minima => minima,
        :metric_spans => spans,
        :field_scale => fitted[:field_scale],
        :calibration_coefficients => [
            SCRIBE.eof_coefficients(params, roms[:data][:, snapshot])
            for snapshot in calibration_ids
        ],
    )
    target = eof_target_field(
        link=:magnitude,
        floor=mission[:target][:floor_fraction] * scenario[:field_scale],
        name=:absolute_curl,
    )
    scenario[:target_floor] = mission[:target][:floor_fraction] * scenario[:field_scale]
    template = eof_field_score(
        target;
        kernel_bandwidth=scenario[:kernel_bandwidth],
        discrepancy_scale=1.0,
        β_max=mission[:filter][:beta_max],
        maturity_half_time=mission[:filter][:maturity_half_time],
        maturity_power=mission[:filter][:maturity_power],
        location=state -> (
            Float64.(state) .- scenario[:metric_minima]
        ) ./ scenario[:metric_spans],
    )
    discrepancy_unit = calibrate_discrepancy_scale(
        scenario[:context],
        template,
        scenario[:calibration_coefficients],
    )
    println(
        "Offline MMD² unit=$(round(discrepancy_unit; sigdigits=4))",
    )
    score = eof_field_score(
        target;
        kernel_bandwidth=scenario[:kernel_bandwidth],
        discrepancy_scale=discrepancy_unit,
        β_max=mission[:filter][:beta_max],
        maturity_half_time=mission[:filter][:maturity_half_time],
        maturity_power=mission[:filter][:maturity_power],
        location=state -> (
            Float64.(state) .- scenario[:metric_minima]
        ) ./ scenario[:metric_spans],
    )
    proposal = Dict{Symbol,Any}(
        :mechanism => Symbol(mission[:filter][:proposal][:mechanism]),
        :covariance_scale => Float64(
            mission[:filter][:proposal][:covariance_scale],
        ),
        :optimizer_steps => Int(
            mission[:filter][:proposal][:optimizer_steps],
        ),
    )
    worlds = trial_worlds(mission, scenario)
    Dict(
        :mission => mission,
        :archive => archive,
        :scenario => scenario,
        :score => score,
        :proposal => proposal,
        :worlds => worlds,
    )
end

function main(mission_path=nothing)
    if isnothing(mission_path)
        mission_path = joinpath(
            @__DIR__,
            "missions",
            "curl_mmd_multi_trial.json",
        )
    end
    prepared = prepare_mission(mission_path)
    mission = prepared[:mission]
    scenario = prepared[:scenario]
    score = prepared[:score]
    proposal = prepared[:proposal]
    worlds = prepared[:worlds]
    archive = prepared[:archive]
    snapshots = unique(getindex.(worlds, :snapshot))
    flow_directions = read_roms_flow_directions(
        archive,
        scenario[:roms],
        snapshots,
    )
    trials = [begin
        println("Running trial $trial/$(length(worlds))")
        run_trial(
            mission,
            scenario,
            score,
            proposal,
            flow_directions[world[:snapshot]],
            world,
            trial,
        )
    end for (trial, world) in enumerate(worlds)]
    output = save_results(mission, scenario, trials)
    println("Saved mission results to $output")
end
