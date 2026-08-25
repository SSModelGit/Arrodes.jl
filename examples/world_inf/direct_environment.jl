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

function main()
    mission_path = joinpath(
        @__DIR__,
        "missions",
        "direct_environment.json",
    )
    mission = copy(JSON3.read(read(mission_path, String)))
    println("Preparing $(mission[:name]) ...")
    settings = mission[:roms]
    roms = prepare_roms_component(
        normpath(joinpath(@__DIR__, mission[:roms_archive])),
        :u;
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
    ego_coefficients = SCRIBE.eof_coefficients(params, roms[:data][:, ego_snapshot])
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
    context = world_inference_context(
        model;
        quadrature,
        quadrature_weights=params.decomposition.weights[quadrature_rows],
        prior_covariance,
    )
    calibration_ids = unique(round.(Int, range(
        fitted[:n_training] + 1,
        fitted[:calibration_end];
        length=settings[:calibration_worlds],
    )))
    calibration_coefficients = [
        SCRIBE.eof_coefficients(params, roms[:data][:, snapshot])
        for snapshot in calibration_ids
    ]
    pairwise_distances = [
        norm(view(quadrature, left, :) - view(quadrature, right, :))
        for left in axes(quadrature, 1), right in axes(quadrature, 1)
    ]
    kernel_bandwidth = 2.5median([
        minimum(filter(>(0.0), pairwise_distances[row, :]))
        for row in axes(pairwise_distances, 1)
    ])
    target = eof_target_field(
        link=Symbol(mission[:target][:link]),
        floor=mission[:target][:floor_fraction] * fitted[:field_scale],
        name=Symbol(mission[:target][:name]),
    )
    template = eof_field_score(
        target;
        kernel_bandwidth,
        discrepancy_scale=1.0,
        β_max=mission[:filter][:beta_max],
        maturity_half_time=mission[:filter][:maturity_half_time],
        maturity_power=mission[:filter][:maturity_power],
    )
    scale = calibrate_discrepancy_scale(
        context,
        template,
        calibration_coefficients,
    )
    score = eof_field_score(
        target;
        kernel_bandwidth,
        discrepancy_scale=scale,
        β_max=mission[:filter][:beta_max],
        maturity_half_time=mission[:filter][:maturity_half_time],
        maturity_power=mission[:filter][:maturity_power],
    )
    validation_count = size(roms[:data], 2) - fitted[:validation_start] + 1
    snapshot = fitted[:validation_start] + round(
        Int,
        mission[:observed][:validation_fraction] * (validation_count - 1),
    )
    observed_coefficients = SCRIBE.eof_coefficients(
        model,
        roms[:data][:, snapshot],
    )
    observed_field = SCRIBE.reconstruct_eof_field(
        model;
        coefficients=observed_coefficients,
    )
    target_problem = WorldInferenceProblem(
        context=context,
        score=score,
        observations=TrajectoryObservation[],
    )
    sites = [Tuple(Float64.(row)) for row in eachrow(context.quadrature)]
    start = sites[argmin(sum(abs2, row) for row in eachrow(context.quadrature))]
    bounds = VulcanJ.coordinate_bounds(sites)
    planned_path, _, _, _ = VulcanJ.kernel_ergodic_trajectory(
        start,
        sites,
        target_measure(target_problem, observed_coefficients),
        bounds,
        mission[:trajectory][:samples] - 1;
        density_bandwidth=kernel_bandwidth,
        kernel_bandwidth,
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
    trajectory = Dict(
        :observations => [begin
            state = vec(points[timestep, :])
            previous = timestep == 1 ?
                state :
                vec(points[timestep - 1, :])
            TrajectoryObservation(state=state, action=state - previous)
        end for timestep in axes(points, 1)],
        :site_indices => [
            argmin(sum(abs2, point - site)
                   for site in eachrow(context.quadrature))
            for point in eachrow(points)
        ],
    )
    problem = WorldInferenceProblem(
        context=context,
        score=score,
        observations=trajectory[:observations],
    )
    proposal = Dict{Symbol,Any}(
        :mechanism => Symbol(mission[:filter][:proposal][:mechanism]),
        :scale => Float64(mission[:filter][:proposal][:scale]),
    )
    result = infer_world(
        problem;
        n_particles=mission[:filter][:particles],
        ess_threshold=mission[:filter][:ess_threshold],
        rejuvenation_steps=mission[:filter][:rejuvenation_steps],
        proposal,
        rng=MersenneTwister(UInt64(mission[:seed]) + 1),
    )
    posterior = world_posterior(result)
    output = normpath(joinpath(@__DIR__, mission[:output]))
    mkpath(output)
    path = wet_grid_locations(
        roms,
        quadrature_rows[trajectory[:site_indices]],
    )
    field_plot = (field, title, limit) -> plot_roms_field(
        field,
        roms;
        title,
        clims=(-limit, limit),
    )
    save_world_inference_visualizations(
        output,
        problem,
        result,
        observed_coefficients,
        observed_field,
        path,
        context.model.ϕ,
        prior_covariance,
        field_plot;
        frame_count=mission[:visualization][:animation_frames],
        fps=mission[:visualization][:fps],
    )
    initial_rmse = sqrt(mean(abs2, SCRIBE.reconstruct_eof_field(model) - observed_field))
    final_rmse = sqrt(mean(abs2, posterior[:map_mean] - observed_field))
    println("Initial field RMSE: $initial_rmse")
    println("Final field RMSE: $final_rmse")
    println("Saved mission results to $output")
end

main()
