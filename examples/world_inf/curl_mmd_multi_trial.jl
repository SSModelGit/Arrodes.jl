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

function trial_snapshots(mission, scenario)
    candidates = unique(round.(Int, range(
        scenario[:validation_start],
        size(scenario[:roms][:data], 2);
        length=mission[:trials][:candidate_snapshots],
    )))
    factor = cholesky(Symmetric(scenario[:context].prior_covariance)).L
    distances = [norm(factor \ (
        SCRIBE.eof_coefficients(
            scenario[:model],
            scenario[:roms][:data][:, snapshot],
        ) - scenario[:context].model.ϕ
    )) for snapshot in candidates]
    ordering = sortperm(distances)
    positions = unique(round.(Int, range(
        1,
        length(ordering);
        length=mission[:trials][:count],
    )))
    candidates[ordering[positions]]
end

function run_trial(
    mission,
    scenario,
    score,
    proposal,
    flow_directions,
    snapshot,
    trial,
)
    raw_field = Vector{Float64}(view(
        scenario[:roms][:data],
        :,
        snapshot,
    ))
    coefficients = SCRIBE.eof_coefficients(
        scenario[:model],
        raw_field,
    )
    observed = Dict(
        :coefficients => coefficients,
        :field => SCRIBE.reconstruct_eof_field(
            scenario[:model];
            coefficients,
        ),
    )
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
        "  curl audit: raw RMS=$(round(1e3curl_diagnostics[:raw_rms]; digits=3)), " *
        "rank-$(mission[:roms][:eof_rank]) RMS=$(round(1e3curl_diagnostics[:eof_rms]; digits=3)) " *
        "(10⁻³ s⁻¹), relative EOF error=$(round(curl_diagnostics[:relative_reconstruction_error]; digits=3))",
    )
    target_problem = WorldInferenceProblem(
        context=scenario[:context],
        score=score,
        observations=TrajectoryObservation[],
    )
    density = target_measure(target_problem, observed[:coefficients])
    seed = UInt64(mission[:trials][:seed]) + UInt64(2trial)
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
        density_bandwidth=scenario[:kernel_bandwidth],
        kernel_bandwidth=scenario[:kernel_bandwidth],
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
                   for site in eachrow(scenario[:context].quadrature))
            for point in eachrow(points)
        ],
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
    inferred_coefficients = result.coefficient_means[:, end]
    inferred_field = SCRIBE.reconstruct_eof_field(
        result.model;
        coefficients=inferred_coefficients,
    )
    factor = cholesky(Symmetric(scenario[:context].prior_covariance)).L
    discrepancy_cache = Dict{Symbol,Any}()
    Dict(
        :trial => trial,
        :snapshot => snapshot,
        :prior_distance => norm(factor \ (
            observed[:coefficients] - scenario[:context].model.ϕ
        )),
        :initial_discrepancy => target_measure_mmd(
            problem,
            scenario[:context].model.ϕ,
            observed[:coefficients],
            discrepancy_cache,
        ),
        :final_discrepancy => target_measure_mmd(
            problem,
            inferred_coefficients,
            observed[:coefficients],
            discrepancy_cache,
        ),
        :truth_field => observed[:field],
        :inferred_field => inferred_field,
        :truth_coefficients => observed[:coefficients],
        :curl_diagnostics => curl_diagnostics,
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
            maximum(abs, trial[:truth_field]),
            maximum(abs, trial[:inferred_field]),
            eps(Float64),
        )
        truth = plot_roms_curl(
            trial[:truth_field],
            trial[:flow_directions],
            roms;
            arrow_stride=2arrow_stride,
            title="T$(lpad(trial[:trial], 2, '0')) observed (±$(round(1e3limit; sigdigits=2)) ×10⁻³ s⁻¹)",
            limit,
            colorbar=false,
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
            trial[:inferred_field],
            trial[:flow_directions],
            roms;
            arrow_stride=2arrow_stride,
            title="T$(lpad(trial[:trial], 2, '0')) inferred",
            limit,
            colorbar=false,
        ))
    end
    savefig(
        plot(
            panels...;
            layout=(5, 4),
            size=(3600, 2100),
            plot_title="Signed curl and equal-length ROMS flow directions",
            plot_titlefontsize=20,
            titlefontsize=12,
        ),
        path,
    )
end

function save_results(mission, scenario, trials)
    output = normpath(joinpath(@__DIR__, mission[:output]))
    mkpath(output)
    diagnostics = Dict(
        "curl_units" => "s^-1",
        "display_units" => "10^-3 s^-1",
        "definition" => "vertical vorticity dv/dx - du/dy",
        "velocity_assumption" => "u eastward and v northward on the collocated lon/lat grid",
        "eof_rank" => mission[:roms][:eof_rank],
        "trials" => [merge(
            Dict(
                "trial" => trial[:trial],
                "snapshot" => trial[:snapshot],
            ),
            Dict(
                String(key) => value
                for (key, value) in trial[:curl_diagnostics]
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
    end
    output
end

function main()
    mission_path = joinpath(
        @__DIR__,
        "missions",
        "curl_mmd_multi_trial.json",
    )
    mission = copy(JSON3.read(read(mission_path, String)))
    println("Preparing $(mission[:name]) from ROMS curl snapshots ...")
    settings = mission[:roms]
    archive = normpath(joinpath(@__DIR__, mission[:roms_archive]))
    roms = prepare_roms_curl(
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
    pairwise_distances = [
        norm(view(quadrature, left, :) - view(quadrature, right, :))
        for left in axes(quadrature, 1), right in axes(quadrature, 1)
    ]
    kernel_bandwidth = 2.5median([
        minimum(filter(>(0.0), pairwise_distances[row, :]))
        for row in axes(pairwise_distances, 1)
    ])
    scenario = Dict(
        :roms => roms,
        :model => model,
        :context => context,
        :validation_start => fitted[:validation_start],
        :quadrature_rows => quadrature_rows,
        :kernel_bandwidth => kernel_bandwidth,
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
    template = eof_field_score(
        target;
        kernel_bandwidth=scenario[:kernel_bandwidth],
        discrepancy_scale=1.0,
        β_max=mission[:filter][:beta_max],
        maturity_half_time=mission[:filter][:maturity_half_time],
        maturity_power=mission[:filter][:maturity_power],
    )
    score = eof_field_score(
        target;
        kernel_bandwidth=scenario[:kernel_bandwidth],
        discrepancy_scale=calibrate_discrepancy_scale(
            scenario[:context],
            template,
            scenario[:calibration_coefficients],
        ),
        β_max=mission[:filter][:beta_max],
        maturity_half_time=mission[:filter][:maturity_half_time],
        maturity_power=mission[:filter][:maturity_power],
    )
    proposal = Dict{Symbol,Any}(
        :mechanism => Symbol(mission[:filter][:proposal][:mechanism]),
        :scale => Float64(mission[:filter][:proposal][:scale]),
    )
    snapshots = trial_snapshots(mission, scenario)
    flow_directions = read_roms_flow_directions(archive, roms, snapshots)
    trials = [begin
        println("Running trial $trial/$(length(snapshots))")
        run_trial(
            mission,
            scenario,
            score,
            proposal,
            flow_directions[snapshot],
            snapshot,
            trial,
        )
    end for (trial, snapshot) in enumerate(snapshots)]
    output = save_results(mission, scenario, trials)
    println("Saved mission results to $output")
end

main()
