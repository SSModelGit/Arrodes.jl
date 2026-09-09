using Arrodes
using SCRIBE
using SCRIBE.ROMSTools
using LinearAlgebra: BLAS, Symmetric, cholesky, norm
using Statistics: mean
using Random: MersenneTwister

import JSON
using NCDatasets
using UnPack: @pack!, @unpack

include("_sim_trial_helpers.jl")

BLAS.set_num_threads(1)

function load_som_data(rel_path::String)
    NCDataset(joinpath(@__DIR__, rel_path), "r") do ds
        vertex_names = Symbol.(sort(filter(k->startswith(k, "vertex_0"), collect(keys(ds)))))
        Dict{Symbol, Any}(
            :x => Float64.(ds["x_coords"][:]),
            :y => Float64.(ds["y_coords"][:]),
            :components => ds["component"][:],
            :connectivity => Float64.(ds["topology_connectivity"][:,:]),
            :mask => Bool.(permutedims(ds["mask"][:,:,:], (3,2,1))),
            :vertex_ids => Int.(ds["vertex_id"][:]),
            :vertex_names => vertex_names,
            :vertices => [permutedims(coalesce.(ds[k][:,:,:], NaN), (3,2,1))
                for k in vertex_names
            ],
        )
    end
end

function som_vertex_curl(vertex, x, y)
    u = view(vertex, :, :, 1)
    v = view(vertex, :, :, 2)
    curl = fill(NaN, size(u))

    for j in axes(u, 2), i in axes(u, 1)
        i₀, i₁ = max(i-1, 1), min(i+1, size(u, 1))
        j₀, j₁ = max(j-1, 1), min(j+1, size(u, 2))
        
        stencil = (
            u[i₀, j], u[i₁, j], u[i, j₀], u[i, j₁],
            v[i₀, j], v[i₁, j], v[i, j₀], v[i, j₁]
        )
        if !all(isfinite, stencil); continue; end

        ∂v_∂x = (v[i₁, j] - v[i₀, j]) / (x[i₁] - x[i₀])
        ∂u_∂y = (u[i, j₁] - u[i, j₀]) / (y[j₁] - y[j₀])
        curl[i, j] = ∂v_∂x - ∂u_∂y
    end
    return curl
end

function som_vertex_world(vertex, som, roms)
    wet = roms[:wet_mask]

    curl = som_vertex_curl(vertex, som[:x], som[:y])
    field = abs.(vec(curl)[wet])
    field ./= mean(field)

    u = vec(view(vertex, :, :, 1))[wet]
    v = vec(view(vertex, :, :, 2))[wet]
    speed = hypot.(u, v)

    flow_directions = hcat(u, v)
    moving = speed .> eps(Float64)
    flow_directions[moving, :] ./= reshape(speed[moving], :, 1)
    flow_directions[.!moving, :] .= 0.0
    return Dict(
        :field => field,
        :flow_directions => flow_directions,
    )
end

function trial_worlds(mission, scenario)
    let world_setup_type = mission[:trials][:world_source]
        @match world_setup_type begin
            "roms_snapshots" => trial_worlds_from_roms_snapshots(mission, scenario)
            "som_vertices" => trial_worlds_from_som_vertices(mission, scenario)
        end
    end
end

function distinguish_worlds_by_distance(mission, scenario, candidates)
    context = scenario[:context]
    factor = cholesky(Symmetric(context.prior_covariance)).L
    prior = context.model.ϕ
    distances = [norm(factor \ (world[:coefficients] - prior)) for world in candidates]
    available = trues(length(candidates))

    [begin
        remaining = findall(available)
        selected = remaining[argmin(abs.(distances[remaining] .- target))]
        available[selected] = false

        merge(
            candidates[selected],
            Dict(
                :requested_distance => Float64(target),
                :actual_distance => distances[selected]
            )
        )
    end for target in mission[:trials][:prior_sigma_distances]]
end

function trial_worlds_from_roms_snapshots(mission, scenario)
    @unpack archive, roms = scenario
    roms_data = roms[:data]
    snapshots = collect(scenario[:validation_start]:size(roms_data, 2))
    coefficients = SCRIBE.eof_coefficients(scenario[:model], view(roms_data, :, snapshots))

    candidates = [
        Dict(
            :source => :roms_snapshot,
            :snapshot => snapshots[index],
            :coefficients => Vector{Float64}(view(coefficients, :, index)),
        )
        for index in axes(coefficients, 2)
    ]
    worlds = distinguish_worlds_by_distance(mission, scenario, candidates)

    directions = read_roms_flow_directions(archive, roms, getindex.(worlds, :snapshot))
    return [
        merge(world, 
              Dict(:flow_directions => directions[world[:snapshot]]))
    for world in worlds]
end

function trial_worlds_from_som_vertices(mission, scenario)
    @unpack som_model, roms, model = scenario
    @unpack vertex_ids, vertices = som_model
    candidates = [begin
        vertex_world = som_vertex_world(vertices[index], som_model, roms)
        coefficients = SCRIBE.eof_coefficients(model, vertex_world[:field])
        Dict(
            :source => :som_vertex,
            :vertex_id => vertex_ids[index],
            :coefficients => Vector{Float64}(coefficients),
            :flow_directions => vertex_world[:flow_directions],
        )
    end for index in eachindex(vertices)]
    return distinguish_worlds_by_distance(mission, scenario, candidates)
end

function load_mission_info(mission_path::String)
    mission = JSON.parsefile(mission_path; dicttype=Dict{Symbol,Any})
    println("Preparing $(mission[:name]) from ROMS data and pre-trained SOM maps ...")
    @unpack temporal_stride, spatial_stride, training_fraction,
            eof_rank, eof_oversample, eof_power_iterations,
            quadrature_count, calibration_worlds = mission[:roms]
    archive = normpath(joinpath(@__DIR__, mission[:roms_archive]))

    roms = prepare_roms_curl_shape(
        archive,
        temporal_stride=temporal_stride,
        spatial_stride=spatial_stride
    )
    fitted = fit_roms_eof(roms;
        training_fraction=training_fraction,
        rank=eof_rank,
        oversample=eof_oversample,
        power_iterations=eof_power_iterations
    )
    som_model = load_som_data(joinpath(@__DIR__, mission[:som_model_data]))
    @unpack data, locations = roms
    roms_data, roms_locs = data, locations
    @unpack n_training, field_scale, calibration_end, validation_start = fitted
    params = fitted[:model].params

    let snapshot_fraction = mission[:ego][:snapshot_fraction],
        prior_covariance_multiplier = mission[:observed_world_prior][:archive_covariance_multiplier],
        n_locs_roms = size(roms_locs, 1),
        center_roms = mean(roms_locs; dims=1)

        ego_snapshot = round(Int, snapshot_fraction * n_training)
        ego_coefficients = SCRIBE.eof_coefficients(
            params, roms_data[:, ego_snapshot]
        )
        model = SCRIBE.eof_model_at_coefficients(params, ego_coefficients)
        prior_covariance = prior_covariance_multiplier .*
            SCRIBE.eof_prior_covariance(model)
        selected_quadrature_count = min(quadrature_count, n_locs_roms)
        quadrature_rows = [argmin(vec(sum(abs2, roms_locs .- center_roms; dims=2)))]
        distances = vec(sum(abs2, roms_locs .- roms_locs[first(quadrature_rows), :]'; dims=2))
        while length(quadrature_rows) < selected_quadrature_count
            next_row = argmax(distances)
            push!(quadrature_rows, next_row)
            distances = min.(distances, vec(sum(abs2, roms_locs .- roms_locs[next_row, :]'; dims=2)))
        end
        quadrature = roms_locs[quadrature_rows, :]

        @unpack link, floor_fraction, kernel_bandwidth, name = mission[:target]
        @unpack beta_max, maturity_half_time, maturity_power = mission[:filter]
        minima = vec(minimum(quadrature; dims=1))
        spans = max.(vec(maximum(quadrature; dims=1)) .- minima, eps(Float64))
        kernel_locations = (quadrature .- minima') ./ spans'
        calibration_ids = unique(round.(Int, range(
            n_training + 1, calibration_end;
            length=calibration_worlds
        )))
        context = world_inference_context(
            model;
            quadrature,
            kernel_locations,
            quadrature_weights=params.decomposition.weights[quadrature_rows],
            prior_covariance,
        )
        target_floor = floor_fraction * field_scale
        calibration_coefficients = [
            SCRIBE.eof_coefficients(params, roms_data[:, snapshot])
            for snapshot in calibration_ids
        ]
        scenario = Dict(
            :archive => archive,
            :roms => roms,
            :model => model,
            :context => context,
            :som_model => som_model,
            :ego_snapshot => ego_snapshot,
            :validation_start => validation_start,
            :quadrature_rows => quadrature_rows,
            :kernel_bandwidth => kernel_bandwidth,
            :planner_bandwidth => kernel_bandwidth * maximum(spans),
            :metric_minima => minima,
            :metric_spans => spans,
            :field_scale => field_scale,
            :target_floor => target_floor,
            :calibration_coefficients => calibration_coefficients,
        )

        target = eof_target_field(
            link=Symbol(link), floor=target_floor, name=Symbol(name),
        )
        template = eof_field_score(
            target;
            kernel_bandwidth=kernel_bandwidth,
            discrepancy_scale = 1.0,
            β_max=beta_max,
            maturity_half_time=maturity_half_time,
            maturity_power=maturity_power,
            location=state -> (Float64.(state) .- minima) ./ spans
        )
        discrepancy_unit = calibrate_discrepancy_scale(
            context, template, calibration_coefficients
        )

        score = eof_field_score(
            target;
            kernel_bandwidth=kernel_bandwidth,
            discrepancy_scale=discrepancy_unit,
            β_max=beta_max,
            maturity_half_time=maturity_half_time,
            maturity_power=maturity_power,
            location=state -> (Float64.(state) .- minima) ./ spans
        )
        @unpack mechanism, covariance_scale, optimizer_steps = mission[:filter][:proposal]
        proposal = Dict{Symbol, Any}(
            :mechanism => Symbol(mechanism),
            :covariance_scale => Float64(covariance_scale),
            :optimizer_steps => Int(optimizer_steps)
        )
        worlds = trial_worlds(mission, scenario)
        return Dict(
            :mission => mission,
            :scenario => scenario,
            :score => score,
            :proposal => proposal,
            :worlds => worlds
        )
    end
end

function parse_results(
    result,
    recovery_diagnostics,
    problem,
    observed,
    observed_trajectory,
    flow_directions,
    world,
    scenario,
    trial,
)
    @unpack coefficients, field = observed
    @unpack actual_distance = world
    @unpack context, roms, quadrature_rows, target_floor = scenario
    @unpack inferred_target_field,
            posterior_predictive_trajectory_discrepancy,
            world_rmse,
            target_rmse,
            particle_mmd,
            posterior_world_rmse,
            posterior_target_rmse,
            posterior_mmd,
            behavioral_mmd,
            ess_history,
            coefficient_spread,
            prior_field_rmse,
            posterior_field_rmse,
            prior_target_field_rmse,
            posterior_target_field_rmse = recovery_diagnostics

    truth_coefficients = coefficients
    truth_field = field
    full_weights = result.model.params.decomposition.weights
    truth_target_field = nonnegative_curl_target(
        truth_field,
        full_weights,
        target_floor,
    )
    discrepancy_cache = Dict{Symbol,Any}()
    horizon = length(problem.observations)
    trajectory_discrepancy = Dict(
        :prior => kernel_discrepancy(
            problem,
            horizon,
            context.model.ϕ,
            discrepancy_cache,
        ),
        :posterior_predictive => posterior_predictive_trajectory_discrepancy,
    )
    reported_diagnostics = Dict{Symbol,Any}()
    @pack! reported_diagnostics = world_rmse,
        target_rmse,
        particle_mmd,
        posterior_world_rmse,
        posterior_target_rmse,
        posterior_mmd,
        behavioral_mmd,
        ess_history,
        coefficient_spread,
        prior_field_rmse,
        posterior_field_rmse,
        prior_target_field_rmse,
        posterior_target_field_rmse

    Dict(
        :trial => trial,
        :prior_distance => actual_distance,
        :truth_field => truth_field,
        :truth_target_field => truth_target_field,
        :inferred_target_field => inferred_target_field,
        :truth_coefficients => truth_coefficients,
        :trajectory_discrepancy => trajectory_discrepancy,
        :flow_directions => flow_directions,
        :problem => problem,
        :result => result,
        :elapsed_times => observed_trajectory[:elapsed_times],
        :recovery_diagnostics => reported_diagnostics,
        :trajectory => wet_grid_locations(
            roms,
            quadrature_rows[observed_trajectory[:site_indices]],
        ),
    )
end

function run_trial(
    mission, scenario, score, proposal, flow_directions, world, trial
)
    observed_coefficients = world[:coefficients]
    observed = Dict(
        :coefficients => observed_coefficients,
        :field => SCRIBE.reconstruct_eof_field(
            scenario[:model]; coefficients=observed_coefficients,
        ),
    )
    seed = UInt64(mission[:trials][:seed]) + UInt64(2trial)

    observed_trajectory = simulate_observable_trajectory(
        mission,
        scenario,
        score,
        observed_coefficients;
        behavior_type=:ergodic,
    )
    problem = WorldInferenceProblem(
        context=scenario[:context], score=score,
        observations=observed_trajectory[:observations]
    )
    @unpack particles, ess_threshold, rejuvenation_steps = mission[:filter]
    recovery_diagnostics = construct_world_recovery_diagnostics_cache(
        problem,
        observed,
        scenario[:target_floor],
        particles;
        top_count=10,
    )
    result = infer_world(problem;
        n_particles=particles,
        ess_threshold=ess_threshold,
        rejuvenation_steps=rejuvenation_steps,
        diagnostics=recovery_diagnostics[:callbacks],
        proposal=proposal,
        rng=MersenneTwister(seed + 1)
    )
    return parse_results(
        result,
        recovery_diagnostics,
        problem,
        observed,
        observed_trajectory,
        flow_directions,
        world,
        scenario,
        trial,
    )
end

function main(mission_path::Union{String, Nothing}=nothing)
    if isnothing(mission_path)
        mission_path = joinpath(@__DIR__, "missions/", "som_comparison_multi_trial.json")
    end
    mission_info = load_mission_info(mission_path)
    let mission = mission_info[:mission],
        scenario = mission_info[:scenario],
        score = mission_info[:score],
        proposal = mission_info[:proposal],
        worlds = mission_info[:worlds]
        trials = [begin
            println("Running trial $(trial)/$(length(worlds))")
            run_trial(
                mission,
                scenario,
                score,
                proposal,
                world[:flow_directions],
                world,
                trial
            )
        end for (trial, world) in enumerate(worlds)]
        output = save_results(mission, scenario, trials)
        println("Saved mission results to $output")
    end
end
