using JSON: parsefile
using LinearAlgebra: BLAS, Symmetric, cholesky, diag, dot, norm
using MAT: matwrite
using SCRIBE: eof_coefficients, reconstruct_eof_field
using Statistics: mean

include("som_comparison_multi_trial.jl")

BLAS.set_num_threads(1)

function simplex_fit(gram, cross; iterations=600, tolerance=1e-10)
    weights = zeros(size(gram, 1))
    weights[argmin(diag(gram) .- 2 .* cross)] = 1.0

    for _ in 1:iterations
        gradient = gram * weights - cross
        vertex = argmin(gradient)
        gap = dot(weights, gradient) - gradient[vertex]
        gap <= tolerance && break

        direction = -weights
        direction[vertex] += 1.0
        curvature = dot(direction, gram * direction)
        step = curvature <= eps(Float64) ?
            1.0 : clamp(gap / curvature, 0.0, 1.0)
        weights .+= step .* direction
    end

    weights
end

function best_snapshots(
    candidates, scores, count, snapshots, used, snapshot_gap,
)
    selected = Int[]
    for index in sort(candidates; by=index -> scores[index])
        index in used && continue
        all(abs(snapshots[index] - snapshots[other]) >= snapshot_gap
            for other in selected) || continue
        push!(selected, index)
        length(selected) == count && break
    end
    selected
end

mission_path = isempty(ARGS) ? joinpath(
    @__DIR__, "missions", "som_comparison_multi_trial.json",
) : first(ARGS)
mission = parsefile(mission_path; dicttype=Dict{Symbol,Any})
prepared = prepare_mission(mission)
scenario = prepared[:scenario]

model = scenario[:eof_model]
params = model.params
roms = scenario[:roms]
prior_factor = cholesky(Symmetric(scenario[:prior_covariance])).L
som_coefficients = scenario[:som_coefficients]
som_whitened = prior_factor \ som_coefficients
som_whitened_mean = vec(mean(som_whitened; dims=2))
som_anomalies = som_whitened .- som_whitened_mean
som_whitened_covariance = som_anomalies * som_anomalies' /
    (size(som_anomalies, 2) - 1)

snapshots = collect(scenario[:validation_start]:size(roms[:data], 2))
snapshot_fields = view(roms[:data], :, snapshots)
snapshot_coefficients = eof_coefficients(params, snapshot_fields)
snapshot_whitened = prior_factor \ snapshot_coefficients
snapshot_prior_distances = [
    norm(view(snapshot_whitened, :, index))
    for index in axes(snapshot_whitened, 2)
]

coefficient_gram = som_whitened' * som_whitened
spatial_weights = scenario[:spatial_weights]
normalized_weights = spatial_weights ./ sum(spatial_weights)
weighted_som_fields = sqrt.(normalized_weights) .* scenario[:som_fields]
weighted_snapshot_fields = sqrt.(normalized_weights) .* snapshot_fields
field_gram = weighted_som_fields' * weighted_som_fields

snapshot_som_hull_distances = zeros(length(snapshots))
snapshot_som_outside_sigmas = zeros(length(snapshots))
snapshot_som_rmse_floors = zeros(length(snapshots))
snapshot_eof_rmse_floors = zeros(length(snapshots))
snapshot_hull_projections = similar(snapshot_whitened)

println("Evaluating $(length(snapshots)) held-out ROMS worlds ...")
for index in eachindex(snapshots)
    whitened = view(snapshot_whitened, :, index)
    coefficient_weights = simplex_fit(
        coefficient_gram,
        som_whitened' * whitened,
    )
    hull_projection = som_whitened * coefficient_weights
    difference = whitened - hull_projection
    hull_distance = norm(difference)
    direction = difference ./ max(hull_distance, eps(Float64))
    directional_sigma = sqrt(max(
        dot(direction, som_whitened_covariance * direction),
        eps(Float64),
    ))

    field = view(weighted_snapshot_fields, :, index)
    field_weights = simplex_fit(field_gram, weighted_som_fields' * field)
    field_residual = weighted_som_fields * field_weights - field
    reconstructed = reconstruct_eof_field(
        model; coefficients=view(snapshot_coefficients, :, index),
    )

    snapshot_hull_projections[:, index] = hull_projection
    snapshot_som_hull_distances[index] = hull_distance
    snapshot_som_outside_sigmas[index] = hull_distance / directional_sigma
    snapshot_som_rmse_floors[index] = norm(field_residual)
    snapshot_eof_rmse_floors[index] = weighted_rmse(
        reconstructed,
        view(snapshot_fields, :, index),
        spatial_weights,
    )
end

worlds_per_level = mission[:trials][:worlds_per_level]
snapshot_gap = mission[:trials][:snapshot_gap]
distance_targets = collect(range(
    minimum(snapshot_som_hull_distances),
    maximum(snapshot_som_hull_distances);
    length=10,
))
selected_snapshots = Int[]
selected_levels = Int[]
selected_coefficients = Vector{Float64}[]
selected_prior_distances = Float64[]
selected_som_hull_distances = Float64[]
used = Set{Int}()
real_worlds = Dict{Int,Vector{Int}}()

for level in reverse(eachindex(distance_targets))
    target_distance = distance_targets[level]
    scores = abs.(snapshot_som_hull_distances .- target_distance)
    real = best_snapshots(
        eachindex(snapshots), scores, worlds_per_level,
        snapshots, used, snapshot_gap,
    )
    union!(used, real)
    real_worlds[level] = real
end

for level in eachindex(distance_targets)
    real = real_worlds[level]
    for index in real
        push!(selected_snapshots, snapshots[index])
        push!(selected_levels, level)
        push!(selected_coefficients, Vector{Float64}(
            view(snapshot_coefficients, :, index),
        ))
        push!(selected_prior_distances, snapshot_prior_distances[index])
        push!(selected_som_hull_distances,
            snapshot_som_hull_distances[index])
    end
end

selected_coefficient_matrix = hcat(selected_coefficients...)
level_som_hull_distances = [
    mean(selected_som_hull_distances[selected_levels .== level])
    for level in eachindex(distance_targets)
]

output = normpath(joinpath(@__DIR__, mission[:world_space_data]))
mkpath(dirname(output))
matwrite(output, Dict(
    "som_coefficients" => som_coefficients,
    "som_whitened" => som_whitened,
    "som_whitened_mean" => som_whitened_mean,
    "som_whitened_covariance" => som_whitened_covariance,
    "snapshot_ids" => snapshots,
    "snapshot_coefficients" => snapshot_coefficients,
    "snapshot_prior_distances" => snapshot_prior_distances,
    "snapshot_som_hull_projections_whitened" => snapshot_hull_projections,
    "snapshot_som_hull_distances" => snapshot_som_hull_distances,
    "snapshot_som_outside_sigmas" => snapshot_som_outside_sigmas,
    "snapshot_som_rmse_floors" => snapshot_som_rmse_floors,
    "snapshot_eof_rmse_floors" => snapshot_eof_rmse_floors,
    "selected_snapshot_ids" => selected_snapshots,
    "selected_levels" => selected_levels,
    "selected_coefficients" => selected_coefficient_matrix,
    "selected_prior_distances" => selected_prior_distances,
    "selected_som_hull_distances" => selected_som_hull_distances,
    "level_som_hull_distances" => level_som_hull_distances,
))

println(
    "Saved $(length(selected_levels)) selected worlds to $output; " *
    "held-out ROMS hull distances span " *
    "$(round.(extrema(snapshot_som_hull_distances); digits=2)) prior σ.",
)
