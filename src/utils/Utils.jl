module Utils

using Random, Statistics, LinearAlgebra
import GeoInterface as GI
using MuKumari

Base.@kwdef struct MuEnvSpec
    M::Int = 3
    μ_order::Vector{Symbol} = [:sin, :exp, :lin]
end
    
include("support.jl")
export nanmean, randcat, replace_nan_with_zero,
       _dims_to_bounds, _grid_from_mdp, xy_path_from_state_matrix, _goal_targets, _max_pairwise_dist, kworld_annotations

include("env.jl")
export MuEnvSpec, build_kagent_pomdp, build_shared_menv, agent_params_from_mdp,
       onehot_cols_to_aidx

"""Convert one-hot action columns to one-based action indices."""
onehot_cols_to_aidx(actions::AbstractMatrix) = [argmax(view(actions, :, t)) for t in axes(actions, 2)]

end
