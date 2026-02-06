module Utils

using Random, Statistics, LinearAlgebra
using BSON, JLD2

# shape_state_as_obs, KAgentPOMDP, MuEnv
using MuKumari
# ExperienceBuffer
using Crux

import ..Arrodes: MuEnvSpec, ScoreΠDist, RunPack
    
include("support.jl")
export nanmean, randcat, replace_nan_with_zero,
       _dims_to_bounds, _grid_from_mdp, xy_path_from_state_matrix, _goal_targets, _max_pairwise_dist, kworld_annotations

include("buffers.jl")
export mk_experience_buffer, alloc_buffer_dict, wrap_like, anonymize_buffer_location!, onehot_cols_to_aidx, data_cleaner

include("env.jl")
export build_kagent_pomdp, build_shared_menv, agent_params_from_mdp

include("dataset_io.jl")
export safe_get_obstacle_count, _normalize_run_payload, load_runpacks, select_skeleton_mdps

end