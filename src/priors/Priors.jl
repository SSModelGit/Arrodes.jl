module Priors

using LinearAlgebra, Statistics, Random
using Distributions: Categorical

using MuKumari

import ..Arrodes: FourierDiscreteCfg, ScoreΠDist, PriorDiscreteCfg, RBFDiscreteCfg
import ..Utils: randcat

include("fields.jl")
export K_probs,
    make_pomdp_objective_from_field,
    objective_grid_from_field,
    objective_grid_from_mdp

include("fourier.jl")
export freq_bin_support_and_probs,
    amp_bin_support_and_probs,
    phase_bin_support_and_probs,
    f_from_i,
    A_from_i,
    ϕ_from_i
export sample_fourier_key,
    decode_fourier_key,
    hamming_fourier_key,
    nearest_trained_key,
    make_fourier_scalar_field,
    objective_grid_from_key

include("rbf.jl")
export N_probs,
    center_bin_support_and_probs,
    amplitude_bin_support_and_probs
export sample_rbf_key,
    decode_rbf_key,
    hamming_rbf_key,
    nearest_trained_key_rbf,
    make_rbf_scalar_field,
    objective_grid_from_rbf_key

end
