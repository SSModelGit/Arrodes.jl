module Priors

using Random, LinearAlgebra, Statistics

using MuKumari

import ..Arrodes: FourierDiscreteCfg, ScoreΠDist

include("fields.jl")
export make_pomdp_objective_from_field, objective_grid_from_field, objective_grid_from_mdp

include("fourier.jl")
export K_probs, freq_bin_support_and_probs, amp_bin_support_and_probs, phase_bin_support_and_probs, f_from_i, A_from_i, ϕ_from_i
export sample_fourier_key, decode_fourier_key, hamming_fourier_key, nearest_trained_key, make_fourier_scalar_field, objective_grid_from_key

end