module Inference

using Random, LinearAlgebra, Statistics
using Gen: @gen, @trace, Distribution, UnknownChange, NoChange, categorical, choicemap, get_choice, get_choices, get_retval
using GenParticleFilters: pf_initialize, pf_rejuvenate!, pf_resample!, pf_update!, effective_sample_size, select, mh
using GenParticleFilters: get_traces, get_log_weights

import ..Priors
import ..Core
import ..Arrodes: FourierDiscreteCfg, ScoreΠDist

include("gen_model.jl")
export gen_K, gen_mode_indices, gen_fourier_bank_fixed, inference_model

include("particle_filter.jl")
export particle_filter

end