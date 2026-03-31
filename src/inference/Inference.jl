module Inference

using LinearAlgebra, Statistics
using Gen: @gen, @trace, Distribution, UnknownChange, NoChange, categorical, choicemap, get_choice, get_choices, get_retval
using GenParticleFilters: pf_initialize, pf_rejuvenate!, pf_resample!, pf_update!, effective_sample_size, select, mh
using GenParticleFilters: get_traces, get_log_weights

using MuKumari: blindstart_KAgentState
using POMDPs: actions

import ..Priors
import ..RL
import ..Utils
import ..Arrodes: FourierDiscreteCfg, ScoreΠDist, RLConfig, InferenceConfig

include("gen_model.jl")
export gen_K, gen_mode_indices, gen_fourier_bank_fixed, inference_model

include("gen_model_continuous.jl")
export inference_model_continuous, extract_component_info, reconstruct_objective_from_trace

include("particle_filter.jl")
export particle_filter, extract_particle_component_info, best_particle

end