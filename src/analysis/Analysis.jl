module Analysis

using Random, LinearAlgebra, Statistics
using BSON, JLD2

using MuKumari

using CUDA, cuDNN
using Crux
using Flux

using Gen: @gen, @trace, Distribution, UnknownChange, NoChange, categorical, choicemap, get_choice, get_choices, get_retval
using GenParticleFilters: pf_initialize, pf_rejuvenate!, pf_resample!, pf_update!, effective_sample_size, select, mh
using GenParticleFilters: get_traces, get_log_weights

using ..Utils
using ..Priors
using ..Core
using ..Inference
import ..Arrodes: RunPack

include("metrics.jl")
export pf_degeneracy, objective_recon_metrics, policy_match_acc

include("summarize.jl")
export eval_pack, summarize_eval, pack_feat, diversify_packs,
       summarize_ablation, degmask_from_summary, cache_record_for_eval, best_eval_by_accuracy

include("ablations.jl")
export eval_ablation_mdp, eval_ablation_from_cache, run_ablation_suite, ablation_main,
       reconstruct_mdp_from_cache, generate_and_cache_ablation_data, load_ablation_cache

end