module Analysis

using Random, LinearAlgebra, Statistics, ArgCheck
using BSON, JLD2, Dates

using MuKumari

using CUDA, cuDNN
using Crux
using Flux

using Gen:
    @gen,
    @trace,
    Distribution,
    UnknownChange,
    NoChange,
    categorical,
    choicemap,
    get_choice,
    get_choices,
    get_retval
using GenParticleFilters:
    pf_initialize,
    pf_rejuvenate!,
    pf_resample!,
    pf_update!,
    effective_sample_size,
    select,
    mh
using GenParticleFilters: get_traces, get_log_weights

import ..Utils
import ..Priors
import ..RL
import ..Inference
import ..Arrodes: FourierDiscreteCfg, ScoreΠDist, MuEnvSpec, RunPack

include("metrics.jl")
export pf_degeneracy, objective_recon_metrics, policy_match_acc

include("summarize.jl")
export eval_pack,
    summarize_eval,
    pack_feat,
    diversify_packs,
    summarize_ablation,
    degmask_from_summary,
    cache_record_for_eval,
    best_eval_by_accuracy

include("ablations.jl")
export eval_ablation_from_cache,
    ablation_main,
    reconstruct_mdp_from_cache,
    generate_and_cache_ablation_data,
    load_ablation_cache,
    write_wholesale_metadata
end
