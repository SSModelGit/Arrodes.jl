module RL

using LinearAlgebra, Statistics
using Match: @match
using POMDPs, POMDPTools, POMDPLinter, MCTS

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

using ..Utils
using ..Priors
import ..Arrodes: ScoreΠDist, RLConfig, InferenceConfig

include("solvers.jl")
export beliefstate_for_pomdp,
    van_solver,
    dpw_solver,
    mcts_solver,
    deep_q_solver,
    deep_q_metrics,
    solver_from_type,
    quick_policy_compute_for_pomdp,
    quick_IQL

include("scoredist.jl")
export get_proposal_names,
    get_proposal_prior,
    get_idxable_proposal_prior_list,
    π_alist,
    π_a_1hot,
    π_a_1hotall,
    ensure_mdp!,
    get_𝒮_proposal,
    get_π_proposal,
    store_π_iql,
    get_π_iql
include("training.jl")

include("policies.jl")
export proposal_boltzmann

end
