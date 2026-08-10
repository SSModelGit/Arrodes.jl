export SCRIBEWorldContext, TrajectoryObservation, BehaviorTrajectory
export DirectErgodicEvidence, PlannerWorldEvidence, CompositeBehaviorEvidence
export GaussianDiscrepancyKernel, WorldEnergyConfig, WorldEnergy
export WorldInferenceProblem, WorldTrace, WorldMoveRecord, WorldPosteriorSummary
export CandidateWorldScore, WorldRecoveryDiagnostics
export candidate_model, candidate_field, posterior_coefficient_moments

@with_kw_noshow struct SCRIBEWorldContext
    model::SCRIBE.EOFClimateModel
    information::SCRIBE.KFEnvInfo
    prior_mean::Vector{Float64}
    prior_covariance::Matrix{Float64}
    quadrature::Matrix{Float64}
    quadrature_weights::Vector{Float64}
    quadrature_mean::Vector{Float64}
    quadrature_basis::Matrix{Float64}
    model_time::Int
end

@with_kw struct TrajectoryObservation
    state
    action = nothing
    dwell_time::Float64 = 1.0
    environment_time::Int = 1
end

@with_kw_noshow mutable struct BehaviorTrajectory
    states::Vector{Any} = Any[]
    actions::Vector{Any} = Any[]
    dwell_times::Vector{Float64} = Float64[]
    environment_times::Vector{Int} = Int[]
end

abstract type AbstractDiscrepancyKernel end

@with_kw struct GaussianDiscrepancyKernel <: AbstractDiscrepancyKernel
    bandwidth::Float64
end

@with_kw_noshow struct WorldEnergyConfig
    discrepancy_scale::Float64
    reward_scale::Float64
    reward_reference::Float64 = 0.0
    β_max::Float64 = 1.0
    maturity_half_time::Float64 = 20.0
    maturity_power::Float64 = 2.0
    ucb_scale::Float64 = 1.0
    mixture_time::Float64 = 20.0
    evaluation::Symbol = :combined
end

@with_kw struct WorldEnergy
    total::Float64
    discrepancy::Float64
    mean_reward::Float64
    scaled_discrepancy::Float64
    scaled_reward::Float64
    maturity::Float64
    β::Float64
    reward_weight::Float64
    discrepancy_weight::Float64
end

@with_kw_noshow struct DirectErgodicEvidence <: AbstractBehaviorEvidence
    location::Function
    reward::Union{Nothing,Function} = nothing
    reward_gradient::Union{Nothing,Function} = nothing
    importance::Function
    target_jacobian::Union{Nothing,Function} = nothing
    kernel::AbstractDiscrepancyKernel
    energy::WorldEnergyConfig
    mean_dependent::Bool = true
end

@with_kw_noshow struct PlannerWorldEvidence <: AbstractBehaviorEvidence
    objective
    mdp_builder::Function
    behavior::BehaviorModel
    state_adapter::Function = (mdp, state, timestep) -> state
    metadata = NamedTuple()
end

@with_kw_noshow struct CompositeBehaviorEvidence <: AbstractBehaviorEvidence
    components::Vector{AbstractBehaviorEvidence}
end

@with_kw_noshow mutable struct WorldInferenceProblem <: AbstractBehaviorInferenceProblem
    context::SCRIBEWorldContext
    evidence::AbstractBehaviorEvidence
    trajectory::BehaviorTrajectory = BehaviorTrajectory()
    planning_cache::PlanningCache = PlanningCache()
    horizon::Int = 1
end

@with_kw struct WorldMoveRecord
    stage::InferenceStage
    branch::Symbol
    transport_fraction::Float64
    log_forward::Float64
    log_backward::Float64
end

"""Compact proposal trace retained with a world particle.

Coefficient histories live in the global ancestry record.  This trace retains the
SMC-P3 choices needed to audit how the current particle was transported without
copying dense metrics or complete coefficient paths into every particle.
"""
@with_kw_noshow mutable struct WorldTrace
    moves::Vector{WorldMoveRecord} = WorldMoveRecord[]
end

@with_kw_noshow struct WorldPosteriorSummary
    stage::InferenceStage
    coefficient_mean::Vector{Float64}
    coefficient_covariance::Matrix{Float64}
    map_mean::Vector{Float64}
    map_variance::Vector{Float64}
    mean_energy::Float64
    posterior_prior_kl::Float64
    contraction::Float64
    identifiability::Symbol
end

@with_kw struct CandidateWorldScore
    id::Symbol
    energy::Float64
    discrepancy::Float64
    mean_reward::Float64
end

@with_kw struct WorldRecoveryDiagnostics
    timestep::Int
    coefficient_rmse::Float64
    posterior_mahalanobis::Float64
    map_rmse::Float64
    map_correlation::Float64
    target_discrepancy::Float64
    marginal_coverage::Float64
    ess::Float64
    contraction::Float64
end
