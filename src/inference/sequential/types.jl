export AbstractBehaviorInferenceProblem, AbstractBehaviorEvidence
export AbstractPairedKernel, AbstractInvariantMove, AbstractStageScheduler
export InferenceStage, WeightedParticle, ParticleCloud, MoveRecord
export StageAncestry, StageDiagnostics, SequentialState, SMCResult, SMCConfig
export OneStagePerObservation, IdentityKernel, NoInvariantMove
export initial_stage, stages_for_observation, logtarget, initial_proposal
export propose, step, summarize, observe!
export validate_problem

abstract type AbstractBehaviorInferenceProblem end
abstract type AbstractBehaviorEvidence end
abstract type AbstractPairedKernel end
abstract type AbstractInvariantMove end
abstract type AbstractStageScheduler end

@with_kw struct InferenceStage
    observation::Int = 0
    bridge::Int = 0
    λ::Float64 = observation == 0 ? 0.0 : 1.0
    environment_time::Int = 1
end

@with_kw_noshow mutable struct WeightedParticle{X,T}
    value::X
    trace::T
    log_weight::Float64
    lineage::Int
    branch::Symbol = :initial
end

@with_kw_noshow mutable struct ParticleCloud{P}
    particles::Vector{P}
    stage::InferenceStage
    log_normalizer::Float64 = 0.0
    next_lineage::Int = length(particles) + 1
end

"""A paired proposal and every density term used by its SMC-P3 correction."""
@with_kw_noshow struct MoveRecord{X,M}
    value::X
    log_forward::Float64
    log_backward::Float64
    log_jacobian::Float64 = 0.0
    branch::Symbol = :identity
    metadata::M = NamedTuple()
end

@with_kw_noshow struct StageAncestry
    stage::InferenceStage
    proposal_parents::Vector{Int}
    resampling_parents::Vector{Int}
    invariant_parents::Vector{Int}
    branches::Vector{Symbol}
end

@with_kw struct StageDiagnostics
    stage::InferenceStage
    ess::Float64
    cess::Float64
    log_normalizer_increment::Float64
    resampled::Bool
    invariant_acceptance::Float64 = 0.0
end

@with_kw_noshow mutable struct SequentialState{P,C,R}
    problem::P
    cloud::C
    rng::R
    cache::Dict{Symbol,Any} = Dict{Symbol,Any}()
    ancestry::Vector{StageAncestry} = StageAncestry[]
    diagnostics::Vector{StageDiagnostics} = StageDiagnostics[]
    summaries::Vector{Any} = Any[]
end

@with_kw_noshow struct SMCResult{S}
    state::S
end

@with_kw_noshow struct SMCConfig
    n_particles::Int = 256
    ess_threshold::Float64 = 0.5
    paired_moves_per_stage::Int = 1
    scheduler::AbstractStageScheduler = OneStagePerObservation()
    kernel::AbstractPairedKernel = IdentityKernel()
    invariant_move::AbstractInvariantMove = NoInvariantMove()
    invariant_steps::Int = 0
    resampling::Symbol = :systematic
    seed::UInt64 = 0x6172_726f_6465_7301
end

struct OneStagePerObservation <: AbstractStageScheduler end
struct IdentityKernel <: AbstractPairedKernel end
struct NoInvariantMove <: AbstractInvariantMove end

initial_stage(::AbstractBehaviorInferenceProblem) = InferenceStage()
validate_problem(::AbstractBehaviorInferenceProblem) = nothing

function stages_for_observation(
    ::OneStagePerObservation,
    problem::AbstractBehaviorInferenceProblem,
    observation,
    cloud::ParticleCloud,
    cache,
)
    [InferenceStage(
        observation=cloud.stage.observation + 1,
        bridge=1,
        λ=1.0,
        environment_time=cloud.stage.environment_time,
    )]
end

function logtarget(problem::AbstractBehaviorInferenceProblem, stage::InferenceStage, value, cache)
    throw(MethodError(logtarget, (problem, stage, value, cache)))
end

function initial_proposal(problem::AbstractBehaviorInferenceProblem, rng, cache)
    throw(MethodError(initial_proposal, (problem, rng, cache)))
end

function propose(kernel::AbstractPairedKernel, problem::AbstractBehaviorInferenceProblem,
                 old_stage::InferenceStage, new_stage::InferenceStage,
                 particle::WeightedParticle, rng, cache)
    throw(MethodError(propose, (kernel, problem, old_stage, new_stage, particle, rng, cache)))
end

function step(move::AbstractInvariantMove, problem::AbstractBehaviorInferenceProblem,
              stage::InferenceStage, particle::WeightedParticle, rng, cache)
    throw(MethodError(step, (move, problem, stage, particle, rng, cache)))
end

summarize(problem::AbstractBehaviorInferenceProblem, cloud::ParticleCloud,
          stage::InferenceStage, cache) = nothing

function observe!(problem::AbstractBehaviorInferenceProblem, observation)
    throw(MethodError(observe!, (problem, observation)))
end
