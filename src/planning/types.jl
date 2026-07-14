abstract type AbstractPlanner end
abstract type AbstractPlanArtifact end
abstract type AbstractActionLikelihood end

"""A planner paired with the observation model used to score its behavior."""
struct BehaviorModel{P<:AbstractPlanner,L<:AbstractActionLikelihood}
    planner::P
    likelihood::L
end

"""Information available when preparing or querying a planner."""
Base.@kwdef struct PlanningContext{S,A,R,M}
    hypothesis_id::Symbol
    timestep::Int = 1
    states::S = Any[]
    actions::A = Any[]
    horizon::Int = 1
    rng::R = Random.default_rng()
    metadata::M = NamedTuple()
end

"""Mutable cache of MDPs and planner artifacts for named hypotheses."""
Base.@kwdef mutable struct PlanningCache
    mdps::Dict{Symbol,Any} = Dict{Symbol,Any}()
    artifacts::Dict{Any,Any} = Dict{Any,Any}()
end

struct PolicyArtifact{S,P} <: AbstractPlanArtifact
    solver::S
    policy::P
end

struct OpenLoopArtifact{S,A,R} <: AbstractPlanArtifact
    states::S
    actions::A
    raw::R
end

struct CallbackArtifact{T} <: AbstractPlanArtifact
    value::T
end

"""Score deterministic plans using both action noise and the MDP transition noise."""
Base.@kwdef struct MovementNoiseLikelihood{D} <: AbstractActionLikelihood
    n_transition_samples::Int = 64
    bandwidth::Float64 = 0.25
    action_epsilon::Float64 = 0.02
    state_distance::D = nothing
end
