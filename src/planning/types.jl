abstract type AbstractPlanner end
abstract type AbstractPlanArtifact end
abstract type AbstractActionLikelihood end

"""A planner paired with the observation model used to score its behavior."""
struct BehaviorModel{P<:AbstractPlanner,L<:AbstractActionLikelihood}
    planner::P
    likelihood::L
end

"""Information available when preparing or querying a planner."""
@with_kw_noshow struct PlanningContext
    hypothesis_id::Symbol
    timestep::Int = 1
    states = Any[]
    actions = Any[]
    horizon::Int = 1
    rng = Random.default_rng()
    metadata = NamedTuple()
end

"""Mutable cache of MDPs and planner artifacts for named hypotheses."""
@with_kw_noshow mutable struct PlanningCache
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
@with_kw_noshow struct MovementNoiseLikelihood <: AbstractActionLikelihood
    n_transition_samples::Int = 64
    bandwidth::Float64 = 0.25
    action_epsilon::Float64 = 0.02
    state_distance::Union{Nothing,Function} = nothing
end
