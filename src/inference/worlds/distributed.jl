export AgentBehaviorEvidence, DistributedTrajectoryObservation
export AbstractDistributedBeliefModel, SharedWorldBelief
export AgentSpecificWorldBeliefs, HierarchicalWorldBelief
export AbstractEvidenceCoupling, ConditionallyIndependentEvidence
export JointBehaviorCompatibility, DistributedWorldInferenceProblem
export AgentWorldCoefficients, HierarchicalWorldCoefficients
export DistributedWorldPosteriorSummary, infer_distributed_world

abstract type AbstractDistributedBeliefModel end
struct SharedWorldBelief <: AbstractDistributedBeliefModel end
struct AgentSpecificWorldBeliefs <: AbstractDistributedBeliefModel end

@with_kw struct HierarchicalWorldBelief <: AbstractDistributedBeliefModel
    offset_scale::Float64 = 0.25
end

abstract type AbstractEvidenceCoupling end
struct ConditionallyIndependentEvidence <: AbstractEvidenceCoupling end

@with_kw_noshow struct JointBehaviorCompatibility <: AbstractEvidenceCoupling
    logcompatibility::Function
end

@with_kw_noshow mutable struct AgentBehaviorEvidence
    id::Symbol
    evidence::AbstractBehaviorEvidence
    trajectory::BehaviorTrajectory = BehaviorTrajectory()
    planning_cache::PlanningCache = PlanningCache()
end

@with_kw struct DistributedTrajectoryObservation
    agent_id::Symbol
    observation::TrajectoryObservation
end

@with_kw_noshow struct AgentWorldCoefficients
    values::Vector{Vector{Float64}}
end

@with_kw_noshow struct HierarchicalWorldCoefficients
    common::Vector{Float64}
    offsets::Vector{Vector{Float64}}
end

@with_kw_noshow mutable struct DistributedWorldInferenceProblem <: AbstractBehaviorInferenceProblem
    context::SCRIBEWorldContext
    agents::Vector{AgentBehaviorEvidence}
    belief_model::AbstractDistributedBeliefModel
    coupling::AbstractEvidenceCoupling
    latest_agent::Int = 0
    horizon::Int = 1
end

@with_kw_noshow struct DistributedWorldPosteriorSummary
    stage::InferenceStage
    agent_ids::Vector{Symbol}
    agent_means::Vector{Vector{Float64}}
    common_mean::Union{Nothing,Vector{Float64}}
end

function distributed_agent(problem, id)
    index = findfirst(agent -> agent.id === id, problem.agents)
    isnothing(index) && throw(KeyError(id))
    index
end

function observe!(problem::DistributedWorldInferenceProblem,
                  distributed::DistributedTrajectoryObservation)
    index = distributed_agent(problem, distributed.agent_id)
    observation = distributed.observation
    observation.environment_time == problem.context.model_time ||
        throw(ArgumentError("distributed static worlds freeze SCRIBE model time"))
    trajectory = problem.agents[index].trajectory
    push!(trajectory.states, observation.state)
    push!(trajectory.actions, observation.action)
    push!(trajectory.dwell_times, observation.dwell_time)
    push!(trajectory.environment_times, observation.environment_time)
    problem.latest_agent = index
    problem
end

function validate_problem(problem::DistributedWorldInferenceProblem)
    isempty(problem.agents) && throw(ArgumentError("distributed inference needs an agent"))
    ids = getfield.(problem.agents, :id)
    length(ids) == length(unique(ids)) ||
        throw(ArgumentError("distributed agent IDs must be unique"))
    nothing
end

distributed_coefficients(::SharedWorldBelief, value, index) = value
distributed_coefficients(::AgentSpecificWorldBeliefs, value::AgentWorldCoefficients, index) =
    value.values[index]
distributed_coefficients(::HierarchicalWorldBelief, value::HierarchicalWorldCoefficients, index) =
    value.common + value.offsets[index]

function initial_proposal(problem::DistributedWorldInferenceProblem, rng, cache)
    context = problem.context
    model = problem.belief_model
    value, logdensity = @match model begin
        ::SharedWorldBelief => begin
            coefficients = gaussian_sample(rng, context.prior_mean, context.prior_covariance)
            coefficients, gaussian_logdensity(
                coefficients, context.prior_mean, context.prior_covariance,
            )
        end
        ::AgentSpecificWorldBeliefs => begin
            values = [gaussian_sample(rng, context.prior_mean, context.prior_covariance)
                      for _ in problem.agents]
            AgentWorldCoefficients(values=values),
                sum(gaussian_logdensity(v, context.prior_mean, context.prior_covariance)
                    for v in values)
        end
        hierarchy::HierarchicalWorldBelief => begin
            common = gaussian_sample(rng, context.prior_mean, context.prior_covariance)
            covariance = hierarchy.offset_scale^2 .* context.prior_covariance
            offsets = [gaussian_sample(rng, zeros(length(common)), covariance)
                       for _ in problem.agents]
            HierarchicalWorldCoefficients(common=common, offsets=offsets),
                gaussian_logdensity(common, context.prior_mean, context.prior_covariance) +
                sum(gaussian_logdensity(offset, zeros(length(common)), covariance)
                    for offset in offsets)
        end
    end
    (value=value, trace=WorldTrace(), logdensity=logdensity)
end

function distributed_logprior(problem, value)
    context = problem.context
    @match problem.belief_model begin
        ::SharedWorldBelief => gaussian_logdensity(
            value, context.prior_mean, context.prior_covariance,
        )
        ::AgentSpecificWorldBeliefs => sum(
            gaussian_logdensity(v, context.prior_mean, context.prior_covariance)
            for v in value.values
        )
        hierarchy::HierarchicalWorldBelief => begin
            covariance = hierarchy.offset_scale^2 .* context.prior_covariance
            gaussian_logdensity(value.common, context.prior_mean, context.prior_covariance) +
            sum(gaussian_logdensity(offset, zeros(length(value.common)), covariance)
                for offset in value.offsets)
        end
    end
end

function agent_problem(problem, index)
    agent = problem.agents[index]
    WorldInferenceProblem(
        context=problem.context,
        evidence=agent.evidence,
        trajectory=agent.trajectory,
        planning_cache=agent.planning_cache,
        horizon=problem.horizon,
    )
end

function distributed_compatibility(problem, value, omit_latest, cache,
                                   ::ConditionallyIndependentEvidence)
    sum(eachindex(problem.agents); init=0.0) do index
        local_problem = agent_problem(problem, index)
        t = length(problem.agents[index].trajectory.states) -
            (omit_latest && index == problem.latest_agent)
        coefficients = distributed_coefficients(problem.belief_model, value, index)
        agent_cache = get!(cache, Symbol("agent_", problem.agents[index].id)) do
            Dict{Symbol,Any}()
        end
        world_compatibility(
            local_problem, problem.agents[index].evidence, t, coefficients, agent_cache,
        )
    end
end

function distributed_compatibility(problem, value, omit_latest, cache,
                                   coupling::JointBehaviorCompatibility)
    coupling.logcompatibility(problem, value, omit_latest, cache)
end

function logtarget(problem::DistributedWorldInferenceProblem, stage::InferenceStage,
                   value, cache)
    full = distributed_compatibility(problem, value, false, cache, problem.coupling)
    previous = distributed_compatibility(problem, value, true, cache, problem.coupling)
    distributed_logprior(problem, value) + (1 - stage.λ) * previous + stage.λ * full
end

function distributed_agent_values(problem, particle)
    [distributed_coefficients(problem.belief_model, particle.value, index)
     for index in eachindex(problem.agents)]
end

function summarize(problem::DistributedWorldInferenceProblem, cloud::ParticleCloud,
                   stage::InferenceStage, cache)
    log_weights = [particle.log_weight for particle in cloud.particles]
    weights = exp.(log_weights .- logsumexp(log_weights))
    agent_means = [sum(weight .* distributed_agent_values(problem, particle)[index]
                       for (weight, particle) in zip(weights, cloud.particles))
                   for index in eachindex(problem.agents)]
    common_mean = problem.belief_model isa HierarchicalWorldBelief ?
        sum(weight .* particle.value.common
            for (weight, particle) in zip(weights, cloud.particles)) :
        problem.belief_model isa SharedWorldBelief ? first(agent_means) : nothing
    DistributedWorldPosteriorSummary(
        stage=stage,
        agent_ids=[agent.id for agent in problem.agents],
        agent_means=agent_means,
        common_mean=common_mean,
    )
end

posterior(result::SMCResult{<:SequentialState{<:DistributedWorldInferenceProblem}}) =
    last(result.state.summaries)
function infer_distributed_world(problem::DistributedWorldInferenceProblem, observations,
                                 config::SMCConfig)
    problem.horizon = length(observations)
    run_smc(problem, observations, config)
end
