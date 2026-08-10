function planner_world_loglikelihood(problem::WorldInferenceProblem,
                                     evidence::PlannerWorldEvidence,
                                     timestep, coefficients)
    model = candidate_model(problem.context, coefficients)
    mdp = evidence.mdp_builder(model, evidence.objective)
    states = Any[evidence.state_adapter(mdp, problem.trajectory.states[t], t)
                 for t in 1:timestep]
    action_list = collect(POMDPs.actions(mdp))
    actions = Any[resolve_action(problem.trajectory.actions[t], action_list)
                  for t in 1:max(timestep - 1, 0)]
    observed_action = resolve_action(problem.trajectory.actions[timestep], action_list)
    isnothing(observed_action) && return -Inf
    context = PlanningContext(
        hypothesis_id=:world,
        timestep=timestep,
        states=states,
        actions=actions,
        horizon=problem.horizon,
        rng=MersenneTwister(hash((timestep, coefficients), UInt(0))),
        metadata=evidence.metadata,
    )
    artifact = prepare(evidence.behavior.planner, mdp, context)
    observation_loglikelihood(
        evidence.behavior.likelihood,
        evidence.behavior.planner,
        artifact,
        mdp,
        states[end],
        observed_action,
        action_list,
        context,
    )
end

function cached_world_energy(problem, evidence::DirectErgodicEvidence, t, coefficients, cache)
    history = get!(cache, :world_energy_cache) do
        Dict{Int,Any}()
    end
    energies = get!(history, t) do
        Dict{Tuple,WorldEnergy}()
    end
    newest = maximum(keys(history))
    for old_t in collect(keys(history))
        old_t < newest - 1 && delete!(history, old_t)
    end
    get!(energies, Tuple(coefficients)) do
        world_energy(problem, evidence, t, coefficients, cache)
    end
end

world_compatibility(problem, evidence::DirectErgodicEvidence, t, coefficients, cache) =
    -cached_world_energy(problem, evidence, t, coefficients, cache).total

function world_compatibility(problem, evidence::PlannerWorldEvidence, t, coefficients, cache)
    sum((planner_world_loglikelihood(problem, evidence, step, coefficients) for step in 1:t);
        init=0.0)
end

function world_compatibility(problem, evidence::CompositeBehaviorEvidence, t, coefficients, cache)
    sum(world_compatibility(problem, component, t, coefficients, cache)
        for component in evidence.components)
end

evidence_identifiability(evidence::DirectErgodicEvidence) =
    evidence.mean_dependent ? :mean_dependent : :covariance_only
evidence_identifiability(::PlannerWorldEvidence) = :planner_dependent
function evidence_identifiability(evidence::CompositeBehaviorEvidence)
    all(component -> evidence_identifiability(component) === :covariance_only,
        evidence.components) ? :covariance_only : :mean_dependent
end
