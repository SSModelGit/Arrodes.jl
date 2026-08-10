function resolve_action(observed_action, action_list)
    if observed_action isa Integer
        return action_list[Int(observed_action)]
    end
    index = findfirst(isequal(observed_action), action_list)
    isnothing(index) ? nothing : action_list[index]
end

function planning_context(problem::ObjectiveInferenceProblem, index::Int, timestep::Int)
    hypothesis = problem.hypotheses[index]
    mdp = hypothesis_mdp(problem, index)
    states = Any[problem.state_adapter(mdp, observation, t)
                 for (t, observation) in enumerate(problem.states[1:timestep])]
    action_list = collect(POMDPs.actions(mdp))
    actions = Any[resolve_action(action, action_list)
                  for action in problem.actions[1:max(timestep - 1, 0)]]
    PlanningContext(
        hypothesis_id=hypothesis.id,
        timestep=timestep,
        states=states,
        actions=actions,
        horizon=problem.horizon,
        rng=MersenneTwister(hash((problem.seed, hypothesis.id, timestep), UInt(0))),
        metadata=hypothesis.metadata,
    )
end

function objective_loglikelihood(problem::ObjectiveInferenceProblem, index::Int, timestep::Int)
    hypothesis = problem.hypotheses[index]
    mdp = hypothesis_mdp(problem, index)
    context = planning_context(problem, index, timestep)
    action_list = collect(POMDPs.actions(mdp))
    observed_action = resolve_action(problem.actions[timestep], action_list)
    isnothing(observed_action) && return -Inf
    artifact = prepare_cached!(problem.planning_cache, hypothesis.behavior.planner, mdp, context)
    query_artifact = deepcopy(artifact)
    observation_loglikelihood(
        hypothesis.behavior.likelihood,
        hypothesis.behavior.planner,
        query_artifact,
        mdp,
        context.states[end],
        observed_action,
        action_list,
        context,
    )
end

function objective_loglikelihood(problem, index, timestep, cache)
    scores = get!(cache, :objective_scores) do
        Dict{Tuple{Int,Int},Float64}()
    end
    get!(scores, (index, timestep)) do
        objective_loglikelihood(problem, index, timestep)
    end
end

function hypothesis_artifact(problem::ObjectiveInferenceProblem, id::Symbol)
    index = hypothesis_index(problem, id)
    hypothesis = problem.hypotheses[index]
    mdp = hypothesis_mdp(problem, index)
    context = planning_context(problem, index, max(length(problem.actions), 1))
    prepare_cached!(problem.planning_cache, hypothesis.behavior.planner, mdp, context)
end
