function clear!(cache::PlanningCache)
    empty!(cache.mdps)
    empty!(cache.artifacts)
    return cache
end

function _artifact_key(planner::AbstractPlanner, mdp, context::PlanningContext)
    scope = cache_scope(planner)
    identity = (context.hypothesis_id, hash(typeof(planner)), hash(mdp))
    scope === :hypothesis && return (:hypothesis, identity)
    scope === :initial_state && return (:initial_state, context.hypothesis_id,
        hash(typeof(planner)), hash(mdp), isempty(context.states) ? nothing : hash(first(context.states)))
    scope === :history && return (:history, context.hypothesis_id, context.timestep,
        hash(typeof(planner)), hash(mdp), hash(context.states), hash(context.actions))
    scope === :none && return nothing
    throw(ArgumentError("unsupported planner cache scope $(repr(scope))"))
end

function prepare_cached!(cache::PlanningCache, planner::AbstractPlanner, mdp,
                         context::PlanningContext)
    key = _artifact_key(planner, mdp, context)
    key === nothing && return prepare(planner, mdp, context)
    return get!(cache.artifacts, key) do
        prepare(planner, mdp, context)
    end
end
