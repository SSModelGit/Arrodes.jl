"""
    cache_scope(planner) -> Symbol

Declare when a prepared artifact may be reused. Valid scopes are `:hypothesis`,
`:initial_state`, `:history`, and `:none`.
"""
cache_scope(::AbstractPlanner) = :hypothesis

function prepare(planner::AbstractPlanner, mdp, context::PlanningContext)
    throw(MethodError(prepare, (planner, mdp, context)))
end

function planned_action(planner::AbstractPlanner, artifact::AbstractPlanArtifact, mdp, state,
                        context::PlanningContext)
    throw(MethodError(planned_action, (planner, artifact, mdp, state, context)))
end

action_scores(::AbstractPlanner, ::AbstractPlanArtifact, mdp, state, context::PlanningContext) = nothing

function _default_rollout(planner::AbstractPlanner, artifact::AbstractPlanArtifact, mdp,
                          initial_state, horizon::Integer, context::PlanningContext)
    states = Any[initial_state]
    actions_taken = Any[]
    state = initial_state
    for t in 1:horizon
        local_context = PlanningContext(
            hypothesis_id = context.hypothesis_id,
            timestep = t,
            states = states,
            actions = actions_taken,
            horizon = horizon,
            rng = context.rng,
            metadata = context.metadata,
        )
        action = planned_action(planner, artifact, mdp, state, local_context)
        push!(actions_taken, action)
        transition = POMDPs.gen(mdp, state, action, context.rng)
        state = transition.sp
        push!(states, state)
        POMDPs.isterminal(mdp, state) && break
    end
    return (states = states, actions = actions_taken)
end

rollout(planner::AbstractPlanner, artifact::AbstractPlanArtifact, mdp, initial_state,
        horizon::Integer, context::PlanningContext) =
    _default_rollout(planner, artifact, mdp, initial_state, horizon, context)

function action_distribution(likelihood::AbstractActionLikelihood, planner::AbstractPlanner,
                             artifact::AbstractPlanArtifact, mdp, state, action_list,
                             context::PlanningContext)
    throw(MethodError(action_distribution,
        (likelihood, planner, artifact, mdp, state, action_list, context)))
end

function _call_factory(factory, mdp, context)
    applicable(factory, mdp, context) && return factory(mdp, context)
    applicable(factory, mdp) && return factory(mdp)
    applicable(factory) && return factory()
    throw(ArgumentError("planner factory must accept (mdp, context), (mdp), or no arguments"))
end

function _validate_distribution(probabilities, n_actions::Integer)
    p = Float64.(vec(probabilities))
    length(p) == n_actions || throw(DimensionMismatch(
        "action distribution has $(length(p)) entries; expected $n_actions"))
    all(isfinite, p) || throw(ArgumentError("action probabilities must be finite"))
    all(x -> x >= 0, p) || throw(ArgumentError("action probabilities must be nonnegative"))
    total = sum(p)
    total > 0 || throw(ArgumentError("action probabilities must have positive mass"))
    return p ./ total
end

function _action_index(action_list, action)
    idx = findfirst(isequal(action), action_list)
    idx === nothing && throw(ArgumentError("planner returned action $(repr(action)) outside the MDP action space"))
    return idx
end
