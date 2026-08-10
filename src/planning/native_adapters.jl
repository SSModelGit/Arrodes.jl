"""Thin adapter for any native solver implementing `POMDPs.solve` and `POMDPs.action`."""
struct POMDPSolverPlanner <: AbstractPlanner
    solver_factory::Function
end

POMDPSolverPlanner(solver::POMDPs.Solver) = POMDPSolverPlanner((_mdp, _context) -> deepcopy(solver))

function prepare(planner::POMDPSolverPlanner, mdp, context::PlanningContext)
    solver = _call_factory(planner.solver_factory, mdp, context)
    return PolicyArtifact(solver, POMDPs.solve(solver, mdp))
end

planned_action(::POMDPSolverPlanner, artifact::PolicyArtifact, mdp, state,
               context::PlanningContext) = POMDPs.action(artifact.policy, state)

"""General adapter for user-provided preparation and decision functions."""
@with_kw_noshow struct CallbackPlanner <: AbstractPlanner
    prepare_fn::Function
    action_fn::Union{Nothing,Function} = nothing
    scores_fn::Union{Nothing,Function} = nothing
    rollout_fn::Union{Nothing,Function} = nothing
    scope::Symbol = :hypothesis
end

cache_scope(planner::CallbackPlanner) = planner.scope
prepare(planner::CallbackPlanner, mdp, context::PlanningContext) =
    CallbackArtifact(planner.prepare_fn(mdp, context))

function planned_action(planner::CallbackPlanner, artifact::CallbackArtifact, mdp, state,
                        context::PlanningContext)
    isnothing(planner.action_fn) && throw(ArgumentError("CallbackPlanner has no action_fn"))
    return planner.action_fn(artifact.value, mdp, state, context)
end

function action_scores(planner::CallbackPlanner, artifact::CallbackArtifact, mdp, state,
                       context::PlanningContext)
    isnothing(planner.scores_fn) && return nothing
    return planner.scores_fn(artifact.value, mdp, state, context)
end

function rollout(planner::CallbackPlanner, artifact::CallbackArtifact, mdp, initial_state,
                 horizon::Integer, context::PlanningContext)
    isnothing(planner.rollout_fn) &&
        return _default_rollout(planner, artifact, mdp, initial_state, horizon, context)
    return planner.rollout_fn(artifact.value, mdp, initial_state, horizon, context)
end

"""Built-in MCTS/DPW planner for MuKumari belief-state POMDPs."""
@with_kw struct MCTSPlanner <: AbstractPlanner
    variant::Symbol = :dpw
    n_iterations::Int = 1000
    depth::Int = 20
    exploration_constant::Float64 = 10.0
end

function prepare(planner::MCTSPlanner, mdp, context::PlanningContext)
    base_solver = @match planner.variant begin
        :dpw => MCTS.DPWSolver(
            n_iterations = planner.n_iterations,
            depth = planner.depth,
            exploration_constant = planner.exploration_constant,
        )
        :vanilla => MCTS.MCTSSolver(
            n_iterations = planner.n_iterations,
            depth = planner.depth,
            exploration_constant = planner.exploration_constant,
        )
        _ => error("unknown MCTS variant: $(planner.variant)")
    end
    updater = MuKumari.KAgentBeliefUpdater(
        state_dims = length(mdp.start),
        env_dims = mdp.menv.M,
    )
    solver = MCTS.BeliefMCTSSolver(base_solver, updater)
    return PolicyArtifact(solver, POMDPs.solve(solver, mdp))
end

planned_action(::MCTSPlanner, artifact::PolicyArtifact, mdp, state,
               context::PlanningContext) = POMDPs.action(artifact.policy, state)

"""VulcanJ risk-bounded information-MCTS planner."""
struct VulcanMCTSPlanner <: AbstractPlanner
    solver_factory::Function
    quiet::Bool
end

VulcanMCTSPlanner(factory; quiet::Bool = true) =
    VulcanMCTSPlanner(factory, quiet)
VulcanMCTSPlanner(solver::VulcanJ.RiskBoundedInfoMCTS; quiet::Bool = true) =
    VulcanMCTSPlanner((_mdp, _context) -> deepcopy(solver); quiet = quiet)

function prepare(planner::VulcanMCTSPlanner, mdp, context::PlanningContext)
    solver = _call_factory(planner.solver_factory, mdp, context)
    solver isa VulcanJ.RiskBoundedInfoMCTS || throw(ArgumentError(
        "VulcanMCTSPlanner factory must return RiskBoundedInfoMCTS"))
    return PolicyArtifact(solver, POMDPs.solve(solver, mdp))
end

function planned_action(planner::VulcanMCTSPlanner, artifact::PolicyArtifact, mdp, state,
                        context::PlanningContext)
    planner.quiet || return POMDPs.action(artifact.policy, state)
    return redirect_stdout(devnull) do
        POMDPs.action(artifact.policy, state)
    end
end

"""VulcanJ one-shot ergodic path planner."""
struct VulcanErgodicPlanner <: AbstractPlanner
    gp_factory::Function
    n_steps::Int
    kwargs::NamedTuple
end

function VulcanErgodicPlanner(gp_factory; n_steps::Integer, kwargs...)
    n_steps >= 0 || throw(ArgumentError("n_steps must be nonnegative"))
    return VulcanErgodicPlanner(gp_factory, Int(n_steps), (; kwargs...))
end

cache_scope(::VulcanErgodicPlanner) = :initial_state

_make_vulcan_gp(factory, mdp, initial_state, context) =
    factory(mdp, initial_state, context)

function prepare(planner::VulcanErgodicPlanner, mdp, context::PlanningContext)
    isempty(context.states) && throw(ArgumentError(
        "VulcanErgodicPlanner requires an observed initial state"))
    initial_state = first(context.states)
    gp = _make_vulcan_gp(planner.gp_factory, mdp, initial_state, context)
    result = VulcanJ.one_shot_ergodic_planner(
        mdp,
        gp,
        planner.n_steps;
        initial_state = initial_state,
        rng = context.rng,
        planner.kwargs...,
    )
    return OpenLoopArtifact(result.states, result.actions, result)
end

function planned_action(::VulcanErgodicPlanner, artifact::OpenLoopArtifact, mdp, state,
                        context::PlanningContext)
    1 <= context.timestep <= length(artifact.actions) ||
        throw(BoundsError(artifact.actions, context.timestep))
    return artifact.actions[context.timestep]
end

rollout(::VulcanErgodicPlanner, artifact::OpenLoopArtifact, mdp, initial_state,
        horizon::Integer, context::PlanningContext) =
    (states = artifact.states, actions = artifact.actions[1:min(horizon, length(artifact.actions))])

"""Adapter for planners which construct a complete action sequence."""
@with_kw_noshow struct OpenLoopPlanner <: AbstractPlanner
    prepare_fn::Function
    scope::Symbol = :initial_state
end

OpenLoopPlanner(prepare_fn) = OpenLoopPlanner(prepare_fn, :initial_state)

cache_scope(planner::OpenLoopPlanner) = planner.scope

function prepare(planner::OpenLoopPlanner, mdp, context::PlanningContext)
    result = planner.prepare_fn(mdp, context)
    hasproperty(result, :actions) || throw(ArgumentError("open-loop result must expose `.actions`"))
    states = hasproperty(result, :states) ? result.states : Any[]
    return OpenLoopArtifact(states, result.actions, result)
end

function planned_action(::OpenLoopPlanner, artifact::OpenLoopArtifact, mdp, state,
                        context::PlanningContext)
    1 <= context.timestep <= length(artifact.actions) || throw(BoundsError(
        artifact.actions, context.timestep))
    return artifact.actions[context.timestep]
end

rollout(::OpenLoopPlanner, artifact::OpenLoopArtifact, mdp, initial_state, horizon::Integer,
        context::PlanningContext) =
    (states = artifact.states, actions = artifact.actions[1:min(horizon, length(artifact.actions))])

"""Crux Soft-Q planner retained as an explicit built-in planner, not an inference default."""
@with_kw struct SoftQPlanner <: AbstractPlanner
    n_iterations::Int = 200
    epochs::Int = 2
    batch_size::Int = 512
    hidden_sizes::Tuple{Vararg{Int}} = (64, 64)
end

function prepare(planner::SoftQPlanner, mdp, context::PlanningContext)
    action_list = collect(POMDPs.actions(mdp))
    state_space = POMDPs.state_space(mdp)
    input_dim = first(Crux.dim(state_space))
    layers = Any[]
    for width in planner.hidden_sizes
        push!(layers, Flux.Dense(input_dim, width, Flux.relu))
        input_dim = width
    end
    push!(layers, Flux.Dense(input_dim, length(action_list)))
    network = Crux.DiscreteNetwork(Flux.Chain(layers...), action_list; dev = Flux.cpu)
    solver = Crux.SoftQ(
        π = network,
        α = Float32(0.1),
        S = state_space,
        N = planner.n_iterations,
        ΔN = 1,
        c_opt = (; epochs = planner.epochs, batch_size = planner.batch_size),
        interaction_storage = [],
    )
    return PolicyArtifact(solver, POMDPs.solve(solver, mdp))
end

function planned_action(::SoftQPlanner, artifact::PolicyArtifact, mdp, state,
                        context::PlanningContext)
    return POMDPs.action(artifact.policy, state)
end


function action_scores(::SoftQPlanner, artifact::PolicyArtifact, mdp, state,
                       context::PlanningContext)
    action_list = collect(POMDPs.actions(mdp))
    encoded_actions = Float64.(Flux.onehotbatch(action_list, action_list))
    observation = MuKumari.shape_state_as_obs(mdp, state)
    return vec(Crux.value(artifact.policy, observation, encoded_actions))
end
