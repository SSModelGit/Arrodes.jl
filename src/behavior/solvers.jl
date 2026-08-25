struct BehaviorModel{S,L}
    solver::S
    likelihood::L
end

struct KnownActionPlanner{A}
    action::A
end

@with_kw struct MCTSPlanner
    variant::Symbol = :dpw
    n_iterations::Int = 1000
    depth::Int = 20
    exploration_constant::Float64 = 10.0
end

@with_kw struct VulcanErgodicPlanner{F}
    gp::F
    n_steps::Int
    options::Dict{Symbol,Any} = Dict{Symbol,Any}()
end

@with_kw struct SoftQPlanner
    n_iterations::Int = 200
    epochs::Int = 2
    batch_size::Int = 512
    hidden_sizes::Tuple{Vararg{Int}} = (64, 64)
    device::Symbol = :gpu
end

prepare_behavior(solver::POMDPs.Solver, mdp, context) =
    POMDPs.solve(deepcopy(solver), mdp)

prepare_behavior(::KnownActionPlanner, mdp, context) = nothing

function prepare_behavior(planner::MCTSPlanner, mdp, context)
    base = @match planner.variant begin
        :dpw => MCTS.DPWSolver(
            n_iterations=planner.n_iterations,
            depth=planner.depth,
            exploration_constant=planner.exploration_constant,
        )
        :vanilla => MCTS.MCTSSolver(
            n_iterations=planner.n_iterations,
            depth=planner.depth,
            exploration_constant=planner.exploration_constant,
        )
    end
    updater = MuKumari.KAgentBeliefUpdater(
        state_dims=length(mdp.start),
        env_dims=mdp.menv.M,
    )
    POMDPs.solve(MCTS.BeliefMCTSSolver(base, updater), mdp)
end

function prepare_behavior(planner::VulcanErgodicPlanner, mdp, context)
    initial_state = first(context[:states])
    gp = planner.gp(mdp, initial_state, context)
    VulcanJ.one_shot_ergodic_planner(
        mdp,
        gp,
        planner.n_steps;
        initial_state,
        rng=context[:rng],
        planner.options...,
    )
end

function prepare_behavior(planner::SoftQPlanner, mdp, context)
    device = @match planner.device begin
        :gpu => Flux.gpu
        :cpu => Flux.cpu
    end
    actions = collect(POMDPs.actions(mdp))
    state_space = POMDPs.state_space(mdp)
    input_dimension = first(Crux.dim(state_space))
    layers = Any[]
    for width in planner.hidden_sizes
        push!(layers, Flux.Dense(input_dimension, width, Flux.relu))
        input_dimension = width
    end
    push!(layers, Flux.Dense(input_dimension, length(actions)))
    network = device(Crux.DiscreteNetwork(Flux.Chain(layers...), actions; dev=device))
    solver = Crux.SoftQ(
        π=network,
        α=Float32(0.1),
        S=state_space,
        N=planner.n_iterations,
        ΔN=1,
        c_opt=(; epochs=planner.epochs, batch_size=planner.batch_size),
        interaction_storage=[],
    )
    POMDPs.solve(solver, mdp)
end

planned_action(::POMDPs.Solver, policy, mdp, state, context) =
    POMDPs.action(policy, state)

planned_action(planner::KnownActionPlanner, policy, mdp, state, context) =
    planner.action

planned_action(::MCTSPlanner, policy, mdp, state, context) =
    POMDPs.action(policy, state)

function planned_action(
    ::VulcanJ.RiskBoundedInfoMCTS,
    policy,
    mdp,
    state,
    context,
)
    redirect_stdout(devnull) do
        POMDPs.action(policy, state)
    end
end

planned_action(::VulcanErgodicPlanner, plan, mdp, state, context) =
    plan.actions[context[:timestep]]

planned_action(::SoftQPlanner, policy, mdp, state, context) =
    POMDPs.action(policy, state)

action_scores(planner, policy, mdp, state, context) = nothing

function action_scores(::SoftQPlanner, policy, mdp, state, context)
    actions = collect(POMDPs.actions(mdp))
    encoded = policy.device(Float32.(Flux.onehotbatch(actions, actions)))
    observation = MuKumari.shape_state_as_obs(mdp, state)
    vec(Crux.value(policy, observation, encoded))
end

function rollout_behavior(planner, artifact, mdp, initial_state, horizon, context)
    states = Any[initial_state]
    actions = Any[]
    state = initial_state
    for timestep in 1:horizon
        local_context = copy(context)
        local_context[:timestep] = timestep
        local_context[:states] = states
        local_context[:actions] = actions
        action = planned_action(planner, artifact, mdp, state, local_context)
        push!(actions, action)
        transition = POMDPs.gen(mdp, state, action, context[:rng])
        state = transition.sp
        push!(states, state)
        POMDPs.isterminal(mdp, state) && break
    end
    Dict(:states => states, :actions => actions)
end

function rollout_behavior(
    ::VulcanErgodicPlanner,
    plan,
    mdp,
    initial_state,
    horizon,
    context,
)
    Dict(
        :states => plan.states,
        :actions => plan.actions[1:min(horizon, length(plan.actions))],
    )
end
