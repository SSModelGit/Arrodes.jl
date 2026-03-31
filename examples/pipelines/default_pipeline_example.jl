using Random
using POMDPs
using MuKumari: blindstart_KAgentState, shape_state_as_obs, expert_simulator
using Arrodes
using Flux
using Plots

import GeoInterface as GI

rng = Random.MersenneTwister(42)

# Setup observable agent
spec = MuEnvSpec()
menv = build_shared_menv(spec)

agent_params = Dict(
    :start => [1.0 1.0],
    :dimensions => (0.0, 10.0),
    :menv => menv,
    :obcs => [
        GI.Polygon([[(2.0, 2.0), (2.0, 3.0), (3.0, 3.0), (3.0, 2.0), (2.0, 2.0)]]),
        GI.Polygon([[(5.0, 5.0), (5.0, 6.0), (6.0, 6.0), (6.0, 5.0), (5.0, 5.0)]])
    ]
)

# true_objective_fn = (x, y) -> (
#     0.6 * cos(0.5*x + 0.5*y + π/4) +
#     0.4 * exp(-((x-5.0)^2 + (y-5.0)^2) / (2*0.5^2))
# )

true_objective_fn = x -> (0.6 * cos(0.5*x[1,1] + 0.5*x[1,2] + π/4))
mdp_objective = x -> (true_objective_fn(x.x), false)

# For visualization: define (x, y) -> Real version
true_objective_fn_viz = (x, y) -> 0.6 * cos(0.5*x + 0.5*y + π/4)

mdp = build_kagent_pomdp(agent_params, mdp_objective)

# Generate observations using learned policy
alist = collect(POMDPs.actions(mdp))
n_timesteps = 15
obs_dims = length(rand(initialobs(mdp, blindstart_KAgentState(mdp, mdp.start))))

# Create MCTS planner on the true MDP
# solver = MCTSSolver(n_iterations=100, depth=15, exploration_constant=1.0)
solver = solver_from_type(mdp, :mcts; solver_params=[:dpw, 1000, 10.0])
bup = solver.updater
planner = solve(solver, mdp)

# Generate expert experience using MuKumari's expert_simulator
experience = expert_simulator(mdp, planner, bup;
                              max_steps=n_timesteps, sim_limit=20, obs_dims=obs_dims,
                              nonterminal_system=true)

# Extract state data and actions
state_data = experience[:s]  # (obs_dim × n_timesteps)

# Extract action indices from one-hot encoded actions
A = experience[:a]  # (n_actions × n_timesteps, boolean)
observations = onehot_cols_to_aidx(Float64.(A))

# Setup component priors
fourier_field = RandomFourierField(amplitude_max=10.0, freq_max=π)
rbf_field = RadialBasisField(
    x_min=0.0, x_max=10.0, 
    y_min=0.0, y_max=10.0,
    amp_min=0.1, amp_max=10.0, 
    σ=0.5
)

component_tuples = [
    (fourier_field, fourier_params_sampler(fourier_field)),
    (rbf_field, rbf_params_sampler(rbf_field))
]

# Configure inference
param_switch, component_fields = build_component_param_switch(component_tuples)
ct_sampler = component_type_sampler(component_fields)

rl_config = RLConfig(
    n_iterations=200,
    epochs=2,
    batch_size=256,
    temperature=1.5
)

config = InferenceConfig(
    component_tuples=component_tuples,
    component_params_switch=param_switch,
    component_type_sampler=ct_sampler,
    k_components=1,
    rl_config=rl_config,
    agent_params=agent_params
)

# Run inference
π_dist = ScoreΠDist(
    mdp_params=[
        alist,
        a -> Float64.(Flux.onehot(a, alist)),
        Float64.(Flux.onehotbatch(alist, alist))
    ]
)

n_particles = 10
pf_state = particle_filter(
    observations,
    config,
    π_dist,
    state_data,
    n_particles
)

# Extract and display results
best_idx, best_weight, comp_idxs, comp_params, obj_fn = 
    best_particle(pf_state, config, component_fields)

println("Inferred component: $(["Fourier", "RBF"][comp_idxs[1]])")
println("Best particle weight: $best_weight")

# Visualize particle filter explanation
p = Visualizations.plot_particle_filter_explanation(
    pf_state, config, component_fields, true_objective_fn_viz,
    state_data, agent_params, π_dist, mdp;
    gridsize=150, n_top=10
)

display(p)
