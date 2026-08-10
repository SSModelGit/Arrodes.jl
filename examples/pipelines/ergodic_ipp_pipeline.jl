using Arrodes
using GaussianProcesses
using LinearAlgebra
using MuKumari
using Plots
using POMDPs
using Parameters: @with_kw
using Random
using SpecialFunctions: erf
using VulcanJ

# A mixed-planner inverse-planning example based on VulcanJ's
# `mukumari_volcano_search_risk_bounded.jl` world. InfoMCTS acquires information
# about four GP-derived mission statistics; the MI-field hypothesis uses the
# native one-shot ergodic planner.

const PEAK_THRESHOLD = 7.0
const VOLCANO_HORIZON = 8

# Example-local Ayton query Q = <f_Q, J_Q, Delta_Q>. The volcano pipeline only
# needs the query function and objective class, so no general query framework is
# introduced into Arrodes itself.
@with_kw struct VolcanoQuery
    id::Symbol
    query_function::Function
    kind::Symbol
    posterior_specialization = :none
    prior_mass::Union{Nothing,Float64} = nothing
    sufficient_reward::Union{Nothing,Float64} = nothing
    description::String
end

@with_kw struct VolcanoObjective{Q<:VolcanoQuery}
    query::Q
    threshold::Float64 = PEAK_THRESHOLD
end

struct ObjectiveVolcanoMDP <: MDP{KAgentState,Symbol}
    base::KAgentMDP
    objective::VolcanoObjective
    dimensions::Tuple{Float64,Float64}
    obj::Function
    sites::Vector{Matrix{Float64}}
    prior_sites::Vector{Matrix{Float64}}
end

function random_volcanoes(bounds, count; rng = MersenneTwister(23))
    lo, hi = Float64.(bounds)
    span = hi - lo
    return [begin
        center = [lo + span * rand(rng) lo + span * rand(rng)]
        radius = span * (0.05 + 0.07 * rand(rng))
        (; center, height = 5.0 + 8.0 * rand(rng),
            spread = inv(2radius^2), ring_radius = radius * (0.25 + 0.2rand(rng)),
            ring_width = radius * (0.08 + 0.08rand(rng)), ring_gain = 2.5 + 5rand(rng))
    end for _ in 1:count]
end

function volcanic_elevation(volcanoes)
    return function (X)
        x, y = Float64(X[1]), Float64(X[2])
        sum(volcanoes) do volcano
            cx, cy = volcano.center
            distance = hypot(x - cx, y - cy)
            volcano.height * exp(-volcano.spread * distance^2) +
                volcano.ring_gain * exp(-(distance - volcano.ring_radius)^2 /
                    (2volcano.ring_width^2))
        end
    end
end

function build_volcano_world(; rng = MersenneTwister(19), dimensions = (0.0, 100.0))
    elevation = volcanic_elevation(random_volcanoes(dimensions, 14; rng = rng))
    environment = MuEnv(1, [:elevation], Dict(:elevation => elevation))
    exclusion_zone = Dict(
        :poly => [(47.0, 47.0), (47.0, 53.0), (53.0, 53.0),
            (53.0, 47.0), (47.0, 47.0)],
        :risk => 150.0,
        :impact => 0.1,
    )
    landscape = AgentObjectiveLandscape(objectives = [
        (:goal, Dict(:target => [92.0 92.0], :strength => 0.0, :influence => 35.0, :size => 4.0)),
        (:robc, [exclusion_zone]),
        (:horz, 0.0),
    ])
    return init_standard_KAgentMDP(
        name = "arrodes_volcano_ipp", start = [8.0 8.0], dimensions = dimensions,
        objl = landscape, menv = environment, digits = 3, agent_width = 0.1,
        agent_speed = 8.0, ag_mvt_noise = 0.35, obs_noise = 0.05,
        mdp_horizon_discount = 0.97,
    )
end

elevation(mdp::KAgentMDP, state) = Float64(mdp.menv.μf[:elevation](VulcanJ.extract_location(state)))

function initial_gp(mdp::KAgentMDP, prior_sites)
    inputs = hcat([vec(site) for site in prior_sites]...)
    outputs = [elevation(mdp, site) for site in prior_sites]
    return GPE(inputs, outputs, MeanZero(), SE(fill(log(12.0), 2), 0.0))
end

function exceedance_probability(gp, state, threshold)
    location = VulcanJ.extract_location(state)
    mean, variance = predict_f(gp, location')
    μ, σ² = first(vec(mean)), max(first(vec(variance)), eps())
    return 0.5 * (1 - erf((threshold - μ) / sqrt(2σ²)))
end

function objective_proxy(objective, gp, state)
    location = VulcanJ.extract_location(state)
    _, variance = predict_f(gp, location')
    σ² = max(first(vec(variance)), eps())
    objective.query.id === :minimize_uncertainty_trace && return -σ²
    objective.query.id in (:maximize_mutual_information, :ergodic_mutual_information) && return log1p(σ²)
    return exceedance_probability(gp, state, objective.threshold)
end

function wrap_objective(base, objective; resolution = (8, 8))
    lo, hi = Float64.(base.dimensions)
    sites = [reshape([Float64(x), Float64(y)], 1, 2)
        for x in range(lo, hi; length = resolution[1])
        for y in range(lo, hi; length = resolution[2])]
    prior_sites = [[8.0 8.0], [8.0 92.0], [92.0 8.0], [92.0 92.0], [50.0 50.0]]
    gp = initial_gp(base, prior_sites)
    plotted_field = Ref{Function}(state -> objective_proxy(objective, gp, state))
    mdp = ObjectiveVolcanoMDP(base, objective, (lo, hi),
        state -> (plotted_field[](state), false), sites, prior_sites)

    # Plot the same one-step information reward that VulcanJ evaluates inside
    # InfoMCTS and when constructing an ergodic target density. The ergodic case
    # uses VulcanJ's normalization because that normalized density is the actual
    # spatial objective optimized by `one_shot_ergodic_planner`.
    rewards = [VulcanJ.expected_single_observation_reward(mdp, gp, site, 3)
        for site in sites]
    field_values = objective.query.id === :ergodic_mutual_information ?
        VulcanJ.normalize_density(rewards) : rewards
    plotted_field[] = function (state)
        location = vec(VulcanJ.extract_location(state))
        _, index = findmin([sum(abs2, vec(site) .- location) for site in sites])
        return field_values[index]
    end
    return mdp
end

# Delegate simulation to the shared MuKumari world. Every hypothesis therefore has
# identical dynamics and differs only in its GP statistic and planning method.
POMDPs.actions(mdp::ObjectiveVolcanoMDP) = POMDPs.actions(mdp.base)
POMDPs.actions(mdp::ObjectiveVolcanoMDP, state) = POMDPs.actions(mdp.base)
POMDPs.initialstate(mdp::ObjectiveVolcanoMDP) = POMDPs.initialstate(mdp.base)
POMDPs.discount(mdp::ObjectiveVolcanoMDP) = POMDPs.discount(mdp.base)
POMDPs.isterminal(mdp::ObjectiveVolcanoMDP, state) = POMDPs.isterminal(mdp.base, state)
POMDPs.gen(mdp::ObjectiveVolcanoMDP, state::KAgentState, action, rng) =
    POMDPs.gen(mdp.base, state, action, rng)
POMDPs.gen(mdp::ObjectiveVolcanoMDP, state::Matrix, action, rng) =
    POMDPs.gen(mdp.base, blindstart_KAgentState(mdp.base, reshape(Float64.(state), 1, 2)), action, rng)

VulcanJ.extract_location(state::KAgentState) = reshape(Float64.(state.x), 1, :)
VulcanJ.get_initial_gp(mdp::ObjectiveVolcanoMDP, state) = initial_gp(mdp.base, mdp.prior_sites)
VulcanJ.add_obs_to_gp(state::Union{KAgentState,Matrix}, observation::Real, gp::GPE) = begin
    location = VulcanJ.extract_location(state)
    GPE(hcat(gp.x, location'), vcat(gp.y, Float64(observation)), gp.mean, gp.kernel)
end
VulcanJ.cellsites(mdp::ObjectiveVolcanoMDP) = mdp.sites
VulcanJ.horizon(::ObjectiveVolcanoMDP) = VOLCANO_HORIZON
VulcanJ.get_failure_prob(::ObjectiveVolcanoMDP, state, action) = 0.0

const ANY_PEAK_CACHE = IdDict{GPE,Float64}()

function VulcanJ.posterior_phenomenon_prob(mdp::ObjectiveVolcanoMDP, gp::GPE, state)
    objective = mdp.objective
    if objective.query.id === :minimize_uncertainty_trace
        _, variance = predict_f(gp, VulcanJ.extract_location(state)')
        σ² = max(first(vec(variance)), eps())
        return clamp(σ² / (1 + σ²), eps(), 1 - eps())
    elseif objective.query.id === :information_any_peak_existence
        return get!(ANY_PEAK_CACHE, gp) do
            probabilities = [exceedance_probability(gp, site, objective.threshold) for site in mdp.sites]
            clamp(1 - prod(1 - probability for probability in probabilities), eps(), 1 - eps())
        end
    end
    return clamp(exceedance_probability(gp, state, objective.threshold), eps(), 1 - eps())
end

objectives = [
    VolcanoObjective(query = VolcanoQuery(
        id = :minimize_uncertainty_trace,
        query_function = (_path, gp_values) -> gp_values,
        kind = :information,
        description = "Reduce uncertainty in the GP field (covariance-trace criterion)")),
    VolcanoObjective(query = VolcanoQuery(
        id = :maximize_mutual_information,
        query_function = (_path, elevation_field) -> elevation_field,
        kind = :information,
        description = "Maximize mutual information about the elevation field")),
    VolcanoObjective(query = VolcanoQuery(
        id = :ergodic_mutual_information,
        query_function = (_path, elevation_field) -> elevation_field,
        kind = :information,
        description = "Sample ergodically with respect to elevation-field information")),
    VolcanoObjective(query = VolcanoQuery(
        id = :maximize_peak_count,
        query_function = (_path, elevation_field) -> count(>(PEAK_THRESHOLD), elevation_field),
        kind = :value,
        description = "Maximize the expected count of above-threshold peaks"),
        threshold = PEAK_THRESHOLD),
    VolcanoObjective(query = VolcanoQuery(
        id = :information_any_peak_existence,
        query_function = (_path, elevation_field) -> any(>(PEAK_THRESHOLD), elevation_field),
        kind = :information,
        posterior_specialization = :none,
        description = "Maximize mutual information about the Boolean event that any above-threshold peak exists"),
        threshold = PEAK_THRESHOLD),
]

info_behavior() = BehaviorModel(
    VulcanMCTSPlanner((mdp, context) -> RiskBoundedInfoMCTS(
        lookahead = 3, time_budget = 0.08, quad_order = 3, risk_budget = 3.0,
        alpha = 0.0, reference_reward = 1.0, rng = context.rng)),
    EpsilonGreedyLikelihood(epsilon = 0.08),
)

ergodic_behavior() = BehaviorModel(
    VulcanErgodicPlanner((mdp, state, context) -> VulcanJ.get_initial_gp(mdp, state);
        n_steps = VOLCANO_HORIZON, optimizer_iters = 45, max_speed = 12.0,
        observe_fn = (mdp, state) -> elevation(mdp.base, state)),
    MovementNoiseLikelihood(n_transition_samples = 32, bandwidth = 12.0,
        action_epsilon = 0.04),
)

hypotheses = [ObjectiveHypothesis(
    id = objective.query.id,
    objective = objective,
    behavior = objective.query.id === :ergodic_mutual_information ? ergodic_behavior() : info_behavior(),
    prior_probability = 0.2,
    metadata = (; description = objective.query.description, threshold = objective.threshold),
) for objective in objectives]

shared_world = build_volcano_world()
problem = ObjectiveInferenceProblem(
    hypotheses = hypotheses,
    mdp_builder = (objective, hypothesis) -> wrap_objective(shared_world, objective),
    state_adapter = (mdp, observation, timestep) -> observation isa KAgentState ? observation :
        blindstart_KAgentState(mdp.base, reshape(Float64.(observation), 1, 2)),
    seed = 0x4552_474f_4449_4350,
)

# Generate the demonstration with objective 3 as the hidden truth.
true_id = :ergodic_mutual_information
true_hypothesis = hypotheses[hypothesis_index(problem, true_id)]
true_mdp = problem.mdp_builder(true_hypothesis.objective, true_hypothesis)
initial_state = rand(MersenneTwister(31), POMDPs.initialstate(true_mdp))
true_context = PlanningContext(hypothesis_id = true_id, states = Any[initial_state],
    horizon = VOLCANO_HORIZON,
    rng = MersenneTwister(hash((problem.seed, true_id, 1), UInt(0))),
    metadata = true_hypothesis.metadata)
true_artifact = prepare(true_hypothesis.behavior.planner, true_mdp, true_context)
demonstration = rollout(true_hypothesis.behavior.planner, true_artifact, true_mdp,
    initial_state, VOLCANO_HORIZON, true_context)
observed_actions = demonstration.actions
observed_states = [vec(VulcanJ.extract_location(state))
    for state in demonstration.states[1:length(observed_actions)]]

smc = SMCConfig(n_particles = 240, ess_threshold = 0.65,
    invariant_move = ObjectiveReplayMove(), invariant_steps = 3)
result = infer_objectives_smc(problem, observed_states, observed_actions, smc)

println("True objective: ", true_id)
println("Posterior: ", Dict(h.id => probability
    for (h, probability) in zip(hypotheses, posterior(result))))
println("Best explanation: ", best_hypothesis(result).hypothesis.id)
println("ESS history: ", getfield.(result.state.diagnostics, :ess))
println("Resampling stages: ", [d.stage for d in result.state.diagnostics if d.resampled])

output_dir = joinpath(@__DIR__, "res", "ergodic_ipp")
mkpath(output_dir)

# Elevation is the latent environment, not the agent's objective. Preserve it as
# a separately and accurately labelled diagnostic.
environment_axis = range(first(true_mdp.dimensions), last(true_mdp.dimensions); length = 80)
environment_plot = heatmap(environment_axis, environment_axis,
    [elevation(shared_world, [x y]) for y in environment_axis, x in environment_axis];
    aspect_ratio = :equal, color = :viridis, title = "Ground-truth volcanic elevation",
    xlabel = "x", ylabel = "y")
savefig(environment_plot, joinpath(output_dir, "volcano_environment.png"))

final_plot = plot_particle_filter_explanation(result;
    true_mdp = true_mdp, n_top = 5, gridsize = 30,
    rollout_horizon = VOLCANO_HORIZON)
savefig(final_plot, joinpath(output_dir, "final_explanation.png"))

# Reproduce the complete diagnostic animation set from the default pipeline.
# These are computationally heavier here because each displayed InfoMCTS path is
# produced by a real VulcanJ tree search.
animation_options = (
    true_mdp = true_mdp,
    n_top = length(hypotheses),
    gridsize = 30,
)

start_frame = make_particle_filter_frame_fn(result;
    animation_options...,
    trace_from_current = false,
    rollout_horizon = VOLCANO_HORIZON)
current_frame = make_particle_filter_frame_fn(result;
    animation_options...,
    trace_from_current = true,
    rollout_horizon = VOLCANO_HORIZON)
heatmaps_frame = make_particle_heatmaps_frame_fn(result; animation_options...)

timesteps = eachindex(observed_actions)
start_frames = [start_frame(t) for t in timesteps]
current_frames = [current_frame(t) for t in timesteps]
heatmap_frames = [heatmaps_frame(t) for t in timesteps]

save_particle_filter_animation(
    start_frames, joinpath(output_dir, "plans_from_start.gif"))
save_particle_filter_animation(
    current_frames, joinpath(output_dir, "plans_from_current.gif"))
save_particle_filter_animation(
    heatmap_frames, joinpath(output_dir, "objective_heatmaps.gif"))
