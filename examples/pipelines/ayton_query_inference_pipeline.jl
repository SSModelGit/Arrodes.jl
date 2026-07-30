using Arrodes
using POMDPs

# This file is intentionally self-contained. It is an example of spelling out an
# Ayton query, not a reusable query-language implementation.
#
# Ayton writes a query as Q = <f_Q, J_Q, Delta_Q>:
#   f_Q       maps a path and possible environment to a variable of interest zeta;
#   J_Q       is a value, probability, or information objective, optionally
#             specialized over posterior values or prior mission outcomes; and
#   Delta_Q   is an optional sufficient reward at which sampling may stop.
Base.@kwdef struct ExampleQuery{F}
    id::Symbol
    f_Q::F
    basic_objective::Symbol
    posterior_specialization = :none
    prior_mass::Union{Nothing,Float64} = nothing
    Delta_Q::Union{Nothing,Float64} = nothing
    description::String
end

const PEAK_LEVEL = 7.0

# Possible environment realization M used by the example query functions.
environment = (
    elevation = [1.0, 8.2, 3.1, 9.4, 2.0, 7.8],
    correlation_length = 12.0,
    eruption_edges = Set([(:pressure, :eruption)]),
)
path = [1, 2, 4]

# The functions below illustrate all three kinds of f_Q output described in
# Section 2.7.1: continuous/quantitative, logical, and classification outputs.
sampled_peak_count(path, M) = count(i -> M.elevation[i] > PEAK_LEVEL, path)
total_peak_count(_path, M) = count(>(PEAK_LEVEL), M.elevation)
any_peak(_path, M) = any(>(PEAK_LEVEL), M.elevation)
correlation_length(_path, M) = M.correlation_length
eruption_is_pressure_driven(_path, M) = (:pressure, :eruption) in M.eruption_edges
function volcano_class(_path, M)
    maximum(M.elevation) < PEAK_LEVEL && return :inactive
    count(>(PEAK_LEVEL), M.elevation) == 1 && return :isolated_peak
    return :volcanic_field
end

# Legal query forms from Sections 2.7.3--2.7.5. In particular:
#   * a constant posterior set is legal only with a value objective;
#   * V*_S is legal with value and probability objectives;
#   * an information objective has no posterior specialization.
queries = [
    ExampleQuery(id = :value,
        f_Q = sampled_peak_count, basic_objective = :value,
        description = "maximize the expected number of sampled peaks"),

    ExampleQuery(id = :probability,
        f_Q = total_peak_count, basic_objective = :probability,
        description = "maximize expected posterior belief in the true peak count"),

    ExampleQuery(id = :information,
        f_Q = correlation_length, basic_objective = :information,
        description = "maximize mutual information about GP correlation length"),

    ExampleQuery(id = :constant_posterior_set,
        f_Q = sampled_peak_count, basic_objective = :value,
        posterior_specialization = zeta -> zeta >= 3,
        description = "maximize posterior probability that at least three peaks are sampled"),

    ExampleQuery(id = :best_value,
        f_Q = total_peak_count, basic_objective = :value,
        posterior_specialization = (:best, 1),
        description = "maximize posterior probability of the largest possible peak count"),

    ExampleQuery(id = :posterior_mode,
        f_Q = volcano_class, basic_objective = :probability,
        posterior_specialization = (:best, 1),
        description = "maximize posterior probability of the most likely volcano class"),

    ExampleQuery(id = :prior_specialization,
        f_Q = sampled_peak_count, basic_objective = :value,
        prior_mass = 0.90,
        description = "maximize sampled peaks on the best 90% of mission outcomes"),

    ExampleQuery(id = :sufficient_query,
        f_Q = eruption_is_pressure_driven, basic_objective = :probability,
        posterior_specialization = (:best, 1), prior_mass = 0.95, Delta_Q = 0.80,
        description = "sample until mode confidence reaches 80% on the best 95% of outcomes"),
]

# Small executable calculations make the notation concrete. A real planner would
# obtain these distributions by sampling its environment model after candidate
# observations; Arrodes only needs the resulting query-specific behavior.
function posterior_reward(query, support, posterior; prior = nothing)
    p = posterior ./ sum(posterior)
    specialization = query.posterior_specialization
    if specialization isa Function
        return sum(probability for (zeta, probability) in zip(support, p)
            if specialization(zeta))
    elseif specialization isa Tuple && first(specialization) === :best
        count = last(specialization)
        ordering = query.basic_objective === :value ?
            sortperm(support; rev = true) : sortperm(p; rev = true)
        return sum(p[ordering[1:min(count, length(ordering))]])
    elseif query.basic_objective === :value
        return sum(Float64(zeta) * probability for (zeta, probability) in zip(support, p))
    elseif query.basic_objective === :probability
        return sum(abs2, p) # E_zeta[p(zeta | observations)], thesis Eq. 2.25.
    end

    q = prior ./ sum(prior)
    return sum(probability > 0 ? probability * log(probability / q0) : 0.0
        for (probability, q0) in zip(p, q))
end

println("Ayton query declarations:")
for query in queries
    println("  ", query.id, ": zeta = ", query.f_Q(path, environment),
        ", J_basic = ", query.basic_objective,
        ", Delta_Q = ", query.Delta_Q)
end

peak_support = collect(0:4)
peak_posterior = [0.02, 0.08, 0.20, 0.50, 0.20]
println("P(at least three peaks) = ",
    posterior_reward(queries[4], peak_support, peak_posterior))
class_support = [:inactive, :isolated_peak, :volcanic_field]
class_posterior = [0.05, 0.15, 0.80]
mode_confidence = posterior_reward(queries[8], class_support, class_posterior)
println("Mode confidence = ", mode_confidence,
    "; Delta_Q reached = ", mode_confidence >= queries[8].Delta_Q)

# Finally, infer which declared query explains an observed behavior. The mapping
# is intentionally obvious so the example stays about writing queries; replace
# CallbackPlanner with an InfoMCTS/ergodic planner in an application.
struct QueryExampleMDP <: MDP{Int,Symbol}
    query::ExampleQuery
end

const QUERY_ACTIONS = [
    :sample_peaks, :resolve_count, :learn_length_scale, :find_three_peaks,
    :seek_largest_count, :classify_volcano, :seek_high_upside, :test_causal_model,
]
POMDPs.actions(::QueryExampleMDP) = QUERY_ACTIONS
POMDPs.discount(::QueryExampleMDP) = 0.98
POMDPs.isterminal(::QueryExampleMDP, state) = false
POMDPs.gen(::QueryExampleMDP, state::Int, action::Symbol, rng) =
    (sp = state + 1, r = 0.0)

action_for(query) = QUERY_ACTIONS[findfirst(q -> q.id === query.id, queries)]
planner = CallbackPlanner(
    prepare_fn = (mdp, context) -> action_for(mdp.query),
    action_fn = (action, mdp, state, context) -> action,
)
hypotheses = [ObjectiveHypothesis(
    id = query.id,
    objective = query,
    behavior = BehaviorModel(planner, EpsilonGreedyLikelihood(epsilon = 0.08)),
    prior_probability = 1 / length(queries),
    metadata = (; query.description),
) for query in queries]
model = DiscreteInferenceConfig(
    hypotheses = hypotheses,
    mdp_builder = (query, hypothesis) -> QueryExampleMDP(query),
)

true_query = :information
states = collect(0:7)
actions = fill(action_for(queries[3]), length(states))
result = infer_objectives_smc(SMCInferenceConfig(
    model = model, n_particles = 240, ess_threshold = 0.7, rejuvenation_steps = 3,
), states, actions)

println("True query: ", true_query)
println("Posterior: ", Dict(h.id => p for (h, p) in zip(hypotheses, posterior(result))))
println("Inferred query: ", best_hypothesis(result).hypothesis.id)
