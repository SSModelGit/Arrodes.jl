# SpockQueryFramework.jl design

Recording a design setup for a future implementation of the queries defined in Spock.
Not meant to be implemented in Arrodes but in a future package called `SpockQueryFramework.jl`.

## Purpose

SpockQueryFramework will adapt the scientific query language into Julia, specifically to help define the space of potential queries as an enumerable space.

A query states:
- which scientific quantity is being asked about;
- how a belief or change in belief is valued;
- any parameters such as variables, regions, or thresholds; and
- when the query is sufficiently answered.

## Relation to existing package ecosystem

| Package | Purpose |
|---|---|---|
| `SCRIBE.jl` | EOF models, coefficient uncertainty, prediction, hypothetical and physical observation updates |
| `SpockQueryFramework.jl` | query definitions and evaluation of their scientific value |
| `VulcanJ.jl` | InfoMCTS, ergodic optimization, and action selection using a supplied query objective |
| `MuKumari.jl` | MDP/POMDP dynamics, agent operation, observations, and mission simulation |
| `Arrodes.jl` | posterior inference from behavior and behavioral score calibration |

## Mathematical definition

A query can be summarized as

\[
Q = \langle f_Q, J_Q, \Delta_Q \rangle,
\]

where:

- \(f_Q\) evaluates the scientific variable or event;
- \(J_Q\) gives the scalar value used by a planner or behavior model; and
- \(\Delta_Q\) determines whether the question is sufficiently answered.

Simply put:
* The measurable field (field function f)
* The objective function (query objective function J)
* The optional threshold (to indicate completion of the query, \Delta)

A peak-existence query, for example, may derive a threshold event with \(f_Q\), value its posterior probability with \(J_Q\), and stop when that probability crosses a confidence threshold with \(\Delta_Q\).

## Minimal Julia surface

The first implementation should begin with functions, not a framework of evaluation-result types:

```julia
query_value(query, belief)::Real
query_value(query, belief, input)::Real
query_gain(query, prior_belief, posterior_belief)::Real
query_satisfied(query, belief)::Bool
query_spec(query)::Dict{Symbol,Any}
```

Only queries that have a meaningful spatial evaluation implement:

```julia
query_field(query, belief, locations)::Dict{Symbol,Any}
```

The dictionary should contain only scientifically necessary fields, nominally:

```julia
Dict(
    :values => values,
    :domain => :nonnegative,  # or :real, :probability
    :meaning => :threshold_probability,
)
```

This prevents a signed field, utility, or information gain from accidentally qualifying as an occupancy distribution.

To improve performance, we can provide a gradient and Jacobian computation, but is unnecessary:
```julia
query_gradient(query, belief, input)
query_field_jacobian(query, belief, locations)
```

Absence of a derivative is not an invalid query. VulcanJ or Arrodes, the current packages intended to consume SpockQueryFramework, will choose use finite differences or sampling when appropriate. Ideally, this will rely on multi-dispatch.

## Query types

The query itself is the persistent scientific object. Initial concrete query types should be introduced only where their parameters and dispatch differ.

The first useful families are:

1. **Trace uncertainty** — minimize the trace of a supplied model covariance.
2. **Mutual information** — maximize information gained between a prior and
   hypothetical posterior.
3. **Threshold exceedance** — value the probability that a field quantity
   crosses a specified threshold at a location or region.
4. **Peak count** — value the expected number of distinct threshold-crossing
   peaks.
5. **Peak existence** — value confidence that at least one qualifying peak
   exists.

## Belief interface

Spock should be model-agnostic, but should define generic operations needed by implemented queries, and packages can add methods for their own types. Likely operations include:

```julia
belief_mean(belief, locations)
belief_covariance(belief, locations)
belief_sample(rng, belief, locations)
```

A SCRIBE adapter can implement these using SCRIBE prediction functions. Information queries should normally receive both beliefs explicitly:

```julia
query_gain(query, prior_model, hypothetical_model)
```

The operation that creates `hypothetical_model` belongs to SCRIBE or another belief package. This keeps query semantics independent of measurement-update mechanics.

## VulcanJ interface

VulcanJ should consume the smallest query operation required by a planner:

- InfoMCTS consumes `query_gain` or `query_value` after hypothetical model updates.
- An ergodic planner consumes `query_field` only after a mission has declared an explicit, mathematically valid target-measure transformation.
- A stopping rule consumes `query_satisfied`.

VulcanJ remains responsible for tree search, trajectory optimization, risk bounds, control costs, and action selection. Spock should not contain planner configuration or return planned paths.

## Arrodes interface

Arrodes uses a query only when it is known to be the objective guiding the observed agent. For a candidate SCRIBE world model and an observed trajectory, the immediate contract needed by the current world filter is equivalent to:

```julia
query_value(query, candidate_model, observation)::Real
query_gradient(query, candidate_model, observation) # optional
```

In relation to the discrepancy energy score used by Arrodes:

- MMD is appropriate when the known behavior objective contains a long-run ergodic target measure.
- \(J_Q\) is appropriate when the known behavior directly optimizes the query value or gain.