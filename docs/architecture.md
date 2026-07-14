# Arrodes: current architecture and migration guide

## Purpose

Arrodes explains an observed MuKumari agent by choosing among a finite set of
domain-authored objective hypotheses. A robotics application states the plausible
objectives, their prior probabilities, and the planning behavior expected for each one.

For hypotheses `H₁, …, Hₙ`, both inference engines target

```text
p(Hᵢ | a₁:t, s₁:t) ∝ p(Hᵢ) ∏ₖ p(aₖ | s₁:k, a₁:k-1, Hᵢ).
```

`infer_objectives_smc` is the scalable default. `infer_objectives` retains exact finite
enumeration as a reference calculation and debugging oracle.

## System flow

```text
ObjectiveHypothesis
  ├─ objective
  ├─ prior_probability
  ├─ metadata
  └─ BehaviorModel
       ├─ AbstractPlanner
       └─ AbstractActionLikelihood
              │
              v
mdp_builder(objective, hypothesis)
              │
              v
MuKumari MDP ──> prepared policy/path ──> action probabilities
              │                              │
              └──────────────────────────────┘
                              │
                              v
              sequential SMC or exact update
                              │
                              v
             posterior history and visual checks
```

## MuKumari is foundational

MuKumari is the core simulation and MDP layer, not an optional extension. Arrodes uses
its `KAgentPOMDP`, states, observation shaping, environment models, geometry, dynamics,
and belief updater. `build_kagent_pomdp`, `build_shared_menv`, and
`agent_params_from_mdp` are native Arrodes helpers around that scaffolding.

`DiscreteInferenceConfig.mdp_builder` is still user-supplied because different domains
assemble objectives and environment metadata differently. Every resulting MDP is
expected to obey the MuKumari/POMDPs API.

## Objective hypotheses

`ObjectiveHypothesis` replaces the former sampled function representation. It contains:

- a stable symbolic ID;
- an objective understood by the configured MDP builder;
- a behavior model;
- a positive prior probability;
- arbitrary domain metadata.

The prior can encode a shared ranking of likely goals or priorities. Different
hypotheses may use different planners—for example, goal-directed MCTS for delivery and
an information-based VulcanJ planner for exploration.

## Planning layer

Every planner prepares an artifact:

```julia
prepare(planner, mdp, context) -> artifact
```

and exposes an action, action scores, or a complete rollout:

```julia
planned_action(planner, artifact, mdp, state, context)
action_scores(planner, artifact, mdp, state, context)
rollout(planner, artifact, mdp, initial_state, horizon, context)
```

### Native planners

- `SoftQPlanner` builds and trains a Crux Soft-Q policy. It exposes Q-derived action
  scores and is normally paired with `BoltzmannScoreLikelihood`.
- `MCTSPlanner` builds MuKumari belief-state vanilla MCTS or DPW policies.
- `VulcanMCTSPlanner` accepts a factory for VulcanJ's
  `RiskBoundedInfoMCTS`, preserving application-specific risk and GP configuration.
- `VulcanErgodicPlanner` calls `one_shot_ergodic_planner` and retains the complete
  optimized path as an `OpenLoopArtifact`.
- `POMDPSolverPlanner` adapts any external `POMDPs.Solver`.
- `CallbackPlanner` accepts application functions directly.
- `OpenLoopPlanner` adapts other full-trajectory optimizers.

### Planning context and caching

`PlanningContext` supplies the hypothesis ID, timestep, observed state/action history,
horizon, deterministic RNG, and hypothesis metadata. Planners declare cache scope:

- `:hypothesis`: one stationary policy per hypothesis;
- `:initial_state`: one open-loop plan for a starting condition;
- `:history`: replan for every distinct observation history;
- `:none`: never cache.

This makes expensive planning reuse explicit instead of hiding it in loosely typed RL
dictionaries.

Planner artifacts are deep-copied before likelihood queries. This matters for stateful
policies such as VulcanJ InfoMCTS: its search tree and mission counters may mutate on
`action`, while SMC replay must remain independent of particle evaluation order.

## Action-likelihood layer

Planning and demonstrator noise are intentionally separate. Inference always requests

```julia
action_distribution(likelihood, planner, artifact, mdp, state, actions, context)
```

- `BoltzmannScoreLikelihood` softmaxes Q-values or other planner scores.
- `EpsilonGreedyLikelihood` gives a deterministic policy a controlled error model.
- `PlanTrackingLikelihood` scores adherence to an open-loop plan.
- `MovementNoiseLikelihood` turns a deterministic plan into an empirical trajectory
  distribution by sampling the MuKumari MDP transition model and kernel-scoring the
  observed successor state. This is intended for ergodic and other open-loop planners.
- `CallbackLikelihood` supports domain-specific behavior models.

This permits MCTS, learned Q-functions, risk-bounded search, and trajectory optimization
to coexist without pretending that they all expose Q-values.

## Sequential particle filtering and P3-style rejuvenation

`SMCInferenceConfig` controls particle count, ESS threshold, systematic resampling,
rejuvenation count, and an optional domain-informed proposal. Each `ParticleTrace`
retains its hypothesis, ancestor, incremental scores, cumulative likelihood, and
hypothesis history. `SMCFilterResult` additionally retains ESS and ancestry at every
timestep.

Each observation produces an incremental factor. Normalized particles are resampled
when ESS drops below the configured fraction, then a resample-move Metropolis-Hastings
kernel proposes alternative objectives and replays the accumulated observation trace.
The default independence proposal draws from the objective prior; a custom P3-style
proposal may return `(index, log_forward, log_reverse)` so the MH correction remains
valid. This preserves the important Gen/GenParticleFilters/SMC-P3 semantics without
coupling Arrodes' planner objects to Gen choice-map addresses.

Within one rejuvenation sweep, replay scores are memoized by objective hypothesis.
Thus an expensive InfoMCTS history is evaluated at most once per proposed objective,
instead of once per particle.

## Exact reference filtering

`initialize_filter` creates normalized log prior weights. `update!` evaluates every
hypothesis for one observed state/action pair and performs a log-sum-exp posterior
update. `infer_objectives` runs the complete sequence and returns:

- the final `DiscreteFilterState`;
- posterior probability at every timestep;
- accumulated log evidence at every timestep;
- cached MuKumari MDPs and planning artifacts.

## Visual and animation infrastructure

All former diagnostic views have equivalents in the new system:

1. `plot_particle_filter_frame(...; trace_from_current=false)` overlays the observed
   path and top-hypothesis rollouts planned from the shared initial state.
2. `plot_particle_filter_frame(...; trace_from_current=true)` replans and overlays paths
   from the current observed state.
3. `plot_particle_heatmaps_frame` renders the true objective and each leading named
   objective as separate heatmaps with posterior probabilities.
4. `plot_particle_filter_explanation` renders the final trajectory/posterior summary.
5. `make_particle_filter_frame_fn` and `make_particle_heatmaps_frame_fn` produce frame
   closures.
6. `animate_particle_filter_from_frames` and `save_particle_filter_animation` assemble
   and save animations.

The diagnostic function names retain “particle filter” for workflow continuity. They
accept both `SMCFilterResult` and `DiscreteFilterResult`.

## What was removed

The following old mechanisms were removed rather than carried alongside the redesign:

- generic compositional objective sampling;
- `ScoreΠDist` and its untyped MDP/solver/policy caches;
- `RLConfig`, which exposed Soft-Q hyperparameters while hiding the planner itself;
- Fourier-specific ablations, cached datasets, plots, and metadata;
- IQ-SIPS paths that depended on the obsolete particle representation.

The continuous RFF/RBF field constructors and samplers are retained in the deprecated
`ObjectiveFields` module for legacy experiments, but are deliberately excluded from
the filtering model. Gen-specific choice-map wiring remains removed: Arrodes now owns
typed traces, ancestry, resampling, and rejuvenation directly.

The other items were removed because they encoded the unsuccessful continuous measure-space
translation, duplicated the new finite model, or depended directly on the old trace
schema. The visualization capability was not conceptually obsolete; its old
particle-address implementation was, so it has been rebuilt around named hypotheses.
