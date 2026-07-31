# Arrodes

Arrodes infers which of a finite set of domain-informed objectives best explains an
observed agent trajectory. Each objective hypothesis explicitly specifies both the
objective used to construct its MuKumari MDP and the planning mechanism expected to
pursue it.

For hypotheses `H₁, …, Hₙ`, Arrodes targets the sequential posterior

```text
p(Hᵢ | a₁:t, s₁:t) ∝ p(Hᵢ) ∏ₖ p(aₖ | s₁:k, a₁:k-1, Hᵢ).
```

The normal engine is trace-preserving sequential Monte Carlo with ESS-based systematic
resampling and Metropolis-Hastings rejuvenation. Exact enumeration remains available
for small hypothesis sets and regression checks.

## Planning API

A behavior model combines a planner with an explicit action-likelihood model:

```julia
behavior = BehaviorModel(
    POMDPSolverPlanner(mdp -> MySolver()),
    EpsilonGreedyLikelihood(epsilon = 0.05),
)
```

Arrodes is built directly on MuKumari and includes native planners for Crux Soft-Q,
MuKumari belief-state MCTS/DPW, VulcanJ risk-bounded information MCTS, and VulcanJ
one-shot ergodic paths. It also accepts any `POMDPs.Solver`/`POMDPs.Policy` pair,
callbacks, and open-loop trajectory planners. Likelihoods include score softmax,
epsilon-greedy decisions, plan tracking, and user callbacks.

External packages can integrate by defining `prepare`, `planned_action`, and optionally
`action_scores` for a subtype of `AbstractPlanner`.

MuKumari supplies the simulation state, dynamics, environment, geometry, and MDP
scaffolding throughout Arrodes; it is not an optional compatibility layer.

## Minimal inference

```julia
hypotheses = [
    ObjectiveHypothesis(
        id = :goal,
        objective = goal_reward,
        behavior = BehaviorModel(
            POMDPSolverPlanner(mdp -> GoalSolver()),
            EpsilonGreedyLikelihood(epsilon = 0.05),
        ),
        prior_probability = 0.7,
    ),
    ObjectiveHypothesis(
        id = :explore,
        objective = information_reward,
        behavior = BehaviorModel(
            OpenLoopPlanner(ergodic_plan),
            PlanTrackingLikelihood(epsilon = 0.1),
        ),
        prior_probability = 0.3,
    ),
]

config = DiscreteInferenceConfig(
    hypotheses = hypotheses,
    mdp_builder = (objective, hypothesis) -> build_mdp(objective),
)

smc = SMCInferenceConfig(model = config, n_particles = 512,
    ess_threshold = 0.5, rejuvenation_steps = 2)
result = infer_objectives_smc(smc, observed_states, observed_actions)
best_hypothesis(result)
```

## Visual diagnostics

The full animation surface is retained for the discrete system:

- `plot_particle_filter_frame(...; trace_from_current=false)` compares plans from the
  shared starting state;
- `plot_particle_filter_frame(...; trace_from_current=true)` compares replanned paths
  from the current observation;
- `plot_particle_heatmaps_frame` compares the true and candidate objective fields;
- `plot_particle_filter_explanation` creates the final diagnostic view;
- the two frame factories and `animate_particle_filter_from_frames` assemble GIFs.

The historical `particle_filter` wording remains in these visualization function names
to keep the diagnostic workflow recognizable.

Deterministic and open-loop planners can use `MovementNoiseLikelihood`, which samples
the MuKumari transition model to construct an empirical distribution around the plan.
Legacy RFF/RBF objective-field utilities remain exported with deprecation warnings and
are not accepted as an implicit objective hypothesis space.

See `examples/pipelines/default_pipeline.jl` for the baseline MuKumari example.
`examples/pipelines/ergodic_ipp_pipeline.jl` is the complete VulcanJ volcano-search example:
five GP mission objectives, mixed VulcanJ InfoMCTS/ergodic planners, a hidden true
objective, SMC inference, transition-noise scoring, and diagnostic animations.
`examples/pipelines/ayton_query_inference_pipeline.jl` is the compact reference for
writing and inferring over every class of Ayton query.
The planner and filtering contracts are described in `docs/architecture.md`.
