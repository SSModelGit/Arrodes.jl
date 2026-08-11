# Arrodes

Arrodes infers hidden causes of agent behavior. It supports two complementary
inverse problems:

- **objective inference:** infer which discrete, domain-authored objective explains
  behavior in a known world;
- **world inference:** infer the observed agent's effective world belief from
  behavior under a known objective.

Arrodes does not duplicate the surrounding robotics stack. MuKumari supplies agent
MDPs and simulation, VulcanJ supplies information-based and ergodic planners, and
SCRIBE supplies EOF environment models and physical observation updates. Arrodes
owns behavior evidence, sequential inference, and deployment of inferred
information.

## Architecture

The package has four public layers:

- `Orchestration` constructs MuKumari agents without taking ownership of their
  simulation mechanics.
- `Planning` provides a common behavior-model interface for Crux Soft-Q, MuKumari
  MCTS/DPW, VulcanJ InfoMCTS and ergodic planning, arbitrary `POMDPs.Solver`
  implementations, and user callbacks.
- `Inference` contains the shared SMC-P3 runtime and the distinct objective and
  world inference models.
- `Visualizations` contains the complete objective-filter and world-filter
  diagnostic animations.

The shared sequential runtime handles target-ratio weighting, paired forward and
backward proposal densities, ESS/CESS, resampling, rejuvenation, ancestry, and
stage diagnostics. Objective and world inference share that machinery while
retaining different latent states, targets, and proposal kernels.

### Objective inference

An `ObjectiveHypothesis` binds a stable identifier and prior probability to an
objective, planner, and action likelihood. Exact enumeration is available as a
reference for small hypothesis sets; SMC is the scalable implementation.

```julia
hypotheses = [
    ObjectiveHypothesis(
        id=:goal,
        objective=goal_reward,
        behavior=BehaviorModel(goal_planner, EpsilonGreedyLikelihood(epsilon=0.05)),
        prior_probability=0.6,
    ),
    ObjectiveHypothesis(
        id=:explore,
        objective=information_reward,
        behavior=BehaviorModel(ergodic_planner, PlanTrackingLikelihood(epsilon=0.1)),
        prior_probability=0.4,
    ),
]

problem = ObjectiveInferenceProblem(
    hypotheses=hypotheses,
    mdp_builder=(objective, hypothesis) -> build_mdp(objective),
)

result = infer_objectives_smc(
    problem,
    observed_states,
    observed_actions,
    SMCConfig(n_particles=512, invariant_move=ObjectiveReplayMove()),
)
```

### World inference

World inference operates in one frozen SCRIBE EOF coordinate system for each
inference window. Candidate coefficient vectors are evaluated from behavior using
kernel discrepancy, trajectory reward, or a calibrated combination of both.
`WorldKernelMixture` combines local Langevin transport, affine transport, and
prior-refresh proposals with exact forward/backward accounting.

```julia
context = scribe_world_context(eof_model, ego_information)
evidence = DirectErgodicEvidence(
    location=state -> state.x,
    reward=known_reward,
    importance=target_measure_link,
    kernel=GaussianDiscrepancyKernel(bandwidth=1.0),
    energy=WorldEnergyConfig(discrepancy_scale=s_D, reward_scale=s_R),
)
problem = WorldInferenceProblem(context=context, evidence=evidence)

result = infer_world(
    problem,
    trajectory_observations,
    SMCConfig(kernel=WorldKernelMixture(), paired_moves_per_stage=2),
)
```

World inference is only appropriate when the observed agent's environmental
measurements and information state are hidden. If either is available, it should be
assimilated directly with SCRIBE.

## Local dependencies

Development expects sibling checkouts of MuKumari, VulcanJ, and SCRIBE:

```julia
using Pkg

Pkg.develop([
    Pkg.PackageSpec(path="../MuKumari"),
    Pkg.PackageSpec(path="../VulcanJ"),
    Pkg.PackageSpec(path="../SCRIBE"),
])
Pkg.instantiate()
```

## Examples

The maintained pipelines are:

- `examples/pipelines/default_pipeline.jl` — compact discrete objective inference
  with the complete objective diagnostic animation set;
- `examples/pipelines/ergodic_ipp_pipeline.jl` — five volcano-search objectives
  using VulcanJ InfoMCTS and ergodic planners;
- `examples/pipelines/scribe_world_inference_pipeline.jl` — ROMS/SCRIBE world
  inference comparing combined, MMD-only, and reward-only evidence.

Examples write generated plots and animations to
`examples/pipelines/results/<pipeline>/`. That directory is ignored and created at
runtime. The ROMS example expects the SCRIBE archive at
`bigdata/rams_head_model_output/stjohn_hourly_5m_velocity_ramhead_v2.mat`; `bigdata/`
is intentionally ignored because the dataset is local input, not package source.

Run an example from its environment with:

```sh
julia --project=examples/pipelines examples/pipelines/default_pipeline.jl
```

## Validation

The tests are small executable pipeline checks rather than a separate mock
architecture:

```sh
julia --project=. -e 'using Pkg; Pkg.test()'
```

The mathematical treatments and implementation retrospectives in `literature/`
are personal design records and are not part of the package runtime.
