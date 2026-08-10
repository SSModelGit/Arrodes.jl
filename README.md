# Arrodes

Arrodes is a behavior-inference engine for distributed agents. It solves two
complementary inverse problems:

- objective inference: the world is known and the agent's objective is one of a
  finite set of domain-authored hypotheses;
- world inference: the objective is known, environmental observations are hidden,
  and the agent's effective world belief is inferred in the ego agent's SCRIBE EOF
  coordinates.

Both problems use one sequential target-ratio runtime for weighted particles,
SMC-P3 paired proposals, normalization, ESS/CESS, resampling, ancestry, and
diagnostics. They retain different latent spaces and evidence models. MuKumari owns
MDP simulation and SCRIBE owns EOF modeling and physical sensor assimilation;
Arrodes owns inversion from behavior.

The current development environment expects sibling checkouts of the three local
packages. Resolve them once with Julia's package manager:

```julia
using Pkg
Pkg.develop([
    Pkg.PackageSpec(path="../MuKumari"),
    Pkg.PackageSpec(path="../VulcanJ"),
    Pkg.PackageSpec(path="../SCRIBE"),
])
```

## Objective inference

An `ObjectiveHypothesis` contains a stable ID, objective, prior mass, planner, and
behavior likelihood. Exact enumeration is the reference backend. SMC uses identity
propagation by default and full-prefix replay Metropolis–Hastings only as an
invariant move after resampling.

```julia
problem = ObjectiveInferenceProblem(
    hypotheses=[
        ObjectiveHypothesis(
            id=:goal,
            objective=goal_reward,
            behavior=BehaviorModel(
                POMDPSolverPlanner((mdp, context) -> GoalSolver()),
                EpsilonGreedyLikelihood(epsilon=0.05),
            ),
            prior_probability=0.7,
        ),
        ObjectiveHypothesis(
            id=:explore,
            objective=information_reward,
            behavior=BehaviorModel(
                OpenLoopPlanner(ergodic_plan),
                PlanTrackingLikelihood(epsilon=0.1),
            ),
            prior_probability=0.3,
        ),
    ],
    mdp_builder=(objective, hypothesis) -> build_mdp(objective),
)

config = SMCConfig(
    n_particles=512,
    invariant_move=ObjectiveReplayMove(),
    invariant_steps=2,
)
result = infer_objectives_smc(problem, observed_states, observed_actions, config)
best_hypothesis(result)
```

Built-in planner adapters cover Crux Soft-Q, MuKumari MCTS/DPW, VulcanJ
InfoMCTS, VulcanJ ergodic paths, any `POMDPs.Solver`, callbacks, and open-loop
optimizers. Planner construction has one canonical signature; planning and
demonstrator noise remain separate.

## Behavior-only world inference

`scribe_world_context` freezes one SCRIBE EOF basis, model time, current
coefficient vector, and coefficient covariance for an inference window. A direct
ergodic evidence model combines:

- the complete three-term kernel MMD between trajectory occupation and a normalized
  candidate-world target measure;
- mean trajectory reward;
- fixed prior-predictive calibration scales;
- a Hill maturity schedule and UCB-inspired reward/discrepancy mixture.

Each observed location creates one natural behavior-prefix target. There is no
additional `λ`-tempered world target. `WorldKernelMixture` moves particles toward that
target through distance-controlled bridges in SCRIBE-prior-whitened EOF coordinates.
Its local branch uses a positive behavioral Fisher/Gauss--Newton information metric;
the other branches provide affine amortized and prior-refresh support. The SMC-P3
correction includes old/new targets, forward/backward densities, branch mass, and
coordinate density terms.

`PriorPCNKernel` is the inexpensive prior-reversible baseline for larger EOF
studies. It preserves exact paired-proposal accounting but is not target-adapted.
The default `AffineAmortizedTransport` is likewise an affine baseline: a genuinely
amortized trajectory-to-world proposal still requires user-supplied parameters
trained on prior-predictive trajectories.

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
config = SMCConfig(
    scheduler=OneStagePerObservation(),
    kernel=WorldKernelMixture(),
    paired_moves_per_stage=2,
)
result = infer_world(problem, trajectory_observations, config)
```

World inference accepts behavior only. If the observed agent's environmental
measurements or information state are available, use SCRIBE's conditioning or
consensus mechanics directly. The example
`scribe_direct_observation_bypass.jl` demonstrates that boundary.

Weighted particles are authoritative. `blended_coefficients` performs the default
mean-only deployment through a maturity/confidence-weighted trust-radius step from
the ego coefficient vector. `deploy_behavior_information` can separately derive a
Gaussian pseudo-information increment for SCRIBE consumers, but accepts it exactly
only when the increment is positive semidefinite. A requested PSD projection is
reported as lossy, and the resulting `KFEnvInfo` has zero sensor-innovation fields.

## Physical time and distributed agents

`DynamicWorldInferenceProblem` carries coefficient paths and applies SCRIBE's `Q`
once per real environment transition—never per observation bridge or proposal move.
`DistributedWorldInferenceProblem` requires explicit shared, agent-specific, or
hierarchical beliefs and explicit conditionally independent or joint behavior
evidence. Parallel evaluation therefore never silently asserts independence.

## Examples and literature

Maintained pipelines live in `examples/pipelines/` and write under a matching
`res/<pipeline-name>/` directory:

- `default_pipeline.jl`: small named-objective inference and all objective animations;
- `ergodic_ipp_pipeline.jl`: five VulcanJ volcano-search objectives using InfoMCTS
  and ergodic planning;
- `scribe_world_inference_pipeline.jl`: behavior-only world inference on the ROMS
  archive used by SCRIBE. It directly reuses SCRIBE's ROMS example loader and EOF
  settings. The SCRIBE scenario snapshots 3030 and 5302 define the ego and
  observed-agent coefficient vectors, VulcanJ generates a 100-location
  kernel-ergodic trajectory, and Arrodes runs matched combined, MMD-only, and
  reward-only filters over all 100 locations. Its field animation is a 2×2 comparison
  with the static observed-agent EOF mean and growing trajectory in the top-left,
  combined inference in the top-right, MMD-only inference in the bottom-left, and
  reward-only inference in the bottom-right. It also produces the combined
  coefficient comparison and consolidated particle-health diagnostics;
- `scribe_direct_observation_bypass.jl`: direct SCRIBE assimilation when measurements
  are available;
- `ayton_query_inference_pipeline.jl`: domain-authored query objectives.

The mathematical designs, teaching treatment, unified architecture, and implementation
goal/style guide are in `literature/`. The executable architecture is described in
`docs/architecture.md`; the current world-branch audit and its remaining limitations
are recorded in `docs/world_inference_audit.md`.

Generic RFF/RBF objective fields remain isolated under `src/deprecated/` for old
experiments. They are not part of either inference target.
