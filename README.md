# Arrodes

Arrodes infers hidden causes of observed agent behavior through two complementary
inverse problems:

- objective inference learns which finite, domain-authored objective explains
  behavior in a known world;
- world inference learns the observed agent's effective SCRIBE EOF world belief
  from behavior under a known objective.

Arrodes is an inference package, not a robotics system architecture. MuKumari
owns agent MDPs and simulation, VulcanJ owns information and ergodic planning,
and SCRIBE owns EOF environment models and physical observation updates. Gen,
GenParticleFilters, and GenSMCP3 execute both particle filters.

## Package structure

The source tree contains five actual Julia modules:

- `BehaviorModels` directly invokes supported native planners when an objective
  particle must predict behavior;
- `ObjectiveInference` defines the discrete Gen trace and its SMCP3 proposals;
- `WorldInference` defines the continuous SCRIBE-coordinate Gen trace, scores,
  and paired forward/backward proposals;
- `Offline` provides the small amount of calibration used by executable
  missions;
- `Visualizations` contains reusable objective- and world-inference plots.

There is no Arrodes particle-filter runtime. Gen represents uncertain traces,
GenParticleFilters owns particles, weights, ESS, resampling, and rejuvenation,
and GenSMCP3 evaluates the paired forward/backward proposal correction.

## Objective inference

An objective trace contains a discrete objective identity. The target after
observing actions through time `t` is

```math
\gamma_t(h)=\rho_h\exp\!\left[\sum_{r=1}^t \ell_r(h)\right].
```

The forward kernel uses a reversible persistence/refresh transition over the
finite objective set. The backward kernel uses the corresponding conditional
law. Native planners are invoked only to evaluate
`p(observed action | objective, known world)`; Arrodes does not execute the
mission planner.

```julia
hypotheses = [
    ObjectiveHypothesis(
        id=:goal,
        objective=goal_reward,
        behavior=BehaviorModel(
            MCTSPlanner(n_iterations=500),
            EpsilonGreedyLikelihood(epsilon=0.05),
        ),
        prior_probability=0.6,
    ),
    ObjectiveHypothesis(
        id=:explore,
        objective=information_reward,
        behavior=BehaviorModel(
            VulcanErgodicPlanner(
                gp=build_gp,
                n_steps=20,
                options=Dict(:optimizer_iters => 80),
            ),
            MovementNoiseLikelihood(),
        ),
        prior_probability=0.4,
    ),
]

problem = ObjectiveInferenceProblem(
    hypotheses=hypotheses,
    mdp_builder=(objective, hypothesis) -> build_mdp(objective),
    states=observed_states,
    actions=observed_actions,
)

result = infer_objectives(problem; n_particles=512)
probabilities = objective_probabilities(result)
```

Exact enumeration remains an internal validation oracle for small finite sets;
it is not a second inference architecture.

## World inference

A world trace contains one candidate SCRIBE EOF coefficient vector for the
observed agent's effective world belief. The calling application supplies a
frozen SCRIBE model, prior covariance, target measure, and observed trajectory.
Arrodes never mutates a live SCRIBE information state.

For ergodic behavior, the generalized log score uses the MMD between trajectory
occupancy and the candidate-world target measure. A known query score can be
added through an explicitly visible horizon schedule, which must be calibrated
offline for that query. There is no implicit online UCB fallback.

```julia
context = world_inference_context(
    ego_eof_model;
    prior_covariance=observed_world_prior_covariance,
)
target = eof_target_field(link=:magnitude)
score = eof_field_score(
    target;
    kernel_bandwidth=0.2,
    discrepancy_scale=calibrated_discrepancy_scale,
    β_max=7.0,
    maturity_half_time=18.0,
)
problem = WorldInferenceProblem(
    context=context,
    score=score,
    observations=trajectory,
)

proposal = default_world_proposal()
result = infer_world(problem; n_particles=512, proposal)
posterior = world_posterior(result)
```

The default proposal is the symmetric SCRIBE-process random walk

```math
\phi^+\sim\mathcal N(\phi^-,c_QQ_\phi),
```

with the matching reversed Gaussian used by GenSMCP3. The mission selects one
proposal mechanism for the whole run. Three alternatives remain available:

- pCN supplies prior-reversible correlated movement using the initial SCRIBE
  covariance `P₀`;
- Langevin uses the target-score gradient and a fixed preconditioner derived
  from the initial SCRIBE covariance `P₀`;
- Gauss–Newton transport moves the existing particle cloud between consecutive
  observation-conditioned Gaussian approximations and uses the inverse affine
  map as its GenSMCP3 backward program.

These mechanisms are alternatives, not a mixture. Changing the random-walk
scale does not change the initial ego-centered prior.

World inference is used only when the observed agent's measurements and
information state are hidden. Available measurements should be assimilated by
SCRIBE directly.

## Native solver support

Objective hypotheses can use:

- an existing `POMDPs.Solver` directly;
- Crux Soft-Q;
- standard or DPW MCTS;
- VulcanJ risk-bounded InfoMCTS;
- VulcanJ one-shot ergodic planning;
- a fixed known action for compact demonstrations and validation.

The adapters are direct multiple-dispatch methods. Arrodes does not define a
policy-artifact hierarchy, callback-planner framework, cache-scope API, or
fallback dependency probing.

## Examples

Examples are research missions, not APIs. Each mission has one JSON dictionary
and a direct script:

- `examples/objective_inf/default_pipeline.jl`;
- `examples/objective_inf/ergodic_ipp_pipeline.jl`;
- `examples/world_inf/direct_environment.jl`;
- `examples/world_inf/curl_mmd_multi_trial.jl`;
- `examples/world_inf/score_comparison.jl`.

The score-comparison mission preserves the requested MMD/query/combined
visual ablation, but it is not the final Spock validation: its checked-in
mixing schedule is an explicit mission artifact. Query-specific supervised
calibration and non-ergodic Ayton-query behavior remain deferred until
SpockQueryFramework exists.

The default objective mission retains plans-from-start, plans-from-current,
objective-heatmap animations, and the final explanation plot. Reusable world
posterior, coefficient, particle-distribution, and particle-health compositions
and animations live in `Arrodes.Visualizations`. Static SCRIBE posterior maps
already live in SCRIBE, while ROMS field maps and curl magnitude/quiver maps
are supplied by `SCRIBE.ROMSTools`. Only aggregate multi-trial plots remain
beside the curl mission. ROMS loading, coordinate conversion, curl construction,
decimation, EOF preparation, and grid conversion also belong to
`SCRIBE.ROMSTools`. Mission-specific ergodic quadrature selection remains with
the world-inference missions.

Every world-inference result produces final posterior and coefficient
comparisons, particle-distribution and particle-health plots, and posterior and
coefficient animations. Score ablations store the full set under `combined/`,
`mmd/`, and `query/`. Multi-trial missions store the full set under one
`trial_XX/` directory per trial in addition to their aggregate plots.

The first world validation is:

```sh
julia --project=examples examples/world_inf/curl_mmd_multi_trial.jl
```

It learns a rank-10 SCRIBE model of normalized absolute-curl shape, generates
VulcanJ trajectories ergodic to that field, and performs ten MMD-only inference
trials spanning nearby through distant observed-agent beliefs. The mission uses
one observation-conditioned Gauss–Newton affine proposal throughout the
GenSMCP3 run. After resampling, the same Gaussian approximation supplies an
independence-MH rejuvenation move; this is a resample–move step, not a mixture
of transport proposals. Curl is treated as
scalar out-of-plane vorticity; equal-length arrows carry only the direction of
the ROMS horizontal flow. VulcanJ and Arrodes evaluate the Gaussian kernel in
the same unit-square domain coordinates. The discrepancy unit is calibrated
from held-out target fields. The evidence temperature is selected from
coefficient recovery on nine separate held-out VulcanJ trajectories; none of
the ten validation worlds participates in calibration. Posterior plots show the
query-observable EOF world model and its normalized absolute-curl target.
Coefficient recovery,
representative particles, and posterior-predictive target recovery answer
different questions and are displayed together; none is treated as a substitute
for the others.

The examples run trials sequentially and set BLAS to one thread. Arrodes does
not run examples during package tests.

## Development

Development expects sibling checkouts of MuKumari, VulcanJ, and SCRIBE and the
unregistered GenSMCP3 dependencies already recorded in the manifest. Dependency
installation is intentionally left to the user.

Tests are two small usage demonstrations rather than API, compilation, or
component-verification suites. One toy trajectory demonstrates objective
inference, and one toy EOF field demonstrates world inference. The ROMS,
planner, visualization, and multi-trial studies remain executable missions in
`examples/` and are not run as package tests.
