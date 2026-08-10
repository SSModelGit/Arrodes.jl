@testset "Declared distributed belief models" begin
    context = tiny_scribe_context()
    agents = [
        AgentBehaviorEvidence(id=:alpha, evidence=tiny_ergodic_evidence()),
        AgentBehaviorEvidence(id=:beta, evidence=tiny_ergodic_evidence()),
    ]
    problem = DistributedWorldInferenceProblem(
        context=context,
        agents=agents,
        belief_model=SharedWorldBelief(),
        coupling=ConditionallyIndependentEvidence(),
    )
    observations = [
        DistributedTrajectoryObservation(
            agent_id=:alpha,
            observation=TrajectoryObservation(state=[0.0, 0.0], action=:move),
        ),
        DistributedTrajectoryObservation(
            agent_id=:beta,
            observation=TrajectoryObservation(state=[1.0, 1.0], action=:move),
        ),
    ]
    result = infer_distributed_world(
        problem, observations,
        SMCConfig(n_particles=16, kernel=IdentityKernel(), seed=UInt64(11)),
    )
    summary = posterior(result)
    @test summary.agent_ids == [:alpha, :beta]
    @test summary.agent_means[1] == summary.agent_means[2]
    @test !isnothing(summary.common_mean)

    hierarchical = DistributedWorldInferenceProblem(
        context=context,
        agents=[AgentBehaviorEvidence(id=:alpha, evidence=tiny_ergodic_evidence())],
        belief_model=HierarchicalWorldBelief(offset_scale=0.2),
        coupling=JointBehaviorCompatibility(
            logcompatibility=(problem, value, omit_latest, cache) -> 0.0,
        ),
    )
    initialized = initialize_smc(
        hierarchical,
        SMCConfig(n_particles=6, kernel=IdentityKernel(), seed=UInt64(12)),
    )
    @test first(initialized.cloud.particles).value isa HierarchicalWorldCoefficients
end
