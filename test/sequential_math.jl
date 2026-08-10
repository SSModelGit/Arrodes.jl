struct QuadraticProblem <: AbstractBehaviorInferenceProblem end
Arrodes.logtarget(::QuadraticProblem, stage::InferenceStage, value, cache) =
    -0.5 * (value - stage.observation)^2

@testset "Sequential target-ratio mathematics" begin
    values = [0.0, -1.0, -2.0]
    @test Arrodes.Inference.logsumexp(values) ≈ log(sum(exp, values))

    particles = [WeightedParticle(value=i, trace=nothing, log_weight=value, lineage=i)
                 for (i, value) in enumerate(values)]
    normalizer = Arrodes.Inference.normalize_logweights!(particles)
    @test normalizer ≈ Arrodes.Inference.logsumexp(values)
    @test sum(exp(particle.log_weight) for particle in particles) ≈ 1.0
    @test 1 <= effective_sample_size(particles) <= length(particles)

    weights = exp.([particle.log_weight for particle in particles])
    ancestors = Arrodes.Inference.systematic_resample(MersenneTwister(3), weights)
    @test length(ancestors) == length(particles)
    @test all(index -> index in eachindex(particles), ancestors)

    old_stage = InferenceStage(observation=0)
    new_stage = InferenceStage(observation=1)
    particle = WeightedParticle(value=0.2, trace=nothing, log_weight=0.0, lineage=1)
    move = MoveRecord(
        value=0.8,
        log_forward=-0.7,
        log_backward=-0.4,
        log_jacobian=0.1,
    )
    expected = Arrodes.logtarget(QuadraticProblem(), new_stage, 0.8, Dict()) -
        Arrodes.logtarget(QuadraticProblem(), old_stage, 0.2, Dict()) - 0.4 + 0.7 + 0.1
    @test Arrodes.Inference.paired_logweight(
        QuadraticProblem(), old_stage, new_stage, particle, move, Dict(),
    ) ≈ expected

    increments = [0.0, -0.5, -1.0]
    @test 0 < conditional_effective_sample_size(
        [particle.log_weight for particle in particles], increments,
    ) <= length(particles)
end
