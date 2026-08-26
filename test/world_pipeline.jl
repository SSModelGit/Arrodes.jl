using Arrodes
using LinearAlgebra
using Random
using SCRIBE
using Test

function toy_world_problem()
    locations = [0.0 0.0; 1.0 0.0; 0.0 1.0; 1.0 1.0]
    snapshots = [
        0.0 1.0 0.2 1.1 0.1
        0.2 1.2 0.1 1.0 0.3
        1.0 0.0 1.1 0.1 0.9
        1.2 0.2 1.0 0.0 1.1
    ]
    covariance = 0.5 .* Matrix{Float64}(I, 2, 2)
    model = initialize_eof_climate_model(
        snapshots;
        locations,
        rank=2,
        process_covariance=0.02 .* covariance,
        prior_covariance=covariance,
    )
    context = world_inference_context(model; prior_covariance=covariance)
    target = eof_target_field(link=:softmax, scale=0.3)
    score = eof_field_score(
        target;
        kernel_bandwidth=0.7,
        discrepancy_scale=0.04,
        β_max=6.0,
        maturity_half_time=4.0,
    )
    truth = context.model.ϕ + [0.8, -0.5]
    target_world = WorldInferenceProblem(
        context=context,
        score=score,
        observations=TrajectoryObservation[],
    )
    masses = target_measure(target_world, truth)
    counts = floor.(Int, 48 .* masses)
    for index in sortperm(48 .* masses .- counts; rev=true)[1:48-sum(counts)]
        counts[index] += 1
    end
    observations = TrajectoryObservation[]
    for index in eachindex(counts), _ in 1:counts[index]
        push!(observations, TrajectoryObservation(
            state=vec(context.quadrature[index, :]),
        ))
    end
    WorldInferenceProblem(context=context, score=score, observations=observations), truth
end

function gaussian_logdensity(value, mean, covariance)
    difference = value - mean
    -0.5 * (
        length(value) * log(2π) +
        logdet(Symmetric(covariance)) +
        dot(difference, covariance \ difference)
    )
end

@testset "symmetric SCRIBE random-walk proposal" begin
    problem, _ = toy_world_problem()
    proposal = default_world_proposal()
    left = problem.context.model.ϕ + [0.2, -0.1]
    right = problem.context.model.ϕ + [-0.3, 0.4]
    left_moments = Arrodes.WorldInference.proposal_moments(
        left,
        problem,
        1,
        proposal,
        Dict{Symbol,Any}(),
    )
    right_moments = Arrodes.WorldInference.proposal_moments(
        right,
        problem,
        1,
        proposal,
        Dict{Symbol,Any}(),
    )

    @test left_moments[:covariance] ≈ right_moments[:covariance]
    @test gaussian_logdensity(
        right,
        left_moments[:mean],
        left_moments[:covariance],
    ) ≈ gaussian_logdensity(
        left,
        right_moments[:mean],
        right_moments[:covariance],
    )

end

@testset "world inference from a toy ergodic trajectory" begin
    problem, truth = toy_world_problem()
    prior_error = norm(problem.context.model.ϕ - truth)
    result = infer_world(
        problem;
        n_particles=128,
        ess_threshold=0.65,
        rejuvenation_steps=1,
        proposal=Dict{Symbol,Any}(
            :mechanism => :gauss_newton,
            :covariance_scale => 1.0,
            :optimizer_steps => 8,
        ),
        check_inverses=true,
        rng=MersenneTwister(18),
    )
    posterior = world_posterior(result)

    cache = Dict{Symbol,Any}()
    @test target_measure_mmd(
        problem,
        result,
        truth,
        cache,
    ) < target_measure_mmd(
        problem,
        problem.context.model.ϕ,
        truth,
        cache,
    )
    @test norm(posterior[:coefficient_mean] - truth) < prior_error
end
