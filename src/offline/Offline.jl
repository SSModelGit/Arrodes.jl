module Offline

using LinearAlgebra
using Statistics

using ..WorldInference: EOFWorldInferenceContext, SOMWorldInferenceContext,
    WorldInferenceProblem, ErgodicBehaviorScore, TrajectoryObservation,
    target_measure, kernel_matrix

export calibrate_discrepancy_scale

function candidate_discrepancy_scale(context, score, candidates)
    problem = WorldInferenceProblem(
        context=context,
        score=score,
        observations=TrajectoryObservation[],
    )
    measures = [target_measure(problem, c) for c in candidates]
    kernel = kernel_matrix(
        score.kernel_bandwidth,
        context.kernel_locations,
        context.kernel_locations,
    )
    discrepancies = [
        dot(
            measures[left] - measures[right],
            kernel * (measures[left] - measures[right]))
        for left in 2:length(measures) for right in 1:left-1
    ]
    max(median(discrepancies), sqrt(eps(Float64)))
end

function calibrate_discrepancy_scale(
    context::EOFWorldInferenceContext,
    score::ErgodicBehaviorScore,
    coefficient_samples
)
    candidate_discrepancy_scale(context, score, coefficient_samples)
end

function calibrate_discrepancy_scale(
    context::SOMWorldInferenceContext,
    score::ErgodicBehaviorScore
)
    candidate_discrepancy_scale(context, score, axes(context.fields, 2))
end

end
