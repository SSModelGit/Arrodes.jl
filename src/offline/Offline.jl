module Offline

using LinearAlgebra
using Statistics

using ..WorldInference: WorldInferenceProblem, TrajectoryObservation, target_measure
import ..WorldInference

export calibrate_discrepancy_scale

"""Estimate the characteristic discrepancy unit used by behavioral energy."""
function calibrate_discrepancy_scale(context, score, coefficient_samples)
    problem = WorldInferenceProblem(
        context=context,
        score=score,
        observations=TrajectoryObservation[],
    )
    measures = [target_measure(problem, coefficients) for coefficients in coefficient_samples]
    kernel = WorldInference.kernel_matrix(
        score.kernel_bandwidth,
        context.kernel_locations,
        context.kernel_locations,
    )
    discrepancies = [
        dot(measures[left] - measures[right], kernel * (measures[left] - measures[right]))
        for left in 2:length(measures) for right in 1:left-1
    ]
    max(median(discrepancies), sqrt(eps(Float64)))
end

end
