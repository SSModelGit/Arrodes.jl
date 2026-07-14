module Inference

using Random
using POMDPs

using ..Planning

include("types.jl")
include("discrete_filter.jl")
include("smc_filter.jl")

export ObjectiveHypothesis,
    AbstractInferenceResult,
    DiscreteInferenceConfig,
    DiscreteFilterState,
    DiscreteFilterResult,
    ParticleTrace,
    SMCInferenceConfig,
    SMCFilterState,
    SMCFilterResult,
    initialize_filter,
    infer_objectives,
    infer_objectives_smc,
    initialize_smc,
    effective_sample_size,
    update!,
    posterior,
    log_posterior,
    best_hypothesis,
    hypothesis_index,
    hypothesis_mdp,
    hypothesis_artifact

end
