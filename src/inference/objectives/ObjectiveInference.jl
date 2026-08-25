module ObjectiveInference

using Gen: @gen
import Gen
import GenParticleFilters
import GenSMCP3
import GenTraceKernelDSL
using GenSMCP3: @kernel
using LinearAlgebra
using Parameters: @with_kw_noshow
import POMDPs
using Random

using ...BehaviorModels: BehaviorModel, observation_loglikelihood, prepare_behavior

include("types.jl")
include("action_likelihood.jl")
include("inference.jl")

export ObjectiveHypothesis, ObjectiveInferenceProblem, ObjectiveInferenceResult
export infer_objectives, objective_probabilities, best_hypothesis

end
