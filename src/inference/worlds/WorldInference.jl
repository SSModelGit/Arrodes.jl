module WorldInference

using Gen: @gen
import Gen
import GenParticleFilters
import GenSMCP3
import GenTraceKernelDSL
using LinearAlgebra
using Match: @match
using Parameters: @with_kw_noshow
using Random
using Statistics: mean
import SCRIBE

include("model.jl")
include("scoring.jl")
include("inference.jl")

export WorldInferenceContext, TrajectoryObservation, ErgodicTargetField
export ErgodicBehaviorScore, WorldInferenceProblem, WorldInferenceResult
export world_inference_context
export eof_target_field, eof_field_score, target_measure, target_measure_mmd
export posterior_target_measure
export kernel_discrepancy
export world_score_components
export default_world_proposal, infer_world, world_posterior

end
