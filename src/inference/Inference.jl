module Inference

using LinearAlgebra
using Match: @match
using Parameters: @with_kw, @with_kw_noshow
using POMDPs
using Random
using SCRIBE
using Statistics

using ..Planning

include("sequential/types.jl")
include("sequential/numerics.jl")
include("sequential/kernels.jl")
include("sequential/runtime.jl")

include("objectives/types.jl")
include("objectives/behavior_evidence.jl")
include("objectives/exact_filter.jl")
include("objectives/kernels.jl")
include("objectives/inference.jl")

include("worlds/types.jl")
include("worlds/scribe_context.jl")
include("worlds/trajectory_statistics.jl")
include("worlds/behavior_evidence.jl")
include("worlds/target.jl")
include("worlds/kernels.jl")
include("worlds/dynamic.jl")
include("worlds/distributed.jl")
include("worlds/deployment.jl")
include("worlds/inference.jl")
include("worlds/diagnostics.jl")

end
