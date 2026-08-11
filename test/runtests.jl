ENV["GKSwstype"] = "100"

using Arrodes
using LinearAlgebra
using POMDPs
using Random
using SCRIBE
using Test

include("sequential_math.jl")
include("planning_pipeline.jl")
include("objective_pipeline.jl")
include("world_pipeline.jl")
include("distributed_pipeline.jl")
include("visualization_pipeline.jl")
