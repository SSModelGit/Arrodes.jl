module Arrodes

using Reexport

# Write your package code here.
include("types.jl")

include(joinpath("utils", "Utils.jl"))
@reexport using .Utils

include(joinpath("priors", "Priors.jl"))
@reexport using .Priors

include(joinpath("core", "Core.jl"))
@reexport using .Core

include(joinpath("rl", "RL.jl"))
@reexport using .RL

include(joinpath("inference", "Inference.jl"))
@reexport using .Inference

include(joinpath("analysis", "Analysis.jl"))
@reexport using .Analysis

include(joinpath("viz", "Visualizations.jl"))
@reexport using .Visualizations

end
