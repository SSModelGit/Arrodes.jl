module Inference
    
import ..Arrodes: FourierDiscreteCfg, ScoreΠDist

include("gen_model.jl")
include("particle_filter.jl")

end