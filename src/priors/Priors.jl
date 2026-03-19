module Priors

using Parameters: @with_kw
using LinearAlgebra, Statistics, Random
using Distributions: Categorical
using Gen

using MuKumari

import ..Arrodes: FourierDiscreteCfg, ScoreΠDist, PriorDiscreteCfg, RBFDiscreteCfg
import ..Utils: randcat

include("generics.jl")
export ComponentField,
    component_type,
    sample_component_params,
    make_component,
    describe_component_params

include("fields.jl")
export K_probs,
    make_pomdp_objective_from_field,
    objective_grid_from_field,
    objective_grid_from_mdp

include("fourier.jl")
export RandomFourierField

include("rbf.jl")
export RadialBasisField


end
