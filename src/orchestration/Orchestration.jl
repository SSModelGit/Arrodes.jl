module Orchestration

using LinearAlgebra
import GeoInterface as GI
using MuKumari
using Parameters: @with_kw_noshow

@with_kw_noshow struct MuEnvSpec
    M::Int = 3
    μ_order::Vector{Symbol} = [:sin, :exp, :lin]
end
    
include("mukumari.jl")
export MuEnvSpec, KAgentMDPConfig, build_kagent_pomdp, build_shared_menv,
       agent_config_from_mdp

end
