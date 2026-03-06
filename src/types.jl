using Parameters: @with_kw
using Gen

export FourierDiscreteCfg, ScoreΠDist, MuEnvSpec, METHOD_LABELS, RunPack, actiondirac, PriorDiscreteCfg, RBFDiscreteCfg

"""
    PriorDiscreteCfg

Abstract supertype for discrete prior configurations.
Concrete subtypes include FourierDiscreteCfg and RBFDiscreteCfg.
"""
abstract type PriorDiscreteCfg end

@with_kw struct FourierDiscreteCfg <: PriorDiscreteCfg
    Kmax::Int = 10
    λK::Float64 = 0.35            # P(K=k) ∝ exp(-λK*(k-1))

    # frequency grid
    Δf::Float64 = 0.1
    Fmax_i::Int = 10              # bins in -Fmax_i:Fmax_i

    # amplitude grid
    ΔA::Float64 = 0.1
    Amax_i::Int = 3               # bins in 0:Amax_i

    # phase grid
    P::Int = 32                   # bins in 0:P-1

    # optional: bias towards lower |freq|
    freq_mag_decay::Float64 = 0.0
end

"""
    RBFDiscreteCfg

Configuration for radial basis function (RBF) field prior.

# Fields
- `N_max::Int`: Maximum number of RBF centers (Gaussians)
- `λN::Float64`: Exponential decay parameter for P(N=n) ∝ exp(-λN*(n-1))
- `σ::Float64`: Standard deviation (bandwidth) of each Gaussian kernel
- `x_min, x_max, y_min, y_max::Float64`: Spatial bounds for center placement
"""
@with_kw struct RBFDiscreteCfg <: PriorDiscreteCfg
    Kmax::Int = 5
    λK::Float64 = 0.5             # P(N=n) ∝ exp(-λN*(n-1))
    
    # Gaussian bandwidth
    σ::Float64 = 1.0
    
    # Spatial bounds for RBF center placement
    x_min::Float64 = 0.0
    x_max::Float64 = 10.0
    y_min::Float64 = 0.0
    y_max::Float64 = 10.0
end

@with_kw struct ScoreΠDist
    ## dynamic/open-ended objective ids (Fourier keys)
    prop_names::Vector = []
    # q_objs::Dict
    # n_compobj_list::Dict
    ## prior weights per proposal (key => weight)
    n_qprop_list::Dict{Any,Float64} = Dict{Any,Float64}()
    ## mdp cache (key => mdp)
    n_propmdp_list::Dict{Any,Any} = Dict{Any,Any}()
    ## solver/policy caches (key => solver, policy)
    n_𝒮_proposals::Dict{Any,Any} = Dict{Any,Any}()
    n_π_proposals::Dict{Any,Any} = Dict{Any,Any}()
    # solver_type::Symbol = :dql
    # solver_params::Vector = [:softq, 10000]
    ## carries action mappings used by inference_model
    mdp_params::Vector = [] # [π_alist, π_a_1hot, π_a_1hotall] (by default)

    ### Open-Ended System Specific
    ## Fourier sampling config
    fourier_cfg::FourierDiscreteCfg = FourierDiscreteCfg()
end

@with_kw struct MuEnvSpec
    variant::Symbol = :default_shared
    M::Int = 3
    μ_order::Vector{Symbol} = [:sin, :exp, :lin]
end

struct ActionDirac <: Gen.Distribution{AbstractVector}
end
Gen.random(::ActionDirac, x::AbstractVector) = x
Gen.logpdf(::ActionDirac, v::AbstractVector, x::AbstractVector) = (argmax(v) == argmax(x)) ? 0.0 : -Inf
Gen.logpdf_grad(::ActionDirac, v, x) = (nothing,)
Gen.has_output_grad(::ActionDirac) = false
Gen.is_discrete(::ActionDirac) = true
const actiondirac = ActionDirac()
(::ActionDirac)(x::AbstractVector) = Gen.random(ActionDirac(), x)

const METHOD_LABELS = ["Open-Ended SIPS", "IQ-SIPS"]

@with_kw struct RunPack
    run_id::Int                 # top-level run index in the BSON
    agent::String               # "ag1".."ag7"
    inst::Int                   # instance index (k)
    mdp::Any                    # KAgentPOMDP
    full::Any                   # ExperienceBuffer (full)
    anon::Any                   # ExperienceBuffer (anon; used for IQL)
    ann::NamedTuple             # (num_goals, num_obstacles, max_goal_separation)
end