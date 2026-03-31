using Parameters: @with_kw
using Gen

export FourierDiscreteCfg, ScoreΠDist, MuEnvSpec, METHOD_LABELS, RunPack, actiondirac,
    PriorDiscreteCfg, RBFDiscreteCfg,
    ComponentField, RandomFourierField, RadialBasisField,
    RLConfig, InferenceConfig

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

"""
    ComponentField

Abstract supertype for component objective field definitions.

Each concrete component field type (e.g., `RandomFourierField`, `RadialBasisField`) 
implements the required interface functions:
- `component_type(::Type{CF}) -> String`
- `sample_component_params(::Type{CF})` (must be @gen function)
- `make_component(::Type{CF}, params::Dict) -> Function`
- `describe_component_params(::Type{CF}) -> String` (optional)
"""
abstract type ComponentField end

"""
    RandomFourierField <: ComponentField

A continuous Fourier component field type for the new ComponentField API.

Represents sinusoidal component objectives with configurable continuous parameter
distributions. Replaces the old discrete implementation using Gen.jl for
probabilistic parameter sampling.

Parameters:
- `amplitude_max::Float64`: Maximum amplitude for uniform sampling [0, amplitude_max]
- `freq_max::Float64`: Maximum frequency for uniform sampling [0, freq_max]
"""
@with_kw struct RandomFourierField <: ComponentField
    amplitude_max::Float64 = 10.0
    freq_max::Float64 = π
end

"""
    RadialBasisField <: ComponentField

A continuous Radial Basis Function (RBF) component field type for the new ComponentField API.

Represents Gaussian RBF component objectives with configurable continuous parameter
distributions. Replaces the old discrete implementation using Gen.jl for
probabilistic parameter sampling.

Parameters stored in struct to shape the distributions:
- `x_min::Float64`: Minimum x-coordinate for center sampling
- `x_max::Float64`: Maximum x-coordinate for center sampling
- `y_min::Float64`: Minimum y-coordinate for center sampling
- `y_max::Float64`: Maximum y-coordinate for center sampling
- `amp_min::Float64`: Minimum amplitude for uniform sampling
- `amp_max::Float64`: Maximum amplitude for uniform sampling
- `σ::Float64`: Gaussian bandwidth (fixed, not sampled)
"""
@with_kw struct RadialBasisField <: ComponentField
    x_min::Float64 = -5.0
    x_max::Float64 = 5.0
    y_min::Float64 = -5.0
    y_max::Float64 = 5.0
    amp_min::Float64 = 0.1
    amp_max::Float64 = 10.0
    σ::Float64 = 0.5
end

@with_kw struct RunPack
    run_id::Int                 # top-level run index in the BSON
    agent::String               # "ag1".."ag7"
    inst::Int                   # instance index (k)
    mdp::Any                    # KAgentPOMDP
    full::Any                   # ExperienceBuffer (full)
    anon::Any                   # ExperienceBuffer (anon; used for IQL)
    ann::NamedTuple             # (num_goals, num_obstacles, max_goal_separation)
end

"""
    RLConfig

Configuration structure for SoftQ-learn hyperparameters in component generative inference.

Fields:
- `temperature::Float64`: Boltzmann temperature for policy (default: 1.0)
- `n_iterations::Int`: Number of SoftQ iterations per particle (default: 100)
- `learning_rate::Float64`: SoftQ learning rate (default: 0.01)
- `value_reg::Float64`: Value function regularization coefficient (default: 0.001)
- `n_samples_per_state::Int`: Number of samples for value estimation (default: 10)
- `epochs::Int`: Number of training epochs for SoftQ solver (default: 2)
- `batch_size::Int`: Batch size for mini-batch updates in SoftQ solver (default: 512)
"""
@with_kw struct RLConfig
    temperature::Float64 = 1.0
    n_iterations::Int = 100
    learning_rate::Float64 = 0.01
    value_reg::Float64 = 0.001
    n_samples_per_state::Int = 10
    epochs::Int = 2
    batch_size::Int = 512
end

"""
    InferenceConfig

Configuration structure that bundles all inference parameters for generative model learning.

Fields:
- `component_tuples::Vector{Tuple}`: Vector of (ComponentField, sampling_function) tuples
- `component_params_switch::Gen.Switch`: Gen switch for sampling component parameters
- `component_type_sampler::Function`: Function to sample component types from available tuples.
- `k_components::Integer`: Number of components to generate (note: NOT the number of component types)
- `rl_config::RLConfig`: SoftQ-learn hyperparameters (default: RLConfig())
- `agent_params::Dict{String, Any}`: Additional agent-specific parameters (default: Dict())
- `iterative_deepening::Bool`: Enable iterative deepening search (default: false)
- `metadata::Dict{String, Any}`: Problem-specific metadata (default: Dict())
"""
@with_kw struct InferenceConfig
    component_tuples::Vector{Tuple}
    component_params_switch::Gen.Switch
    component_type_sampler
    k_components::Integer = 1
    rl_config::RLConfig = RLConfig()
    agent_params::Dict = Dict()
    iterative_deepening::Bool = false
    metadata::Dict = Dict()
end