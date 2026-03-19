# Arrodes Pipeline Redesign: Complete SMC³ + Continuous Component API

**Status:** Planning Phase  
**Last Updated:** March 2026  
**Reference Tag:** Latest work tagged and pushed (baseline for potential revert)

---

## Executive Summary

### Goal Direction

Perform a complete re-build of the Arrodes.jl inference pipeline without forcibly adhering to the legacy discretized approach that was hacked together incrementally. The new pipeline will be architecturally cleaner, mathematically principled, and extensible by users.

### Core Paradigm Shift

**From:** Discrete categorical sampling of pre-discretized Fourier/RBF modes  
**To:** Continuous multivariate distributions over component parameters with extensible component type API

**From:** Fixed hardcoded component types  
**To:** User-extensible API where users implement their own component types via multiple-dispatch

**From:** Basic SMC with ad-hoc rejuvenation  
**To:** SMC³ with warm starts and iterative deepening for open-ended SIPS

### Stated Approach (5-Step Implementation)

1. **Component Type API** - Define Julia type system and interface for user-defined component types with two reference implementations (Fourier + RBF, continuous parameters)
2. **Component Choice Distribution** - Assemble extensible distribution over which component type to select at each position
3. **Configuration Structure** - Create unified struct bundling user parameters, component registry, and type references
4. **Generative Inference Model** - Gen.jl-based model sampling K components, assembling objectives, learning Q-functions + Boltzmann policies, evaluating likelihoods
5. **SMC³ Particle Filter** - Sequential Monte Carlo for Open-Ended SIPS with warm starts and iterative deepening

---

## Part 1: Component Field API

### 1.1 Conceptual Foundation

The component field API is the foundation enabling user extensibility. It defines:
- Abstract type `ComponentField` for all component field types
- Required interface functions every component field type must implement
- Dispatch protocol for runtime polymorphism via multiple-dispatch

Users implement new component field types by:
1. Defining a concrete subtype of `ComponentField`
2. Implementing the three required interface functions
3. Optionally implementing the description function
4. Registering the type in the component registry passed to the inference model

### 1.2 Abstract Type Definition

```julia
# src/priors/generics.jl

"""
    ComponentField

Abstract supertype for all component field types used in the inference pipeline.

A component field type defines a class of scalar field objective functions characterized by
their parameter distributions. Concrete subtypes must implement the required interface
functions to be usable in inference.

The key insight: parameter sampling is traced via Gen.jl (@gen functions), allowing the
particle filter to track and modify which parameters are chosen for each component.
"""
abstract type ComponentField end
```

### 1.3 Required Interface Functions

Every concrete component field type must implement these 3 required functions via multiple-dispatch:

#### Function 1: `component_type(::Type{CF}) -> String`

Returns the human-readable name of this component field type.

**Purpose:** Identification and debugging  
**Input:** `CF::Type{<:ComponentField}` - Component field type (for dispatch)  
**Output:** `String` - Name of the component field type (e.g., "Fourier", "RBF", "UserCustom")  
**Implementation:** Return a constant string unique to this component field type

**Example:**
```julia
component_type(RandomFourierField) == "Fourier"
```

---

#### Function 2: `sample_component_params(::Type{CF})` (Gen-traced @gen function)

Sample a single instance of parameters for this component field type according to its distributions.

**Purpose:** Generate random parameters for this component field type via Gen.jl tracing  
**Input:** `CF::Type{<:ComponentField}` - Component field type (for dispatch)  
**Output:** `Dict{String, Any}` with parameter names as keys and sampled values as values

**Implementation:** Use Gen.jl `@gen` macro to trace parameter sampling. Each parameter must be sampled via a Gen random choice (e.g., `~ Uniform(...)`), allowing Gen.jl to track the trace for particle filtering.

**Key Requirement:** This function MUST be implemented as a Gen.jl `@gen` function so that parameter sampling is traced and can be reweighted during particle filtering.

**Example (for Fourier):**
```julia
@gen function sample_component_params(::Type{RandomFourierField})
    amplitude ~ Uniform(0, 10)
    frequency ~ Uniform(0, π)
    phase ~ Uniform(0, 2π)
    
    return Dict(
        "amplitude" => amplitude,
        "frequency" => frequency,
        "phase" => phase
    )
end
```

---

#### Function 3: `make_component(::Type{CF}, params::Dict{String, Any}) -> Function`

Construct the actual scalar field function from sampled parameters.

**Purpose:** Bridge between abstract parameters and concrete objective field function  
**Input:**
- `CF::Type{<:ComponentField}` - Component field type (for dispatch)
- `params::Dict{String, Any}` - Parameter dictionary from `sample_component_params`

**Output:** `Function` with signature `f(x::Float64, y::Float64)::Float64` that evaluates the scalar field at a given point

**Implementation:** Use parameters to create a closure capturing parameter values, return the function

**Example (for Fourier):**
```julia
function make_component(::Type{RandomFourierField}, params::Dict{String, Any})::Function
    A = params["amplitude"]
    f = params["frequency"]
    φ = params["phase"]
    
    function fourier_field(x::Float64, y::Float64)::Float64
        r = sqrt(x^2 + y^2)
        return A * sin(r * f + φ)
    end
    
    return fourier_field
end
```

---

### 1.4 Optional Interface Function

The following function can be optionally implemented for documentation and debugging:

#### Optional Function 1: `describe_component_params(::Type{CF}) -> String`

Return a human-readable description of this component field type including parameter specifications and their distributions.

**Purpose:** User-facing documentation of the component field type  
**Input:** `CF::Type{<:ComponentField}` - Component field type (for dispatch)  
**Output:** `String` containing:
- Human-readable description of the component field type
- List of all parameters defined by this type
- For each parameter: name, distribution, impact, and bounds
- The field formula

**Implementation:** Return a pre-constructed descriptive string

**Example Output (for Fourier):**
```
Component Field Type: Fourier
Description: Sinusoidal field defined by amplitude, frequency, and phase

Parameters:
  amplitude ∈ [0, 10]
    Distribution: Uniform(0, 10)
    Impact: Controls the peak magnitude of the sinusoid.
    
  frequency ∈ [0, π]
    Distribution: Uniform(0, π)
    Impact: Controls spatial wavelength of oscillation.
    
  phase ∈ [0, 2π)
    Distribution: Uniform(0, 2π)
    Impact: Controls temporal/spatial shift of the sinusoid.

Field Formula: f(x, y) = A * sin(√(x² + y²) * f + φ)
```

---

### 1.5 Extension Pattern: User-Defined Component Field

Users can extend the system by implementing their own component field type:

```julia
# User code: src/my_extensions/my_component.jl

using Gen

struct MyCustomComponent <: ComponentField end

function component_type(::Type{MyCustomComponent})::String
    return "MyCustom"
end

@gen function sample_component_params(::Type{MyCustomComponent})
    param1 ~ Uniform(0, 1)
    param2 ~ Normal(0, 1)
    
    return Dict(
        "param1" => param1,
        "param2" => param2
    )
end

function make_component(::Type{MyCustomComponent}, params::Dict{String, Any})::Function
    p1 = params["param1"]
    p2 = params["param2"]
    
    function my_field(x::Float64, y::Float64)::Float64
        # Implementation of custom field using parameters
        return p1 * x + p2 * y
    end
    
    return my_field
end

# Optional:
function describe_component_params(::Type{MyCustomComponent})::String
    return """
    Component Field Type: MyCustom
    Description: User-defined custom component field
    ...
    """
end
```

Once implemented, users pass `MyCustomComponent` to the component registry and it's available for inference.

---

### 1.4 Reference Implementation 1: Fourier Component

**File:** `src/priors/fourier.jl`

The `RandomFourierField` component type defines random sinusoidal component objectives using continuous parameters.

#### 1.4.1 Type Definition

```julia
struct RandomFourierField <: ComponentField end
```

#### 1.4.2 Interface Implementation

All three required functions must be implemented:

```julia
function component_type(::Type{RandomFourierField})::String
    return "Fourier"
end

@gen function sample_component_params(::Type{RandomFourierField})
    amplitude ~ Uniform(0, 10)
    frequency ~ Uniform(0, π)
    phase ~ Uniform(0, 2π)
    
    return Dict(
        "amplitude" => amplitude,
        "frequency" => frequency,
        "phase" => phase
    )
end

function make_component(::Type{RandomFourierField}, params::Dict{String, Any})::Function
    A = params["amplitude"]
    f = params["frequency"]
    φ = params["phase"]
    
    function fourier_field(x::Float64, y::Float64)::Float64
        r = sqrt(x^2 + y^2)
        return A * sin(r * f + φ)
    end
    
    return fourier_field
end

function describe_component_params(::Type{RandomFourierField})::String
    return """
    Component Field Type: Fourier (RandomFourierField)
    Description: Sinusoidal field with continuous amplitude, frequency, and phase parameters.
    
    Parameters:
      amplitude ∈ [0, 10]
        Distribution: Uniform(0, 10)
        Impact: Controls the peak magnitude of the sinusoid.
        
      frequency ∈ [0, π]
        Distribution: Uniform(0, π)
        Impact: Controls spatial wavelength of oscillation (normalized to Nyquist).
        
      phase ∈ [0, 2π)
        Distribution: Uniform(0, 2π)
        Impact: Controls temporal/spatial shift of the sinusoid.
    
    Field Formula: f(x, y) = A * sin(√(x² + y²) * f + φ)
    """
end
```

#### 1.4.3 Key Characteristics

- **Parameter Distributions:** Continuous multivariate (amplitude, frequency, phase)
- **Parameter Space:** 3D: `[0,10] × [0,π] × [0,2π)`
- **Field Formula:** `f(x,y) = A * sin(√(x² + y²) * f + φ)`
- **Frequency Normalization:** Bounded by Nyquist frequency π
- **Phase Periodicity:** Toric topology on `[0,2π]`

---

### 1.5 Reference Implementation 2: RBF Component

**File:** `src/priors/rbf_continuous.jl`

The `RadialBasisField` component type defines Gaussian radial basis function component objectives using continuous parameters.

#### 1.5.1 Type Definition

```julia
struct RadialBasisField <: ComponentField end
```

#### 1.5.2 Interface Implementation

All three required functions must be implemented:

```julia
function component_type(::Type{RadialBasisField})::String
    return "RBF"
end

@gen function sample_component_params(::Type{RadialBasisField})
    center_x ~ Normal(0, 5)
    center_y ~ Normal(0, 5)
    strength ~ Uniform(0, 10)
    dropoff ~ Uniform(0.1, 2.1)
    
    return Dict(
        "center_x" => center_x,
        "center_y" => center_y,
        "strength" => strength,
        "dropoff" => dropoff
    )
end

function make_component(::Type{RadialBasisField}, params::Dict{String, Any})::Function
    cx = params["center_x"]
    cy = params["center_y"]
    σ = params["strength"]
    λ = params["dropoff"]
    
    function rbf_field(x::Float64, y::Float64)::Float64
        dist_sq = (x - cx)^2 + (y - cy)^2
        return σ * exp(-λ * dist_sq)
    end
    
    return rbf_field
end

function describe_component_params(::Type{RadialBasisField})::String
    return """
    Component Field Type: RBF (RadialBasisField)
    Description: Gaussian radial basis function field with continuous parameters.
    
    Parameters:
      center_x ∈ ℝ
        Distribution: Normal(0, 5)
        Impact: X-coordinate of Gaussian peak in state space.
        
      center_y ∈ ℝ
        Distribution: Normal(0, 5)
        Impact: Y-coordinate of Gaussian peak in state space.
        
      strength ∈ [0, 10]
        Distribution: Uniform(0, 10)
        Impact: Controls amplitude of Gaussian bump.
        
      dropoff ∈ [0.1, 2.1]
        Distribution: Uniform(0.1, 2.1)
        Impact: Inverse variance controlling Gaussian width (higher = narrower).
    
    Field Formula: f(x, y) = σ * exp(-λ * ((x - cx)² + (y - cy)²))
    """
end
```

#### 1.5.3 Key Characteristics

- **Parameter Distributions:** Continuous multivariate (center_x, center_y, strength, dropoff)
- **Parameter Space:** 4D: `ℝ² × [0,10] × [0.1,2.1]`
- **Field Formula:** `f(x,y) = σ * exp(-λ * ((x - cx)² + (y - cy)²))`
- **Center Distribution:** Unbounded Normal(0, 5) for complete domain coverage
- **Dropoff Bounds:** Lower bound at 0.1 prevents infinitely broad fields

---

## Part 2: Component Choice Distribution

### 2.1 Conceptual Foundation

The **component choice distribution** specifies the likelihood of selecting each component type at each position in the K-component objective sum.

In the simplest case: equal probability for each registered component type.  
In general: user-defined function mapping component type registry to probability distribution.

This distribution is sampled K times in the generative inference model (§4), each sample determining one of the K component objectives.

### 2.2 Choice Distribution Design

**File:** `src/priors/component_choice_dist.jl`

```julia
"""
    ComponentChoiceDistribution

Specifies the probability of selecting each registered component type.
Can be:
- UniformChoiceDistribution: equal probability for all types
- WeightedChoiceDistribution: user-provided weights
- CustomChoiceDistribution: user-provided probability function
"""
abstract type AbstractComponentChoiceDistribution end

struct UniformComponentChoiceDistribution <: AbstractComponentChoiceDistribution
    component_types::Vector{Type{<:ComponentField}}
end

struct WeightedComponentChoiceDistribution <: AbstractComponentChoiceDistribution
    component_types::Vector{Type{<:ComponentField}}
    weights::Vector{Float64}  # must sum to 1
end

struct CustomComponentChoiceDistribution <: AbstractComponentChoiceDistribution
    component_types::Vector{Type{<:ComponentField}}
    prob_fn::Function  # (component_type_idx) -> Float64 (unnormalized weight)
end
```

### 2.3 Sampling from Choice Distribution

```julia
"""
    sample_component_type(dist::AbstractComponentChoiceDistribution, rng) -> Type{<:ComponentField}

Sample a component type from the choice distribution.
"""
function sample_component_type(dist::UniformComponentChoiceDistribution, rng::AbstractRNG)
    idx = rand(rng, 1:length(dist.component_types))
    return dist.component_types[idx]
end

function sample_component_type(dist::WeightedComponentChoiceDistribution, rng::AbstractRNG)
    idx = sample(rng, 1:length(dist.component_types); weights=dist.weights)
    return dist.component_types[idx]
end

function sample_component_type(dist::CustomComponentChoiceDistribution, rng::AbstractRNG)
    weights = [dist.prob_fn(i) for i in 1:length(dist.component_types)]
    weights ./= sum(weights)  # normalize
    idx = sample(rng, 1:length(dist.component_types); weights=weights)
    return dist.component_types[idx]
end
```

### 2.4 Factory Functions

```julia
"""
    make_uniform_choice_dist(component_types::Vector{Type}) -> UniformComponentChoiceDistribution

Create uniform probability distribution over all component types.
"""
function make_uniform_choice_dist(component_types::Vector{Type{<:ComponentField}})
    return UniformComponentChoiceDistribution(component_types)
end

"""
    make_weighted_choice_dist(component_types::Vector{Type}, weights::Vector{Float64}) -> WeightedComponentChoiceDistribution

Create weighted probability distribution. Weights automatically normalized.
"""
function make_weighted_choice_dist(component_types::Vector{Type{<:ComponentField}}, weights::Vector{Float64})
    @assert length(component_types) == length(weights)
    weights_normalized = weights ./ sum(weights)
    return WeightedComponentChoiceDistribution(component_types, weights_normalized)
end

"""
    make_custom_choice_dist(component_types::Vector{Type}, prob_fn::Function) -> CustomComponentChoiceDistribution

Create custom probability distribution via user function.
prob_fn should take component type index and return unnormalized weight.
"""
function make_custom_choice_dist(component_types::Vector{Type{<:ComponentField}}, prob_fn::Function)
    return CustomComponentChoiceDistribution(component_types, prob_fn)
end
```

### 2.5 Integration with Gen.jl

In the generative inference model (§4), the choice distribution is sampled K times:

```julia
# Pseudo-code in generative model
for k in 1:K
    component_type ~ sample_component_type(config.choice_dist, rng)
    parameters ~ sample_parameters(component_type, rng, parameter_spec(component_type))
    component ~ assemble_component(component_type, parameters)
    objective_sum += component
end
```

This enables composable, extensible component mixture models.

---

## Part 3: Configuration Structure

### 3.1 Conceptual Foundation

The **configuration structure** bundles all user-defined parameters, component types, and references into a single object passed to the generative inference model and particle filter.

This structure serves a similar role to the current `ScoreΠDist`, but generalized and extensible.

### 3.2 Configuration Structure Definition

**File:** `src/config/inference_config.jl`

```julia
"""
    ComponentRegistry

Registry of all available component types for this inference run.
"""
struct ComponentRegistry
    component_types::Vector{Type{<:ComponentField}}
    
    function ComponentRegistry(types::Vector{Type{<:ComponentField}})
        @assert !isempty(types) "ComponentRegistry must contain at least one component type"
        return new(types)
    end
end

"""
    InferenceConfig

Complete configuration for SMC³ inference.

Fields:
- registry::ComponentRegistry          : Available component types
- choice_dist::AbstractComponentChoiceDistribution : Distribution over component types
- K::Int                               : Number of components in objective sum
- agent_params::Dict                   : MDP agent parameters (discount, etc.)
- rl_config::RLConfig                  : RL learning hyperparameters (SoftQ-learn)
- warmstart_enabled::Bool              : Whether to use warm starts in filter
- iterative_deepening::Bool            : Whether to increase iterations over filter steps
- metadata::Dict{String, Any}          : User-defined metadata for tracking
"""
struct InferenceConfig
    registry::ComponentRegistry
    choice_dist::AbstractComponentChoiceDistribution
    K::Int                              # number of components
    agent_params::Dict{String, Any}
    rl_config::RLConfig
    warmstart_enabled::Bool
    iterative_deepening::Bool
    metadata::Dict{String, Any}
    
    function InferenceConfig(
        registry::ComponentRegistry,
        choice_dist::AbstractComponentChoiceDistribution,
        K::Int;
        agent_params::Dict=Dict(),
        rl_config::RLConfig=RLConfig(),
        warmstart_enabled::Bool=true,
        iterative_deepening::Bool=true,
        metadata::Dict=Dict()
    )
        @assert K >= 1 "K must be at least 1"
        return new(
            registry, choice_dist, K, agent_params, rl_config,
            warmstart_enabled, iterative_deepening, metadata
        )
    end
end

"""
    RLConfig

Configuration for SoftQ-learn parameter learning.

Fields:
- temperature::Float64                 : Boltzmann temperature for policy
- n_iterations::Int                    : Number of SoftQ iterations per particle
- learning_rate::Float64               : SoftQ learning rate
- value_reg::Float64                   : Value function regularization
- n_samples_per_state::Int             : Samples for value estimation
"""
struct RLConfig
    temperature::Float64
    n_iterations::Int
    learning_rate::Float64
    value_reg::Float64
    n_samples_per_state::Int
    
    function RLConfig(;
        temperature::Float64=1.0,
        n_iterations::Int=100,
        learning_rate::Float64=0.01,
        value_reg::Float64=0.001,
        n_samples_per_state::Int=10
    )
        return new(temperature, n_iterations, learning_rate, value_reg, n_samples_per_state)
    end
end
```

### 3.3 Factory Functions

```julia
"""
    make_inference_config(K::Int; kwargs...) -> InferenceConfig

Convenience constructor with sensible defaults.

Includes RandomFourierField and RadialBasisField by default, with uniform choice distribution.
"""
function make_inference_config(K::Int;
    registry::ComponentRegistry=ComponentRegistry([RandomFourierField, RadialBasisField]),
    choice_dist::AbstractComponentChoiceDistribution=
        make_uniform_choice_dist([RandomFourierField, RadialBasisField]),
    agent_params::Dict=Dict(),
    rl_config::RLConfig=RLConfig(),
    warmstart_enabled::Bool=true,
    iterative_deepening::Bool=true,
    metadata::Dict=Dict()
)
    return InferenceConfig(
        registry, choice_dist, K,
        agent_params=agent_params,
        rl_config=rl_config,
        warmstart_enabled=warmstart_enabled,
        iterative_deepening=iterative_deepening,
        metadata=metadata
    )
end
```

### 3.4 Usage Example

```julia
# User code: construct inference config for specific problem

# Define which component types to use
my_registry = ComponentRegistry([RandomFourierField, RadialBasisField, MyCustomComponent])

# Define probability of selecting each type
my_choice_dist = make_weighted_choice_dist(
    [RandomFourierField, RadialBasisField, MyCustomComponent],
    [0.5, 0.3, 0.2]  # 50% Fourier, 30% RBF, 20% custom
)

# Configure inference
config = InferenceConfig(
    registry=my_registry,
    choice_dist=my_choice_dist,
    K=5,  # 5 components in objective
    agent_params=Dict("discount"=>0.99, "horizon"=>100),
    rl_config=RLConfig(temperature=1.5, n_iterations=200),
    warmstart_enabled=true,
    iterative_deepening=true,
    metadata=Dict("experiment"=>"my_exp_v1")
)
```

---

## Part 4: Generative Inference Model

### 4.1 Conceptual Foundation

The **generative inference model** is a Gen.jl `@gen` function that:

1. Samples component types K times from `choice_dist`
2. For each component type, samples parameters from its distribution
3. Assembles each component using the component type's assembly function
4. Sums all K components to form complete objective function
5. Constructs MDP from objective and learns optimal Q-function via SoftQ-learn
6. Learns Boltzmann policy from Q-function
7. Evaluates likelihood of observations under the learned policy

This model is traced probabilistically by Gen.jl, enabling particle filtering.

### 4.2 Model Structure

**File:** `src/inference/gen_model_continuous.jl` (replaces `src/inference/gen_model.jl`)

```julia
using Gen

"""
    InferenceTrace

Structure holding a complete generative trace:
- component_types::Vector{Type}        : Sequence of K sampled component types
- component_params::Vector{Dict}       : Sequence of K parameter dictionaries
- components::Vector{Function}         : Sequence of K assembled scalar fields
- objective::Function                  : Complete summed objective
- mdp::AbstractMDP                     : MDP constructed from objective
- q_function::QLearner                 : Learned Q-function
- policy::Function                     : Boltzmann policy derived from Q
- observation_likelihood::Float64      : P(observations | policy)
"""
struct InferenceTrace
    component_types::Vector{Type{<:ComponentField}}
    component_params::Vector{Dict{String, Any}}
    components::Vector{Function}
    objective::Function
    mdp::AbstractMDP
    q_function::QLearner
    policy::Function
    observation_likelihood::Float64
end
```

### 4.3 Generative Model Implementation

```julia
@gen function inference_model(
    config::InferenceConfig,
    observations::Vector{Int},
    state_data::Matrix{Float64}
)::InferenceTrace
    
    # ========== Phase 1: Component Sampling ==========
    # Sample K components, each from component choice distribution
    
    component_types = Vector{Type{<:ComponentField}}(undef, config.K)
    component_params = Vector{Dict{String, Any}}(undef, config.K)
    components = Vector{Function}(undef, config.K)
    
    for k in 1:config.K
        # Sample component type from choice distribution
        component_type ~ choose_component(config.choice_dist)
        component_types[k] = component_type
        
        # Sample parameters from component type's distribution
        param_spec = parameter_spec(component_type)
        params = Dict{String, Any}()
        
        for (i, param_name) in enumerate(param_spec.names)
            param_dist = param_spec.distributions[i]
            param_sample ~ sample_param(param_dist)
            params[param_name] = param_sample
        end
        component_params[k] = params
        
        # Assemble component scalar field function
        components[k] = assemble_component(component_type, params)
    end
    
    # ========== Phase 2: Objective Construction ==========
    # Assemble complete objective as sum of K components
    
    function complete_objective(x::Float64, y::Float64)::Float64
        total = 0.0
        for comp in components
            total += comp(x, y)
        end
        return total
    end
    
    # ========== Phase 3: MDP Construction & Q-Learning ==========
    # Build MDP from objective, learn Q-function and policy
    
    mdp = construct_mdp_from_objective(complete_objective, state_data, config.agent_params)
    
    q_learner = QLearner(mdp, config.rl_config)
    q_function = learn_q_function(q_learner, observations, state_data)
    
    # ========== Phase 4: Policy Construction ==========
    # Derive Boltzmann policy from Q-function
    
    policy_fn = make_boltzmann_policy(q_function, config.rl_config.temperature)
    
    # ========== Phase 5: Likelihood Evaluation ==========
    # Evaluate P(observations | policy)
    
    log_likelihood = 0.0
    for (t, obs_action) in enumerate(observations)
        state_t = state_data[:, t]
        action_dist = policy_fn(state_t)  # Boltzmann distribution over actions
        log_likelihood += logpdf(action_dist, obs_action)
    end
    observation_likelihood = exp(log_likelihood)
    
    # ========== Phase 6: Return Complete Trace ==========
    
    return InferenceTrace(
        component_types,
        component_params,
        components,
        complete_objective,
        mdp,
        q_function,
        policy_fn,
        observation_likelihood
    )
end
```

### 4.4 Generative Primitives

Gen.jl integration requires defining distribution objects for each sampling step:

#### Primitive 1: Component Type Choice

```julia
struct ComponentTypeChoiceDist <: Gen.Distribution
    choice_dist::AbstractComponentChoiceDistribution
end

function Gen.logpdf(d::ComponentTypeChoiceDist, component_type::Type)
    # Return log-probability of component_type under d.choice_dist
    idx = findfirst(ct -> ct == component_type, d.choice_dist.component_types)
    # Compute probability based on distribution type
    return log(compute_prob(d.choice_dist, idx))
end

function Gen.random(d::ComponentTypeChoiceDist)
    return sample_component_type(d.choice_dist, Random.GLOBAL_RNG)
end
```

```julia
@gen function choose_component(choice_dist::AbstractComponentChoiceDistribution)
    component_type ~ choose_component_dist(ComponentTypeChoiceDist(choice_dist))
    return component_type
end
```

#### Primitive 2: Parameter Sampling

```julia
struct ParameterSamplingDist <: Gen.Distribution
    param_dist_fn::Function  # ()->sample_value
end

function Gen.logpdf(d::ParameterSamplingDist, value::Float64)
    # Depend on param_dist_fn's probability model
    # For continuous: return 0 (improper uniform over ℝ)
    # For proper distributions: return actual log-pdf
    return 0.0
end

function Gen.random(d::ParameterSamplingDist)
    return d.param_dist_fn()
end
```

```julia
@gen function sample_param(param_dist_fn::Function)
    value ~ param_sampling(ParameterSamplingDist(param_dist_fn))
    return value
end
```

### 4.5 Key Differences from Old Approach

| Aspect | Old (Discretized) | New (Continuous) |
|--------|-------------------|------------------|
| **Fourier Parameters** | Categorical over `{fx_idx, fy_idx, A_idx, ϕ_idx}` | Continuous multivariate (amplitude, frequency, phase) |
| **RBF Parameters** | Categorical over discretized grid | Continuous multivariate (center_x, center_y, strength, dropoff) |
| **Component Types** | Hardcoded (Fourier, RBF only) | Extensible via API |
| **Component Mixing** | Fixed weights in `fourier_cfg` | Flexible choice distribution |
| **Q-Learning** | Separate offline phase | Integrated into inference loop per particle |
| **Likelihood Calculation** | Component-agnostic policy evaluation | Component-type-specific likelihood methods |
| **Extensibility** | Requires code modification to add types | Users extend API without touching core code |

---

## Part 5: SMC³ Particle Filter

### 5.1 Conceptual Foundation

The **SMC³ particle filter** (Sequential Monte Carlo Cubed) is a particle filter specifically designed for open-ended SIPS:

1. Each particle represents a complete inference trace: K components + Q-function + policy
2. Particles are weighted by how well their policy explains each observation
3. Resampling removes low-probability particles
4. **Warm starts:** Particles similar to survivors reuse their Q-function as initialization
5. **Iterative deepening:** Number of SoftQ iterations increases as filter progresses

### 5.2 Particle Filter State

**File:** `src/inference/smc3_filter.jl` (replaces `src/inference/particle_filter.jl`)

```julia
"""
    SMC3ParticleState

State of SMC³ particle filter at a given time step.

Fields:
- particles::Vector{InferenceTrace}    : N particles, each a complete inference trace
- weights::Vector{Float64}             : Normalized weights for each particle
- log_evidence::Float64                : Log marginal likelihood of observations so far
- warmstart_cache::Dict                : Q-function cache for warm starts
- t::Int                               : Current time step
"""
struct SMC3ParticleState
    particles::Vector{InferenceTrace}
    weights::Vector{Float64}
    log_evidence::Float64
    warmstart_cache::Dict{UInt64, QLearner}  # hash(objective) -> Q-learner
    t::Int
end
```

### 5.3 Filter Initialization

```julia
"""
    smc3_initialize(config::InferenceConfig, observations::Vector{Int}, 
                    state_data::Matrix{Float64}, n_particles::Int)::SMC3ParticleState

Initialize SMC³ filter with n_particles.
Each particle is an independent trace of inference_model.
"""
function smc3_initialize(config::InferenceConfig, observations::Vector{Int}, 
                        state_data::Matrix{Float64}, n_particles::Int)::SMC3ParticleState
    
    particles = Vector{InferenceTrace}(undef, n_particles)
    weights = ones(Float64, n_particles) / n_particles
    
    # Generate all particles
    for i in 1:n_particles
        # Trace inference_model via Gen.jl
        trace, = Gen.generate(inference_model, 
                            (config, observations, state_data),
                            observations_choicemap)
        particles[i] = trace.retval  # Extract InferenceTrace
    end
    
    # Weight particles by observation likelihood
    for i in 1:n_particles
        weights[i] = particles[i].observation_likelihood
    end
    weights ./= sum(weights)  # normalize
    
    log_evidence = log(mean(particles[i].observation_likelihood for i in 1:n_particles))
    
    return SMC3ParticleState(
        particles, weights, log_evidence,
        Dict{UInt64, QLearner}(), 1
    )
end
```

### 5.4 Filter Update Step

```julia
"""
    smc3_update!(state::SMC3ParticleState, config::InferenceConfig, 
                observations::Vector{Int}, state_data::Matrix{Float64})

Update filter with one more observation. 
Resamples if effective sample size is low.
Performs warm starts and iterative deepening.
"""
function smc3_update!(state::SMC3ParticleState, config::InferenceConfig, 
                     observations::Vector{Int}, state_data::Matrix{Float64})
    
    n_particles = length(state.particles)
    new_weights = zeros(Float64, n_particles)
    
    # ========== Phase 1: Weight Update ==========
    # For each particle: re-learn Q-function with new observation
    
    for i in 1:n_particles
        old_trace = state.particles[i]
        
        # Optionally use warm start: initialize Q-function from cache
        if config.warmstart_enabled
            q_init = get_warmstart_qlearner(state, old_trace, config)
        else
            q_init = nothing
        end
        
        # Optionally increase iterations via iterative deepening
        if config.iterative_deepening
            n_iters = config.rl_config.n_iterations + (state.t - 1) * 50  # increasing
        else
            n_iters = config.rl_config.n_iterations
        end
        
        # Re-learn Q-function with new observation included
        q_learner = QLearner(old_trace.mdp, config.rl_config)
        if q_init !== nothing
            q_learner = warm_start_qlearner(q_learner, q_init)
        end
        
        new_q = learn_q_function(q_learner, observations[1:state.t+1], state_data[:, 1:state.t+1], 
                                n_iters=n_iters)
        
        # Update policy with new Q-function
        new_policy = make_boltzmann_policy(new_q, config.rl_config.temperature)
        
        # Evaluate new likelihood
        new_log_lik = 0.0
        for (t, obs_action) in enumerate(observations[1:state.t+1])
            state_t = state_data[:, t]
            action_dist = new_policy(state_t)
            new_log_lik += logpdf(action_dist, obs_action)
        end
        new_likelihood = exp(new_log_lik)
        
        # Update particle
        new_trace = InferenceTrace(
            old_trace.component_types,
            old_trace.component_params,
            old_trace.components,
            old_trace.objective,
            old_trace.mdp,
            new_q,
            new_policy,
            new_likelihood
        )
        state.particles[i] = new_trace
        new_weights[i] = new_likelihood
    end
    
    # ========== Phase 2: Weight Normalization ==========
    new_weights ./= sum(new_weights)
    state.weights .= new_weights
    
    # ========== Phase 3: Resample if ESS Low ==========
    ess = 1 / sum(w^2 for w in state.weights)
    ess_threshold = 0.5 * n_particles
    
    if ess < ess_threshold
        # Resample particles
        indices = sample(1:n_particles, Weights(state.weights), n_particles, replace=true)
        state.particles .= state.particles[indices]
        state.weights .= ones(n_particles) / n_particles
        
        # Cache Q-learners for warm start
        for i in 1:n_particles
            cache_key = hash(state.particles[i].objective)
            state.warmstart_cache[cache_key] = state.particles[i].q_function
        end
    end
    
    # ========== Phase 4: Update Evidence ==========
    mean_likelihood = mean(state.particles[i].observation_likelihood for i in 1:n_particles)
    state.log_evidence += log(mean_likelihood)
    
    # ========== Phase 5: Increment Time ==========
    state.t += 1
end
```

### 5.5 Warm Start Mechanism

```julia
"""
    get_warmstart_qlearner(state::SMC3ParticleState, particle::InferenceTrace, 
                          config::InferenceConfig)::Union{QLearner, Nothing}

Retrieve warm-start Q-learner from cache if available.
Similarity metric: Euclidean distance in component parameter space.
"""
function get_warmstart_qlearner(state::SMC3ParticleState, particle::InferenceTrace, 
                               config::InferenceConfig)::Union{QLearner, Nothing}
    
    if isempty(state.warmstart_cache)
        return nothing
    end
    
    # Find most similar particle in cache (by component parameters)
    particle_param_vec = param_vector(particle)
    best_key = nothing
    best_dist = Inf
    
    for cached_key in keys(state.warmstart_cache)
        # (In practice, store mapping from cache key to particle params)
        # Here: simplification - return first cached Q-learner
        best_key = cached_key
        break
    end
    
    if best_key !== nothing
        return state.warmstart_cache[best_key]
    else
        return nothing
    end
end

"""
    warm_start_qlearner(new_learner::QLearner, warm_start_learner::QLearner)::QLearner

Initialize new Q-learner with Q-values from warm_start_learner.
"""
function warm_start_qlearner(new_learner::QLearner, warm_start_learner::QLearner)::QLearner
    # Copy Q-function values from warm_start_learner to new_learner
    # (Implementation depends on QLearner internals from Crux.jl)
    return new_learner
end
```

### 5.6 Iterative Deepening Strategy

Iterative deepening increases SoftQ iterations as filter progresses, allowing:
- Early timesteps: quick approximate Q-functions
- Later timesteps: refined Q-functions for better discrimination

```julia
function compute_deepening_iterations(base_n_iters::Int, t::Int, T::Int)::Int
    # Linear schedule: start at base_n_iters, increase to 2*base_n_iters by end
    return base_n_iters + round(Int, (t - 1) / (T - 1) * base_n_iters)
end

# Usage in smc3_update!:
# n_iters_t = compute_deepening_iterations(config.rl_config.n_iterations, state.t, length(observations))
```

### 5.7 Main Filter Loop

```julia
"""
    run_smc3_filter(config::InferenceConfig, observations::Vector{Int}, 
                   state_data::Matrix{Float64}, n_particles::Int = 50)::SMC3ParticleState

Run complete SMC³ filter over all observations.
Returns final particle state.
"""
function run_smc3_filter(config::InferenceConfig, observations::Vector{Int}, 
                        state_data::Matrix{Float64}, n_particles::Int = 50)::SMC3ParticleState
    
    state = smc3_initialize(config, observations, state_data, n_particles)
    
    for t in 2:length(observations)
        smc3_update!(state, config, observations, state_data)
    end
    
    return state
end
```

### 5.8 Key Differences from Old Approach

| Aspect | Old (Basic SMC) | New (SMC³) |
|--------|-----------------|-----------|
| **Particle Contents** | Hardcoded fourier/rbf indices | Complete inference traces (components + Q-functions + policies) |
| **Component Types** | Fixed (Fourier, RBF) | Extensible via registry |
| **Parameter Space** | Discrete indices | Continuous multivariate |
| **Q-Learning** | Separate offline phase, applied post-filter | Integrated per particle at each update step |
| **Warm Starts** | Not implemented | Cache similar Q-functions and reuse |
| **Iterative Deepening** | Not implemented | Increase SoftQ iterations as filter progresses |
| **Likelihood Calculation** | Fixed policy evaluation | Uses learned Boltzmann policy from Q-function |
| **Policy Learning** | SoftQ-learn applied once | SoftQ-learn applied per particle per timestep |

---

## Part 6: Implementation Roadmap

### 6.1 Phase 0: Setup & Refactoring (Foundation)

**Goals:** Establish clean codebase structure for new architecture

#### 6.1.1 Create New Directory Structure

```
src/
├── priors/
│   ├── component_api.jl               [NEW] - ComponentField + interface
│   ├── fourier_continuous.jl          [NEW] - RandomFourierField implementation
│   ├── rbf_continuous.jl              [NEW] - RadialBasisField implementation
│   ├── component_choice_dist.jl       [NEW] - Choice distribution logic
│   └── (old fourier.jl, rbf.jl archived)
├── config/
│   └── inference_config.jl            [NEW] - InferenceConfig + RLConfig structs
├── inference/
│   ├── gen_model_continuous.jl        [NEW] - Generative inference model
│   ├── smc3_filter.jl                 [NEW] - SMC³ particle filter
│   └── (old gen_model.jl, particle_filter.jl archived)
└── types.jl                           [MODIFY] - Remove discrete configs, add new structs
```

#### 6.1.2 Archive Old Code

Move discretized implementations to `archive/` for reference:
- `archive/fourier_discrete.jl`
- `archive/rbf_discrete.jl`
- `archive/particle_filter_old.jl`
- `archive/gen_model_old.jl`

#### 6.1.3 Update Module Exports

Update `src/Arrodes.jl` to export:
- `ComponentField`, `RandomFourierField`, `RadialBasisField`
- `ComponentRegistry`, `InferenceConfig`, `RLConfig`
- `make_uniform_choice_dist`, `make_weighted_choice_dist`, `make_custom_choice_dist`
- `run_smc3_filter`, `smc3_initialize`, `smc3_update!`

**Dependencies:** None (preparation only)

**Testing:** Verify module loads without errors

---

### 6.2 Phase 1: Component Type API

**Goals:** Define extensible component type system

#### 6.2.1 Implement Abstract Type & Interface (0.5 days)

**File:** `src/priors/component_api.jl`

- Define `ComponentField` supertype
- Define interface functions (5 total)
- Create `ParameterSpec` struct
- Document extension pattern

**Testing:**
- Write tests for `ParameterSpec` construction
- Verify dispatch on abstract type works
- Create dummy component type for testing

**Validation:**
- ✅ Interface functions have proper signatures
- ✅ Documentation clear with examples
- ✅ Tests pass

---

#### 6.2.2 Implement Fourier Component (1 day)

**File:** `src/priors/fourier_continuous.jl`

- Define `RandomFourierField <: ComponentField`
- Implement all 5 interface methods
- Parameter distributions: amplitude [0,10), frequency [0,π), phase [0,2π)
- Assembly function: `f(x,y) = A * sin(√(x² + y²) * f + φ)`

**Testing:**
- Unit tests for parameter sampling
- Test component assembly generates proper scalar field
- Test assembly functions are closures properly capturing parameters
- Test likelihood evaluation (initially placeholder)

**Validation:**
- ✅ Sampling produces values in specified bounds
- ✅ Assembled fields evaluate correctly at test points
- ✅ Likelihood function computes without error

---

#### 6.2.3 Implement RBF Component (1 day)

**File:** `src/priors/rbf_continuous.jl`

- Define `RadialBasisField <: ComponentField`
- Implement all 5 interface methods
- Parameter distributions: center ∼ N(0,5), strength [0,10), dropoff [0.1,2.1)
- Assembly function: `f(x,y) = σ * exp(-λ * (x-cx)² - λ * (y-cy)²)`

**Testing:**
- Unit tests for parameter sampling
- Test center sampling produces reasonable spread
- Test component assembly generates Gaussian bumps
- Verify dropoff parameter controls Gaussian width

**Validation:**
- ✅ Sampled parameters in proper ranges
- ✅ Assembled fields behave like Gaussian RBF
- ✅ Components can be summed meaningfully

---

#### 6.2.4 Integration Testing (0.5 days)

- Test both component types can be dispatched on in loop
- Test component registry with both types
- Test extensibility: create test custom component type
- Verify no hardcoded assumptions about specific types

**Validation:**
- ✅ All 5 interface methods callable via multiple dispatch
- ✅ Custom component type works seamlessly

**Checkpoint 1 Passing Criteria:**
- ✅ Component type API implemented and documented
- ✅ Fourier and RBF components fully functional
- ✅ Interface extensible to user types
- ✅ All tests passing

---

### 6.3 Phase 2: Component Choice Distribution

**Goals:** Build flexible component type mixture system

#### 6.3.1 Implement Choice Distribution Types (0.5 days)

**File:** `src/priors/component_choice_dist.jl`

- Define `AbstractComponentChoiceDistribution` abstract type
- Implement `UniformComponentChoiceDistribution`
- Implement `WeightedComponentChoiceDistribution`
- Implement `CustomComponentChoiceDistribution`

**Testing:**
- Unit tests for uniform distribution (all types equal prob)
- Unit tests for weighted distribution (probabilities correct)
- Unit tests for custom distribution (user function called)

**Validation:**
- ✅ Probabilities sum to 1
- ✅ Sampling respects distribution

---

#### 6.3.2 Implement Sampling Functions (0.5 days)

- Implement `sample_component_type()` methods for each distribution type
- Implement factory functions: `make_uniform_choice_dist()`, etc.
- Add proper error checking and bounds validation

**Testing:**
- Generate samples and verify distribution over many trials
- Chi-squared tests for distribution matching

**Validation:**
- ✅ Empirical distributions match specified probabilities
- ✅ Factory functions work correctly

---

#### 6.3.3 Gen.jl Integration (1 day)

- Define `ComponentTypeChoiceDist <: Gen.Distribution`
- Implement `Gen.logpdf()` method
- Implement `Gen.random()` method
- Create `@gen` function wrapper `choose_component()`

**Testing:**
- Test Gen.jl tracing with component choice
- Test probability calculations
- Test reweighting traces with different choice probabilities

**Validation:**
- ✅ Gen.jl can trace component selection
- ✅ Log-probabilities correct for particle weighting

**Checkpoint 2 Passing Criteria:**
- ✅ Choice distribution system implemented
- ✅ Gen.jl integration working
- ✅ All tests passing
- ✅ Can compose complex choice distributions

---

### 6.4 Phase 3: Configuration Structure

**Goals:** Create unified configuration object

#### 6.4.1 Implement Config Structs (0.5 days)

**File:** `src/config/inference_config.jl`

- Define `ComponentRegistry` struct
- Define `RLConfig` struct
- Define `InferenceConfig` struct
- Add validation in constructors

**Testing:**
- Test valid configurations
- Test invalid configurations (reject K<1, etc.)
- Test default parameters sensible

**Validation:**
- ✅ Configurations pass validation
- ✅ Defaults reasonable
- ✅ All fields accessible

---

#### 6.4.2 Implement Factory Functions (0.5 days)

- `make_inference_config()` with sensible defaults
- Convenience constructors for common cases
- Metadata field for tracking experiments

**Testing:**
- Test factory produces valid config
- Test defaults match expectations
- Test metadata storage and retrieval

**Validation:**
- ✅ Factories work as expected
- ✅ Configs ready to pass to inference

**Checkpoint 3 Passing Criteria:**
- ✅ Configuration structures complete
- ✅ Easy to construct and use
- ✅ All fields accessible and validated
- ✅ Tests passing

---

### 6.5 Phase 4: Generative Inference Model

**Goals:** Implement Gen.jl-based probabilistic model

#### 6.5.1 Implement InferenceTrace Struct (0.5 days)

**File:** `src/inference/gen_model_continuous.jl`

- Define `InferenceTrace` struct
- Add methods to extract/inspect trace contents
- Add pretty-printing

**Testing:**
- Test trace construction
- Test field access

**Validation:**
- ✅ Traces construct properly
- ✅ Fields accessible

---

#### 6.5.2 Implement Gen.jl Primitives (1 day)

- Define `ComponentTypeChoiceDist <: Gen.Distribution`
- Define `ParameterSamplingDist <: Gen.Distribution`
- Implement Gen interface methods
- Create wrapper `@gen` functions

**Testing:**
- Test primitives can be traced by Gen.jl
- Test probability calculations
- Test determinism with fixed RNG

**Validation:**
- ✅ Primitives integrate with Gen.jl
- ✅ Traces reproducible with fixed seed

---

#### 6.5.3 Implement Inference Model `@gen` Function (2 days)

**Pseudocode:**
```
@gen function inference_model(config, observations, state_data)
    # Phase 1: Sample K components
    for k in 1:config.K
        component_type ~ choose_component(config.choice_dist)
        for each parameter in parameter_spec(component_type)
            param ~ sample_param(param_dist)
        end
        component ~ assemble_component(component_type, params)
    end
    
    # Phase 2: Construct objective
    objective = sum(components)
    
    # Phase 3: Build MDP & learn Q
    mdp = construct_mdp_from_objective(objective, state_data, config.agent_params)
    q_function = learn_q_function(mdp, observations, config.rl_config)
    
    # Phase 4: Construct policy
    policy = make_boltzmann_policy(q_function, config.rl_config.temperature)
    
    # Phase 5: Evaluate likelihood
    log_likelihood = sum(logpdf(policy(state_t), obs_action) for (t, obs_action) in observations)
    observation_likelihood = exp(log_likelihood)
    
    # Phase 6: Return trace
    return InferenceTrace(component_types, component_params, components,
                         objective, mdp, q_function, policy, observation_likelihood)
end
```

**Key Implementation Details:**
- Integrate with existing `construct_mdp_from_objective()` (from MuKumari)
- Use SoftQ-learn from Crux.jl for Q-function learning
- Make Boltzmann policy from Q-values
- Evaluate likelihood under policy

**Testing:**
- Test on simple 2-action toy MDP
- Test likelihood computation matches manual calculation
- Test different component types produce different traces
- Test Gen.jl can generate and reweight traces

**Validation:**
- ✅ Model traces successfully
- ✅ Likelihoods computed correctly
- ✅ Works with different component types
- ✅ Q-functions learned meaningfully

---

#### 6.5.4 Component-Specific Likelihood Methods (1 day)

Implement `likelihood_component_contribution()` for Fourier and RBF:

**For RandomFourierField:**
- May weight likelihood by component frequency (high-frequency components more uncertain)
- May include regularization on amplitude

**For RadialBasisField:**
- May weight by Gaussian width (narrow RBFs more certain about localized regions)
- May include regularization on strength

**Testing:**
- Unit tests for each component type
- Verify likelihoods reasonable
- Test that different components produce different likelihoods

**Validation:**
- ✅ Likelihoods computed correctly per type
- ✅ Reasonable values (in [0,1] after normalization)

**Checkpoint 4 Passing Criteria:**
- ✅ Generative model fully implemented
- ✅ Gen.jl integration working
- ✅ Q-function learning integrated
- ✅ Likelihood evaluation working
- ✅ Tests passing with realistic scenarios
- ✅ Can trace and weight particles properly

---

### 6.6 Phase 5: SMC³ Particle Filter

**Goals:** Implement Sequential Monte Carlo³ for Open-Ended SIPS

#### 6.6.1 Implement Particle State & Initialization (1 day)

**File:** `src/inference/smc3_filter.jl`

- Define `SMC3ParticleState` struct
- Implement `smc3_initialize()` function
- Add methods to query particle state (best particle, log evidence, etc.)

**Testing:**
- Test initialization creates n_particles particles
- Test weights initialized properly
- Test all particles have valid traces

**Validation:**
- ✅ Correct number of particles
- ✅ Weights sum to 1
- ✅ Log evidence sensible

---

#### 6.6.2 Implement Filter Update Step (1.5 days)

- Implement `smc3_update!()` function
- For each particle: re-learn Q with new observation
- Update weights based on new likelihood
- Perform resampling if ESS low
- Cache Q-functions for warm start

**Key Details:**
- Use observation choicemap to constrain inference
- Re-trace inference model with new observations
- Update Q-functions efficiently
- Implement effective sample size check

**Testing:**
- Test on toy 2-action problem with synthetic observations
- Test weights update correctly
- Test resampling when ESS drops
- Test state transitions are valid

**Validation:**
- ✅ Update produces valid new state
- ✅ Weights change reasonably with new observations
- ✅ ESS computation correct
- ✅ Resampling preserves high-weight particles

---

#### 6.6.3 Implement Warm Starts (1 day)

- Implement `get_warmstart_qlearner()` function
- Cache Q-learners from resampled particles
- Implement similarity metric for particle matching (Euclidean in param space)
- Implement `warm_start_qlearner()` to initialize from cache

**Key Details:**
- Hash objective functions for cache key
- Store mapping from cache key to component parameters
- Similarity threshold for matching
- Graceful fallback if no good match

**Testing:**
- Test similar particles get matched
- Test Q-function initialization from warm start
- Test convergence faster with warm start than without
- A/B test: warm start vs. cold start on same observation sequence

**Validation:**
- ✅ Warm start initialization works
- ✅ Similarity matching reasonable
- ✅ Performance improvement measurable

---

#### 6.6.4 Implement Iterative Deepening (0.5 days)

- Implement `compute_deepening_iterations()` function
- Integrate into `smc3_update!()` to increase iterations over time
- Linear schedule: start at `base_n_iters`, increase to `2 * base_n_iters`

**Testing:**
- Test iteration count increases properly
- Test learning improves over filter steps
- Compare final Q-functions at early vs. late steps

**Validation:**
- ✅ Iterations increase smoothly
- ✅ Late particles have better Q-functions than early ones

---

#### 6.6.5 Implement Main Filter Loop (0.5 days)

- Implement `run_smc3_filter()` function
- Loop over all observations, calling update at each step
- Return final particle state

**Testing:**
- End-to-end test on realistic observation sequence
- Test with small (2-3 obs) and larger (20+ obs) sequences
- Test different particle counts (10, 50, 100)

**Validation:**
- ✅ Filter runs to completion
- ✅ Final particles have learned meaningful Q-functions
- ✅ Most likely particle sensible given observations

---

#### 6.6.6 Integrate with Existing Code (1 day)

- Connect to `construct_mdp_from_objective()` (MuKumari)
- Connect to SoftQ-learn (Crux.jl)
- Connect to Boltzmann policy construction
- Update type exports in `src/Arrodes.jl`

**Testing:**
- Integration tests with real MuKumari objectives
- Test with Crux.jl SoftQ solver
- End-to-end tests matching current ablation pipeline structure

**Validation:**
- ✅ All integrations work
- ✅ No breaking changes to dependencies

**Checkpoint 5 Passing Criteria:**
- ✅ SMC³ particle filter fully implemented
- ✅ Warm starts working
- ✅ Iterative deepening working
- ✅ End-to-end tests passing
- ✅ Performance improvements from warm starts visible
- ✅ Ready for integration with ablation studies

---

### 6.7 Phase 6: Migration & Validation

**Goals:** Transition old pipeline to new architecture

#### 6.7.1 Update Ablation Infrastructure (1 day)

**Files to modify:**
- `src/analysis/ablations.jl` - `ablation_main()`, etc.
- `examples/ablations/iq_sips_ablation.jl` - entry point

**Changes:**
- Replace old `run_inference()` call with `run_smc3_filter()`
- Update key construction: old discrete keys → continuous component parameters
- Update result caching to store complete `InferenceTrace` objects
- Adapt visualization to work with continuous parameters

**Testing:**
- Test ablation pipeline runs to completion
- Test results caching works
- Test visualization generates without errors

**Validation:**
- ✅ Ablation pipeline functional
- ✅ Results saved and loaded correctly
- ✅ Visualizations display

---

#### 6.7.2 Update Visualization (1.5 days)

**Files to modify:**
- `src/viz/objectives.jl` - visualize objectives
- `src/rl/scoredist.jl` - (adapt from discrete keys to continuous)

**Changes:**
- Update objective visualization to handle continuous parameters
- Remove assumptions about discrete key structure
- Add plotting for component parameters
- Create visualizations showing K selected components

**Testing:**
- Test visualizations generate for various configurations
- Test with different K values
- Test with mixed component types

**Validation:**
- ✅ Visualizations informative
- ✅ Handle edge cases (single component, all same type, etc.)

---

#### 6.7.3 Comprehensive Testing (1.5 days)

- Create test suite comparing old vs. new on same observation sequences
- Verify new system produces reasonable results
- Performance benchmarks (time per update, memory usage)
- Sensitivity analysis (effect of K, config parameters, etc.)

**Testing:**
- Regression tests against old implementation on small problems
- Synthetic observation generation and filter performance
- Parameter sweep: K, n_particles, RL config variations
- Stress test: large observation sequences

**Validation:**
- ✅ Results sensible and reproducible
- ✅ No catastrophic failures
- ✅ Performance acceptable

---

#### 6.7.4 Documentation & Examples (1 day)

- Write comprehensive API documentation
- Create example notebooks:
  - Basic usage with default Fourier + RBF
  - Custom component type definition
  - Configuration tuning
  - Ablation study setup
- Update README with new architecture overview

**Testing:**
- Run all example notebooks
- Verify documentation builds correctly

**Validation:**
- ✅ Examples run successfully
- ✅ Documentation clear and complete

**Checkpoint 6 Passing Criteria:**
- ✅ Old pipeline migrated to new architecture
- ✅ Ablation infrastructure working
- ✅ Visualization updated
- ✅ Comprehensive tests passing
- ✅ Documentation complete
- ✅ Examples working
- ✅ Ready for production use

---

## Part 7: Integration & Compatibility

### 7.1 Dependency Integration

**Existing Dependencies (Maintained):**
- `Gen.jl` - Generative modeling (enhanced usage for component sampling)
- `GenParticleFilters.jl` - (May use for reference, but building custom SMC³)
- `MuKumari.jl` - MDP construction, state management
- `Crux.jl` - SoftQ-learn for Q-function learning
- `POMDPs.jl` - MDP interface

**New Patterns:**
- Heavy use of multiple-dispatch for component types
- Gen.jl custom distribution definitions
- Integration of RL learning into inference loop

### 7.2 Backward Compatibility

**What Changes:**
- Key format: discrete indices → continuous vectors
- Configuration API: `ScoreΠDist` → `InferenceConfig`
- Filter interface: `particle_filter()` → `run_smc3_filter()`
- Result format: updated to include `InferenceTrace` objects

**What Stays the Same:**
- MDP interface (POMDPs.jl)
- State representation
- Observation format
- Visualization style (adapts to continuous)
- Ablation study framework

**Migration Path:**
- Archive old implementation in `archive/` for reference
- Provide migration guide for external code
- Gradual deprecation warnings (not applicable for internal codebase)

### 7.3 Performance Expectations

| Metric | Old (Discretized) | New (Continuous) | Notes |
|--------|-------------------|------------------|-------|
| **Particles/Step** | 50 default | 50 default | Same order of magnitude |
| **Time/Particle/Step** | ~10ms | ~15-20ms | Increased Q-learning overhead |
| **Memory/Particle** | ~1MB (discretized) | ~5-10MB (full trace) | Stores complete Q-functions |
| **Convergence** | Variable | Better (continuous + warm starts) | More sample-efficient |

**Performance Optimization Opportunities:**
- Parallel Q-learning across particles
- Caching MDP construction
- Approximate Q-functions in early filter steps
- Adaptive iteration scheduling

---

## Part 8: Success Criteria & Validation

### 8.1 Functional Requirements

- ✅ Component type API implemented and extensible
- ✅ Fourier and RBF components working
- ✅ Component choice distribution flexible
- ✅ Configuration structure encapsulates all parameters
- ✅ Generative model traces and weights properly
- ✅ SMC³ filter runs and converges
- ✅ Warm starts reduce Q-function relearning
- ✅ Iterative deepening improves late-step discrimination
- ✅ Integration with existing ablation pipeline
- ✅ Visualization works with continuous parameters
- ✅ Full API documentation
- ✅ Example notebooks demonstrating usage

### 8.2 Code Quality Requirements

- ✅ All functions have docstrings
- ✅ Type annotations on function signatures
- ✅ Unit tests for all public functions
- ✅ Integration tests for pipeline
- ✅ Code follows Julia style guide
- ✅ No type piracy or namespace pollution
- ✅ Proper error handling and validation

### 8.3 Performance Requirements

- ✅ Filter runs in reasonable time (< 1 min per observation for 50 particles)
- ✅ Memory usage stays within bounds (< 500MB for 50 particles)
- ✅ Convergence at least as fast as old system
- ✅ Warm starts provide measurable speedup (20%+ improvement)

### 8.4 Extensibility Requirements

- ✅ User can define custom component type with minimal code
- ✅ Custom type integrates seamlessly with existing types
- ✅ No need to modify core Arrodes.jl code
- ✅ Clear extension pattern documented with examples

### 8.5 Validation Tests

**Synthetic Data Tests:**
- Generate synthetic observations from known Q-function
- Run filter and verify recovers true objective

**Comparison Tests:**
- Run new system on same problems as old system
- Compare filter convergence
- Compare final Q-functions

**Stress Tests:**
- Large K (20+ components)
- Many observations (100+ steps)
- Large particle counts (200+ particles)
- Various component type mixtures

**Correctness Tests:**
- Likelihood calculations (analytical vs. numerical)
- Weight updates (verify normalization)
- Policy sampling (verify Boltzmann distribution)

---

## Part 9: Timeline & Resource Estimates

### 9.1 Phase Breakdown

| Phase | Task | Duration | Status |
|-------|------|----------|--------|
| 0 | Setup & Refactoring | 0.5 days | Planning |
| 1 | Component Type API | 2.5 days | Planning |
| 2 | Component Choice Distribution | 2 days | Planning |
| 3 | Configuration Structure | 1 day | Planning |
| 4 | Generative Inference Model | 4.5 days | Planning |
| 5 | SMC³ Particle Filter | 4.5 days | Planning |
| 6 | Migration & Validation | 5 days | Planning |
| **Total** | | **20 days** | **Planning** |

### 9.2 Implementation Order

1. **Start:** Phases 0 → 1 (sequential, foundational)
2. **Parallelize:** Phases 2 & 3 (can work in parallel)
3. **Sequential:** Phase 4 (depends on earlier phases)
4. **Sequential:** Phase 5 (depends on phase 4)
5. **Final:** Phase 6 (depends on phases 4 & 5)

### 9.3 Critical Path

Phases 0 → 1 → 4 → 5 → 6 are on critical path (~15 days)

Phases 2-3 can absorb schedule slack.

### 9.4 Testing Schedule

- Unit tests: After each function implementation
- Integration tests: After each phase
- End-to-end tests: After phases 4, 5, 6
- Performance benchmarks: Phase 6

---

## Part 10: File Modification Reference

### 10.1 Files to Create

**New Source Files:**
```
src/priors/component_api.jl                    (~200 lines)
src/priors/fourier_continuous.jl               (~150 lines)
src/priors/rbf_continuous.jl                   (~150 lines)
src/priors/component_choice_dist.jl            (~120 lines)
src/config/inference_config.jl                 (~100 lines)
src/inference/gen_model_continuous.jl          (~300 lines)
src/inference/smc3_filter.jl                   (~400 lines)
```

**Archive (Old Code):**
```
archive/fourier_discrete.jl                    (copy from src/priors/fourier.jl)
archive/rbf_discrete.jl                        (copy from src/priors/rbf.jl)
archive/particle_filter_old.jl                 (copy from src/inference/particle_filter.jl)
archive/gen_model_old.jl                       (copy from src/inference/gen_model.jl)
```

**Test Files:**
```
test/priors/component_api_tests.jl             (~200 lines)
test/priors/fourier_tests.jl                   (~200 lines)
test/priors/rbf_tests.jl                       (~200 lines)
test/priors/choice_dist_tests.jl               (~150 lines)
test/config/inference_config_tests.jl          (~100 lines)
test/inference/gen_model_tests.jl              (~250 lines)
test/inference/smc3_tests.jl                   (~350 lines)
```

**Documentation:**
```
docs/COMPONENT_API.md                          (~200 lines)
docs/MIGRATION_GUIDE.md                        (~150 lines)
examples/custom_component_tutorial.md          (~150 lines)
examples/basic_usage.ipynb                     (Jupyter notebook)
```

### 10.2 Files to Modify

**Core Files:**
- `src/types.jl` - Remove `FourierDiscreteCfg`, `RBFDiscreteCfg`, add new structs
- `src/Arrodes.jl` - Update module exports
- `src/analysis/ablations.jl` - Update to use `run_smc3_filter()`
- `src/viz/objectives.jl` - Adapt visualization for continuous parameters
- `src/rl/scoredist.jl` - Adapt or replace key structure
- `examples/ablations/iq_sips_ablation.jl` - Update entry point

**Test Files:**
- `test/runtests.jl` - Update to include new test modules

### 10.3 Code Statistics

**New Code:** ~1,500 lines  
**Modified Code:** ~200 lines (existing files)  
**Archived Code:** ~1,000 lines (preserved for reference)  
**Test Code:** ~1,600 lines  
**Documentation:** ~700 lines  
**Total New Deliverables:** ~5,600 lines

---

## Part 11: References to Existing Work

### 11.1 Existing Code to Leverage

**MuKumari Integration:**
- `construct_mdp_from_objective()` - Used as-is in generative model
- `state_data` representation - Maintained unchanged
- Observation interface - Maintained unchanged

**Crux.jl Integration:**
- `QLearner` struct - Used for Q-function learning
- `learn_q_function()` - Used within inference model
- SoftQ learning algorithm - Core to particle evaluation

**Gen.jl Integration:**
- Particle filtering framework - Reference only (custom SMC³)
- Trace generation interface - Used for component sampling
- Probability computation - Used for weighting

**POMDPs.jl Integration:**
- `AbstractMDP` - Used for MDP interface
- Standard MDP operations - Unchanged

### 11.2 What Gets Replaced

**Old Discrete Sampling:**
- `fourier.jl`: Discrete categorical sampling → Continuous multivariate
- `rbf.jl`: Discrete grid sampling → Continuous multivariate
- Key format: `(fx_idx, fy_idx, A_idx, ϕ_idx)` → Continuous parameter dict

**Old Filter:**
- `particle_filter.jl` (current Basic SMC) → `smc3_filter.jl` (SMC³)
- Rejuvenation heuristics → Warm starts + iterative deepening
- Fixed policy evaluation → Integrated Q-learning per particle

**Old Configuration:**
- `ScoreΠDist` (hardcoded) → `InferenceConfig` (extensible)
- Discretized keys → Continuous component parameters
- Fixed component types → User-extensible via API

### 11.3 What Stays the Same

**Data Structures:**
- `state_data::Matrix{Float64}` - Unchanged
- `observations::Vector{Int}` - Unchanged
- MDP state/action spaces - Unchanged

**Interfaces:**
- `construct_mdp_from_objective()` - Unchanged (used as-is)
- `learn_q_function()` from Crux.jl - Unchanged (used as-is)
- POMDPs.jl standard interface - Unchanged

**Integration Points:**
- MuKumari.jl objectives - Still constructed from parameters
- Ablation study framework - Adapted but preserved
- Result visualization pipeline - Adapted but preserved

---

## Part 12: Known Challenges & Mitigation

### 12.1 Challenge: Q-Function Caching with Continuous Parameters

**Problem:** Old system cached Q-functions with discrete keys. Continuous parameters don't have natural binning for cache hits.

**Mitigation:**
- Store complete Q-function objects in warm start cache (not just values)
- Use objective hash as cache key (hash of assembled objective function)
- Similarity matching via Euclidean distance in parameter space
- Conservative warm start: only reuse Q-functions for very similar objectives

**Alternative:** If performance issues arise, implement parameter space discretization (e.g., grid hashing) while keeping user-facing API continuous.

### 12.2 Challenge: Gen.jl Custom Distributions

**Problem:** Gen.jl custom distributions require careful implementation of `logpdf()` and `random()`.

**Mitigation:**
- Implement simple distributions first (uniform over parameters)
- Test thoroughly with Gen.jl's inference utilities
- May need to use improper priors (log-pdf = 0) for unbounded parameters
- Reference Gen.jl custom distribution examples

**Fallback:** If integration too complex, implement direct Gen.jl @gen functions without custom distributions.

### 12.3 Challenge: Performance with Larger K

**Problem:** Filter must learn Q-function for objectives that are sums of K components. Larger K may make Q-learning harder/slower.

**Mitigation:**
- Iterative deepening scales iterations with K automatically
- Warm starts reduce redundant computation
- Profile early to identify bottlenecks
- Consider approximate Q-functions for large K

**Monitoring:** Add performance benchmarks comparing K=1, K=5, K=10, K=20.

### 12.4 Challenge: Likelihood Evaluation with Learned Policies

**Problem:** Boltzmann policy from learned Q-function may be multimodal or have numerical issues.

**Mitigation:**
- Use temperature parameter to control policy smoothness
- Validate policy sampling produces reasonable distributions
- Add numerical safeguards (exp-normalize trick for stability)
- Test likelihood calculation against analytical solutions on toy problems

**Validation:** Unit tests with known policies.

### 12.5 Challenge: Generalization to User-Defined Components

**Problem:** API must be general enough to support unforeseen user components.

**Mitigation:**
- Keep component type interface minimal (5 methods only)
- Use multiple dispatch everywhere (no type-specific conditionals)
- Test with synthetic user component types early
- Clear documentation and examples of extension pattern
- May need slight API revision after first user extensions

---

## Part 13: Experimental & Future Directions

### 13.1 Potential Enhancements (Post-MVP)

1. **Hierarchical Components:** Components containing sub-components (recursive structure)
2. **Component Scaling:** Learn global scale factor for all K components together
3. **Sparse Components:** Support for sparse vector representations of components
4. **Adaptive Component Types:** Automatically select between types based on observation patterns
5. **Bayesian Model Selection:** Infer K (number of components) from data
6. **Transfer Learning:** Warm starts across different observation sequences

### 13.2 Analysis & Benchmarking

Post-implementation tasks:
- Compare filter convergence with old system
- Analyze effect of K on filter performance
- Study warm start effectiveness across parameter ranges
- Benchmark parallelization potential
- Sensitivity analysis on RL hyperparameters

### 13.3 Integration with Ablation Framework

Extend ablation studies to new dimensions:
- Ablation over component types (Fourier vs. RBF vs. custom)
- Ablation over K (1, 3, 5, 10 components)
- Ablation over RL config (temperature, iterations, etc.)
- Ablation over warm start/iterative deepening

---

## Appendix A: Quick Reference for Extension

### A.1 Minimal Custom Component Example

```julia
# User defines new component type

struct MyComponent <: ComponentField end

function component_name(::Type{MyComponent})
    return "MyComponent"
end

function parameter_spec(::Type{MyComponent})::ParameterSpec
    return ParameterSpec(
        names = ["param1", "param2"],
        dims = [1, 1],
        distributions = [()->rand(), ()->randn()],
        bounds = [(0, 1), (-Inf, Inf)]
    )
end

function sample_parameters(::Type{MyComponent}, rng, spec)
    return Dict("param1" => rand(rng), "param2" => randn(rng))
end

function assemble_component(::Type{MyComponent}, params)
    p1 = params["param1"]
    p2 = params["param2"]
    return (x, y) -> p1 * x + p2 * y  # linear component
end

function likelihood_component_contribution(::Type{MyComponent}, params, obs, policy_fn)
    return 0.0  # placeholder
end

# User passes to config

registry = ComponentRegistry([RandomFourierField, RadialBasisField, MyComponent])
choice_dist = make_uniform_choice_dist([RandomFourierField, RadialBasisField, MyComponent])
config = InferenceConfig(registry, choice_dist, K=5)
state = run_smc3_filter(config, observations, state_data, n_particles=50)
```

### A.2 Configuration Examples

**Example 1: Default (Fourier + RBF, equal weights, K=5)**
```julia
config = make_inference_config(K=5)
```

**Example 2: Fourier-only with 10 components**
```julia
registry = ComponentRegistry([RandomFourierField])
choice_dist = make_uniform_choice_dist([RandomFourierField])
config = InferenceConfig(registry, choice_dist, K=10)
```

**Example 3: Custom weights with warm starts**
```julia
registry = ComponentRegistry([RandomFourierField, RadialBasisField, MyComponent])
choice_dist = make_weighted_choice_dist(
    [RandomFourierField, RadialBasisField, MyComponent],
    [0.6, 0.3, 0.1]
)
config = InferenceConfig(
    registry, choice_dist, K=5,
    warmstart_enabled=true,
    iterative_deepening=true
)
```

---

## Appendix B: Mathematical Notation

### B.1 Model Specification

**Generative Model:**
```
For k = 1 to K:
  c_k ~ ComponentChoiceDistribution        (component type)
  θ_k ~ ParameterDistribution(c_k)        (component parameters)
  φ_k(x,y) ~ AssembleComponent(c_k, θ_k) (scalar field function)

Objective: O(x,y) = Σ_{k=1}^K φ_k(x,y)

MDP: M = (S, A, R, P) where R(s) = O(s)

Q* = SoftQ(M, τ)                          (SoftQ-learn)

π(a|s) ∝ exp(Q*(s,a) / τ)                (Boltzmann policy)

L = Π_t P(a_t | π, s_t)                  (Likelihood of observations)
  = Π_t π(a_t | s_t)
```

### B.2 Particle Filter

**Sequential Update:**
```
At time t:
  For i = 1 to N:
    Re-learn: Q_i^{(t)} = SoftQ(M_i, obs_{1:t}, τ, n_iter_t)
    Update: π_i^{(t)} = Boltzmann(Q_i^{(t)})
    Weight: w_i^{(t)} ∝ π_i^{(t)}(a_t | s_t)
  
  Normalize: w_i^{(t)} ← w_i^{(t)} / Σ_j w_j^{(t)}
  
  If ESS < ρ*N:
    Resample indices ~ Categorical(w)
    Cache Q-functions from survivors
  
  Update evidence: Z ← Z * mean_i(w_i^{(t)})
```

---

## Appendix C: Glossary

- **Component Type:** Abstract Julia type representing a class of objective functions (e.g., Fourier, RBF)
- **Component Parameters:** Sampled values specific to one component instance (e.g., amplitude, frequency)
- **Component Choice Distribution:** Probability distribution over which component types to select
- **InferenceConfig:** User-facing configuration structure bundling all parameters
- **ComponentRegistry:** Registry of available component types
- **InferenceTrace:** Complete probabilistic trace from generative model (components + Q-function + policy)
- **SMC³:** Sequential Monte Carlo with three levels of randomness (component choice, parameters, policy learning)
- **Warm Start:** Reusing Q-function from similar particle to initialize new Q-learning
- **Iterative Deepening:** Increasing SoftQ iterations as filter progresses
- **Boltzmann Policy:** Softmax over Q-function values with temperature parameter

---

## Summary: Where We're Going

This redesign transforms Arrodes from a discretized, hacked-together system into a principled, extensible, mathematically-grounded inference engine:

✅ **Continuous Parameters:** No more discretization griddles  
✅ **Extensible API:** Users define custom component types  
✅ **Integrated Learning:** Q-functions learned per-particle during inference  
✅ **SMC³ + Warm Starts:** More sample-efficient filtering  
✅ **Iterative Deepening:** Better late-stage discrimination  
✅ **Clean Architecture:** Separation of concerns throughout  

The 20-day implementation roadmap provides concrete steps to achieve this vision, with rigorous testing and validation at each phase.

**Next Step:** Begin Phase 0 (setup & refactoring) and Phase 1 (component type API).
