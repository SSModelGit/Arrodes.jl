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

The **component choice distribution** specifies the likelihood of selecting each component field at each position in the K-component objective sum.

To avoid Gen.jl's multi-dispatch limitations with `@gen` functions, component choice is implemented via the **Switch combinator**, which routes to type-specific parameter sampling functions based on the selected component index.

### 2.1 Architecture

**User-provided tuples:** Users provide `(ComponentField_instance, sampling_function)` pairs.

**Example:**
```julia
component_tuples = [
    (RandomFourierField(), sample_fourier_params),
    (RadialBasisField(), sample_rbf_params)
]
```

### 2.2 Core Functions

**File:** `src/priors/Priors.jl`

#### `build_component_param_switch(component_tuples::Vector{Tuple})`

Constructs a `Gen.Switch` combinator from component tuples.

```julia
function build_component_param_switch(component_tuples::Vector{Tuple})
    component_fields = [t[1] for t in component_tuples]
    param_sampling_fns = [t[2] for t in component_tuples]
    param_switch = Gen.Switch(param_sampling_fns...)
    return (param_switch, component_fields)
end
```

Returns: `(param_switch, component_fields)` where `param_switch` routes indices to parameter sampling functions.

#### `component_type_sampler(component_fields::Vector)`

Creates a uniform categorical sampler over component indices.

```julia
function component_type_sampler(component_fields::Vector)
    Gen.@gen function sample_component_type(c_fields::Vector=component_fields)
        component_idx ~ Gen.categorical(normalize(ones(length(c_fields)), 1))
        return component_idx  # Return index for Switch routing
    end
    return sample_component_type
end
```

#### `sample_component_and_params(component_switch, component_type_sampler)`

Combined `@gen` function that selects component type and samples parameters.

```julia
Gen.@gen function sample_component_and_params(component_switch::Gen.Switch,
                                              component_type_sampler::Function)
    component_idx ~ component_type_sampler()
    params ~ component_switch(component_idx)
    return (component_idx, params)
end
```

Returns: `(component_idx, params_dict)` for lookup and use.

### 2.3 Usage Pattern

```julia
# Build infrastructure
(param_switch, component_fields) = build_component_param_switch(component_tuples)
type_sampler = component_type_sampler(component_fields)

# Sample in inference model
component_idx, params ~ sample_component_and_params(param_switch, type_sampler)
component_field = component_fields[component_idx]
component_fn = make_component(typeof(component_field), params)
```

### 2.4 Key Design Decisions

- **Switch combinator:** Routes based on index, avoiding multi-dispatch issues with `@gen`
- **Distinct function names:** `sample_fourier_params`, `sample_rbf_params` prevent naming conflicts
- **User-provided tuples:** Explicit pairing of instances with their sampling functions
- **Full tracing:** Both component selection and parameter sampling are traced for particle filtering

## Part 3: Configuration Structure

### 3.1 Conceptual Foundation

The **configuration structure** bundles all user-defined parameters, component sampling infrastructure, and RL hyperparameters into a single object passed to the generative inference model and particle filter.

This structure encapsulates:
- The tuple-based component infrastructure from Part 2 (component tuples + sampling functions)
- RL learning hyperparameters (SoftQ-learn configuration)
- MDP parameters
- Number of components to generate
- Optional iterative deepening strategy

This serves a similar role to the current `ScoreΠDist`, but generalized and extensible.

### 3.2 RL Configuration Structure

**File:** `src/config/inference_config.jl`

The `RLConfig` struct encapsulates SoftQ-learn hyperparameters. It is defined with `@with_kw` to provide default initial arguments aligned with current SoftQ learning usage.

```julia
using Parameters: @with_kw

"""
    RLConfig

Configuration for SoftQ-learn parameter learning.

Fields:
- temperature::Float64                 : Boltzmann temperature for policy (default: 1.0)
- n_iterations::Int                    : Number of SoftQ iterations per particle (default: 100)
- learning_rate::Float64               : SoftQ learning rate (default: 0.01)
- value_reg::Float64                   : Value function regularization (default: 0.001)
- n_samples_per_state::Int             : Samples for value estimation (default: 10)
"""
@with_kw struct RLConfig
    temperature::Float64 = 1.0
    n_iterations::Int = 100
    learning_rate::Float64 = 0.01
    value_reg::Float64 = 0.001
    n_samples_per_state::Int = 10
end
```

### 3.3 Inference Configuration Structure

The `InferenceConfig` struct is defined with `@with_kw` and encapsulates all configuration needed for inference. It requires:

- **component_tuples**: Vector of `(ComponentField_instance, sampling_function)` tuples (no default)
- **component_type_sampler**: Function that provides `@gen` function for uniform categorical selection (defaults to `component_type_sampler(component_fields)` from Priors module)
- **rl_config**: RLConfig with SoftQ hyperparameters (defaults to `RLConfig()`)
- **k_components**: Number of components to generate via `sample_component_and_params` (no default)
- **agent_params**: Dict of MDP parameters (default: `Dict()`)
- **iterative_deepening**: Whether to increase SoftQ iterations over filter steps (default: `false`)
- **metadata**: User-defined metadata for experiment tracking (default: `Dict()`)

```julia
"""
    InferenceConfig

Complete configuration for SMC³ inference.

Fields:
- component_tuples::Vector{Tuple}           : (ComponentField, sampling_function) tuples
- component_type_sampler::Function          : @gen function for component type selection
- rl_config::RLConfig                       : SoftQ-learn hyperparameters
- k_components::Integer                     : Number of components to generate (NOT number of types)
- agent_params::Dict{String, Any}           : MDP agent parameters (discount, horizon, etc.)
- iterative_deepening::Bool                 : Whether to increase SoftQ iterations over filter steps
- metadata::Dict{String, Any}               : User-defined metadata for tracking

Note on warm starts: Warm starting is not implemented in this version because defining
'closeness' of objective functions in continuous parameter space is non-trivial and does
not provide sufficient efficiency gains to justify the added complexity.
"""
@with_kw struct InferenceConfig
    component_tuples::Vector{Tuple}
    component_type_sampler::Function
    rl_config::RLConfig = RLConfig()
    k_components::Integer
    agent_params::Dict{String, Any} = Dict()
    iterative_deepening::Bool = false
    metadata::Dict{String, Any} = Dict()
    
    # Validation
    function InferenceConfig(component_tuples, component_type_sampler, rl_config, k_components, agent_params, iterative_deepening, metadata)
        @assert !isempty(component_tuples) "component_tuples must not be empty"
        @assert k_components >= 1 "k_components must be at least 1"
        return new(component_tuples, component_type_sampler, rl_config, k_components, agent_params, iterative_deepening, metadata)
    end
end
```

## Part 4: Generative Inference Model

### 4.1 Conceptual Foundation

The **generative inference model** is a Gen.jl `@gen` function that:

1. Samples K components via the Part 2 infrastructure (`sample_component_and_params`)
2. For each sampled component, assembles it using the component type's assembly function
3. Sums all K components to form the complete objective function
4. Constructs MDP from objective and learns Q-function via SoftQ-learn
5. Learns Boltzmann policy from Q-function
6. Evaluates likelihood of observations under the learned policy
7. Returns only the likelihood (Float64) as the particle weight

**Key Design Principle:** The `@gen` function is a pure probabilistic model that:
- Traces all stochasticity (component selection + parameter sampling) via Gen.jl
- Performs deterministic computation (objective assembly, Q-learning, policy construction)
- Returns a single scalar likelihood weight for particle filtering

All sampled choices are automatically recorded in the Gen trace; the particle filter extracts them via trace choice addresses.

### 4.2 Model Structure

**File:** `src/inference/gen_model_continuous.jl` (replaces `src/inference/gen_model.jl`)

The `@gen` function has the following structure:

```julia
@gen function inference_model(
    config::InferenceConfig,
    observations::Vector{Int},
    state_data::Matrix{Float64}
)::Float64
    # Phase 1: Sample K components (traced via Gen.jl)
    # Phase 2: Assemble objective (deterministic)
    # Phase 3: Build MDP & learn Q-function (deterministic)
    # Phase 4: Construct policy (deterministic)
    # Phase 5: Evaluate likelihood (deterministic)
    # Phase 6: Return likelihood as Float64 weight
end
```

**Trace Structure:** All sampled components and parameters are stored in the Gen trace via:
- `trace[:sample_component_and_params => k => 1]` - component index for k-th component
- `trace[:sample_component_and_params => k => 2]` - parameter dict for k-th component

The particle filter accesses these via trace choice addresses to reconstruct component information.

### 4.3 Generative Model Implementation

```julia
using Gen

@gen function inference_model(
    config::InferenceConfig,
    observations::Vector{Int},
    state_data::Matrix{Float64}
)::Float64
    
    # ========== Phase 1: Component Sampling (Gen-traced) ==========
    # Build component infrastructure
    (param_switch, component_fields) = build_component_param_switch(config.component_tuples)
    sampler = (config.component_type_sampler !== nothing ? 
               config.component_type_sampler : 
               component_type_sampler(component_fields))
    
    # Sample K components, each traced by Gen.jl
    # All choices stored in trace via Gen's address system
    for k in 1:config.k_components
        (idx, params) ~ sample_component_and_params(param_switch, sampler)
        # Trace automatically stores this under address:
        #   :sample_component_and_params => k => (1|2)
    end
    
    # ========== Phase 2: Objective Construction (Deterministic) ==========
    # Extract component data from locally computed values
    # (In the filter, we'll reconstruct from trace as needed)
    
    components = Vector{Function}(undef, config.k_components)
    for k in 1:config.k_components
        # Note: In actual implementation, extract from trace or recompute
        # Here shown conceptually - the filter reconstructs this from trace data
        # idx = trace[:sample_component_and_params => k => 1]
        # params = trace[:sample_component_and_params => k => 2]
        # components[k] = make_component(typeof(component_fields[idx]), params)
    end
    
    # Assemble complete objective as sum of K components
    objective(x::Float64, y::Float64)::Float64 = sum(c(x, y) for c in components)
    
    # ========== Phase 3: MDP Construction & Q-Learning (Deterministic) ==========
    # Build MDP from objective
    mdp = construct_mdp_from_objective(objective, state_data, config.agent_params)
    
    # Learn Q-function using config's fixed iteration count
    # (Iterative deepening happens in Part 5 filter, not here)
    q_function = learn_q_function(mdp, observations, config.rl_config)
    
    # ========== Phase 4: Policy Construction (Deterministic) ==========
    # Derive Boltzmann policy from Q-function
    policy = make_boltzmann_policy(q_function, config.rl_config.temperature)
    
    # ========== Phase 5: Likelihood Evaluation (Deterministic) ==========
    # Evaluate P(observations | policy)
    log_likelihood = 0.0
    for t in 1:length(observations)
        state_t = state_data[:, t]
        obs_action = observations[t]
        action_dist = policy(state_t)
        log_likelihood += logpdf(action_dist, obs_action)
    end
    
    observation_likelihood = exp(log_likelihood)
    
    # ========== Phase 6: Return Likelihood as Weight ==========
    return observation_likelihood  # Float64 - this becomes trace.retval
end
```

**Key Implementation Notes:**

1. **All sampling is traced:** The loop over `sample_component_and_params` automatically creates Gen trace entries. The filter can access these via `trace[:sample_component_and_params => k => 1]` (index) and `trace[:sample_component_and_params => k => 2]` (params).

2. **No custom Gen.Distribution types needed:** `sample_component_and_params()` already handles tracing via Gen.Switch, so we don't need custom distribution wrappers.

3. **Component reconstruction in filter:** The particle filter will reconstruct component information from the trace using:
   ```julia
   component_idxs = [trace[:sample_component_and_params => k => 1] 
                     for k in 1:config.k_components]
   component_params = [trace[:sample_component_and_params => k => 2] 
                      for k in 1:config.k_components]
   components = [make_component(typeof(component_fields[idx]), params)
                for (idx, params) in zip(component_idxs, component_params)]
   ```

4. **Deterministic computation:** All Q-learning, objective assembly, and policy construction are deterministic and happen within the `@gen` function using fixed `config.rl_config.n_iterations`.

5. **Simple return value:** The function returns only the likelihood as a Float64. This becomes `trace.retval` and is used by the particle filter as the particle weight.

### 4.4 Accessing Particle Information from Traces

The particle filter works with Gen traces directly, extracting data via choice addresses:

```julia
# In the particle filter (Part 5):
function extract_particle_info(trace, config, component_fields)
    # Extract component selections and parameters
    component_idxs = Vector{Int}(undef, config.k_components)
    component_params = Vector{Dict}(undef, config.k_components)
    
    for k in 1:config.k_components
        component_idxs[k] = trace[:sample_component_and_params => k => 1]
        component_params[k] = trace[:sample_component_and_params => k => 2]
    end
    
    # Reconstruct components
    components = [make_component(typeof(component_fields[idx]), params)
                  for (idx, params) in zip(component_idxs, component_params)]
    
    # Reconstruct objective
    objective(x, y) = sum(c(x, y) for c in components)
    
    return (component_idxs, component_params, components, objective)
end

# Particle weight is simply:
weight = trace.retval  # The likelihood Float64
```

### 4.5 Why This Design

**Advantages over previous InferenceTrace approach:**

1. ✅ **Simpler**: No wrapper struct needed - Gen.Trace is the container
2. ✅ **Cleaner code**: All stochasticity naturally traced by Gen.jl
3. ✅ **Better Gen.jl integration**: Works with Gen's standard reweighting and resampling
4. ✅ **Fewer dependencies**: No custom Gen.Distribution types to maintain
5. ✅ **More modular**: Filter can reconstruct data as needed rather than storing it
6. ✅ **Easier to extend**: New component types automatically work without changes to @gen function

**Trade-offs:**

- Filter must reconstruct component information from traces (cheap operation, trace access is fast)
- Separates model definition from trace extraction (cleaner separation of concerns)

### 4.6 Key Differences from Old Approach

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

The **SMC³ particle filter** uses `GenParticleFilters.jl` to efficiently manage particle inference over a sequence of observations.

**Key Architecture:**

1. **Observations as cumulative sequences:** Unlike some particle filters that add one observation at a time, the inference model takes cumulative observation slices `observations[1:t]`. This is efficient because:
   - The inference model can see the full context (all actions taken so far)
   - Q-learning operates on the complete history
   - Policies learned reflect all historical decisions
   - No need to modify model parameters as we go

2. **Each particle represents:**
   - A complete Gen trace from `inference_model_continuous()` containing:
     * K sampled components (via traced `:components => k` choices)
     * Learned Q-function for the assembled objective on observations[1:t]
     * Boltzmann policy derived from Q-function
     * Return value: component indices for downstream analysis

3. **GenParticleFilters.jl provides:**
   - `pf_initialize`: Initialize with first observation slice
   - `pf_update!`: Update particles with new cumulative observation slices
   - `pf_resample!`: Resample low-weight particles when ESS drops
   - Automatic weight management in log-space

**Design Pattern:** 
- Reuse the existing `particle_filter` function by adding an overload that accepts `InferenceConfig`
- Maintains backward compatibility with the old `ScoreΠDist` API
- Both implementations follow the same initialization + sequential update pattern

### 5.2 Filter Structure

### 5.2 Filter Structure

**File:** `src/inference/particle_filter.jl`

The implementation extends the existing `particle_filter` function with an overload for `InferenceConfig`. This maintains backward compatibility while adding support for the continuous component API:

```julia
"""
    particle_filter(observations::Vector{Int}, config::InferenceConfig, 
                    state_data::Matrix{Float64}, n_particles::Int = 50;
                    ess_thresh::Float64 = 0.5, resample_alg::Symbol = :residual)

Run SMC³ particle filter with the continuous component API using `inference_model_continuous`.

This overload adapts the existing particle_filter to work with InferenceConfig instead of the old ScoreΠDist.
The key difference is that observations are passed cumulatively: at each filter step, we update particles
with observations[1:t] rather than just the single timestep.
"""
function particle_filter(observations::Vector{Int}, config::InferenceConfig, 
                        state_data::Matrix{Float64}, n_particles::Int = 50;
                        ess_thresh::Float64 = 0.5, resample_alg::Symbol = :residual)
    
    N = length(observations)
    obs_choices = [choicemap((:actions => n => :aidx, observations[n])) for n in 1:N]
    
    # ========== Phase 1: Initialize with first observation ==========
    state = pf_initialize(inference_model_continuous, 
                         (config, observations[1:1], state_data[:, 1:1]), 
                         obs_choices[1], n_particles)
    
    # ========== Phase 2: Sequential updates ==========
    for n in 2:N
        # Resample if ESS is low
        if effective_sample_size(state) < ess_thresh * n_particles
            pf_resample!(state, resample_alg)
        end
        
    # ========== Phase 2: Sequential updates ==========
    for n in 2:N
        # Resample if ESS is low
        if effective_sample_size(state) < ess_thresh * n_particles
            pf_resample!(state, resample_alg)
            
            # Rejuvenation: use MH to refine component selections and parameters
            # Select addresses for component sampling to allow variation
            sels = Any[]
            for k in 1:config.k_components
                push!(sels, (:components => k) => :component_idx)
                push!(sels, (:components => k) => :params)
            end
            
            # Also allow recent action choices to be refined
            a_lo = max(1, n - 3)  # refine last 3 actions
            for τ in a_lo:(n-1)
                push!(sels, (:actions => τ) => :aidx)
            end
            
            pf_rejuvenate!(state, mh, (select(sels...),))
        end
        
        # Update with cumulative observations up to timestep n
        pf_update!(state,
                   (config, observations[1:n], state_data[:, 1:n]),
                   (NoChange(), UnknownChange(), UnknownChange()),
                   obs_choices[n])
    end
    
    return state
end
```

**Rejuvenation Details:**

After resampling, the filter performs **Metropolis-Hastings (MH) rejuvenation** to increase particle diversity:

1. **Component Rejuvenation:** 
   - Proposes new component selections (indices) and parameters for each of the K components
   - Allows particles to explore different component configurations given the observations so far
   - Helps escape local optima where all particles converged to the same components

2. **Recent Action Refinement:**
   - Refines the last 3 actions taken (backward from current timestep)
   - Allows particles to reconsider recent decisions in light of current information
   - Particularly useful when early action choices interact with later observations

3. **MH Acceptance:** 
   - Proposals are accepted/rejected based on Metropolis-Hastings acceptance probability
   - Maintains proper particle weighting despite changes to trace
   - Automatically handles trace probability changes from reweighting

### 5.3 Particle Information Extraction

```julia
"""
    extract_particle_component_info(trace::Dict, config::InferenceConfig, 
                                    component_fields::Vector)

Extract component selections and parameters from a particle's trace.

Returns (component_idxs, component_params):
- component_idxs::Vector{Int} - Selected component type for each of K components
- component_params::Vector{Dict} - Parameter dictionaries for each component
"""
function extract_particle_component_info(trace::Dict, config::InferenceConfig,
                                        component_fields::Vector)
    component_idxs = Vector{Int}(undef, config.k_components)
    component_params = Vector{Dict}(undef, config.k_components)
    
    for k in 1:config.k_components
        # Access trace addresses set by inference_model_continuous
        component_idxs[k] = trace[:components => k => 1]
        component_params[k] = trace[:components => k => 2]
    end
    
    return (component_idxs, component_params)
end

"""
    best_particle(pf_state, config::InferenceConfig, component_fields::Vector)

Return the highest-weight particle and its component information.

Returns (best_idx, best_weight, component_idxs, component_params, objective_fn)
"""
function best_particle(pf_state, config::InferenceConfig, component_fields::Vector)
    # Get particle traces and weights from GenParticleFilters state
    traces = [pf_state.traces[i] for i in 1:length(pf_state.traces)]
    weights = pf_state.log_weights
    
    # Find best particle (highest log-weight)
    best_idx = argmax(weights)
    best_trace = traces[best_idx]
    best_weight = exp(weights[best_idx])
    
    (idxs, params) = extract_particle_component_info(best_trace, config, component_fields)
    
    # Reconstruct objective from best particle's components
    components = [Priors.make_component(typeof(component_fields[idx]), p)
                 for (idx, p) in zip(idxs, params)]
    objective_fn(x, y) = sum(c(x, y) for c in components)
    
    return (best_idx, best_weight, idxs, params, objective_fn)
end
```

### 5.4 Implementation Tasks

**Task 5.1: Create smc3_filter.jl module**
- Import GenParticleFilters
- Implement run_smc3_filter function using pf_initialize, pf_update!, pf_resample!
- Implement particle extraction utilities: extract_particle_component_info, best_particle
- Export public functions

**Task 5.2: Implement run_smc3_filter**
- Pre-build component infrastructure (component_switch, component_type_sampler) outside filter
- Create model wrapper function with iterative deepening support
- Call pf_initialize with first observation
- Loop over remaining observations calling pf_update!
- Perform resampling when ESS drops below threshold (0.5 * n_particles)

**Task 5.3: Implement particle extraction utilities**
- extract_particle_component_info: Extract component selections/parameters from trace
- best_particle: Get highest-weight particle with reconstructed objective and component info
- Provide access to GenParticleFilters state (traces, log_weights)

**Task 5.4: Integration testing**

### 5.5 Key Advantages of Using GenParticleFilters.jl

| Feature | Benefit |
|---------|---------|
| **pf_initialize** | Clean initialization; handles weight computation and normalization |
| **pf_update!** | Efficient trace updates with support for observation constraints |
| **pf_resample!** | Multiple resampling methods (multinomial, residual, stratified) |
| **effective_sample_size** | Built-in ESS computation; no manual weight calculations needed |
| **pf_rejuvenate!** | Optional MH rejuvenation for particle diversity (future feature) |
| **Standard interface** | Works with standard Gen.jl infrastructure; no custom reimplementation |

### 5.6 Integration with Parts 1-4

| Component | Integration Point |
|-----------|-------------------|
| **Part 1: Component API** | extract_particle_component_info uses component_fields and make_component |
| **Part 2: Choice Distribution** | component_switch and component_type_sampler pre-built outside filter |
| **Part 3: Config** | run_smc3_filter receives InferenceConfig; modified in model wrapper for deepening |
| **Part 4: Gen Model** | inference_model_continuous wrapped in closure; called per timestep with updated observations |

### 5.7 Key Differences from Old Approach

| Aspect | Old (Custom SMC) | New (GenParticleFilters.jl) |
|--------|-----------------|---------------------------|
| **Weight Management** | Manual log-space normalization | pf_initialize, pf_update! handle automatically |
| **Resampling Logic** | Custom ESS + multinomial resampling | Built-in effective_sample_size + multiple methods |
| **Trace Updates** | Manual re-simulation | pf_update! with UnknownChange for full model changes |
| **State Container** | Custom SMC3ParticleState struct | GenParticleFilters.jl ParticleFilterState |
| **Code Complexity** | ~200 lines of custom infrastructure | ~30 lines using library abstractions |
| **Flexibility** | Limited to pre-defined operations | Extensible: rejuvenation, custom weights, filtering |
| **Maintenance** | Requires bug fixes, edge case handling | Maintained by Gen.jl community |

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

#### 6.4.1 Implement RLConfig Struct (0.25 days)

**File:** `src/config/inference_config.jl`

- Define `RLConfig` struct with `@with_kw` macro
- Fields: temperature, n_iterations, learning_rate, value_reg, n_samples_per_state
- All fields have sensible defaults aligned with current SoftQ-learn usage

**Testing:**
- Test default RLConfig is valid
- Test custom RLConfig with modified parameters
- Test field access

**Validation:**
- ✅ Defaults align with existing usage
- ✅ All fields accessible

---

#### 6.4.2 Implement InferenceConfig Struct (0.5 days)

**File:** `src/config/inference_config.jl`

- Define `InferenceConfig` struct with `@with_kw` macro
- Required fields: `component_tuples`, `k_components`
- Optional fields with defaults: `component_type_sampler`, `rl_config`, `agent_params`, `iterative_deepening`, `metadata`
- Auto-generation of `component_type_sampler` from `component_tuples` if not provided
- Validation: k_components >= 1, component_tuples non-empty

**Testing:**
- Test valid InferenceConfig construction
- Test auto-generation of component_type_sampler
- Test custom component_type_sampler override
- Test invalid configurations rejected (empty tuples, k_components < 1)
- Test metadata storage and retrieval

**Validation:**
- ✅ Configurations pass validation
- ✅ Defaults reasonable
- ✅ Auto-generation works
- ✅ All fields accessible

**Checkpoint 3 Passing Criteria:**
- ✅ Configuration structures complete (RLConfig + InferenceConfig)
- ✅ @with_kw macros used for clean default specification
- ✅ Auto-generation of component_type_sampler working
- ✅ Easy to construct and use
- ✅ Tests passing

---

### 6.5 Phase 4: Generative Inference Model

**Goals:** Implement Gen.jl-based probabilistic model

#### 6.5.1 Implement Inference Model `@gen` Function (1.5 days)

**File:** `src/inference/gen_model_continuous.jl`

Implement the lean `inference_model()` function that:
- Samples K components via `sample_component_and_params()` (traced by Gen.jl)
- Assembles objective deterministically
- Builds MDP and learns Q-function with fixed iterations
- Constructs Boltzmann policy
- Evaluates likelihood
- Returns Float64 likelihood weight

**Key Points:**
- All stochasticity (component selection + parameter sampling) is traced by Gen.jl automatically
- No InferenceTrace struct - return only the likelihood
- No custom Gen.Distribution types needed
- Component information stored in trace via Gen's choice address system
- Filter reconstructs component data from trace as needed

**Testing:**
- Test on simple 2-action toy MDP
- Test likelihood computation matches manual calculation
- Test different component types produce different likelihoods
- Test Gen.jl can generate and reweight traces

**Validation:**
- ✅ Model traces successfully
- ✅ Likelihoods computed correctly
- ✅ Works with different component types
- ✅ Q-functions learned meaningfully

---

#### 6.5.2 Implement Trace Extraction Utilities (1 day)

**File:** `src/inference/gen_model_continuous.jl`

Implement helper functions for particle filter to extract information from traces:

```julia
function extract_component_info(trace, config, component_fields)
    component_idxs = [trace[:sample_component_and_params => k => 1] 
                      for k in 1:config.k_components]
    component_params = [trace[:sample_component_and_params => k => 2] 
                       for k in 1:config.k_components]
    return (component_idxs, component_params)
end

function reconstruct_objective_from_trace(trace, config, component_fields)
    (idxs, params) = extract_component_info(trace, config, component_fields)
    components = [make_component(typeof(component_fields[idx]), p)
                 for (idx, p) in zip(idxs, params)]
    return (x, y) -> sum(c(x, y) for c in components)
end
```

**Testing:**
- Test extraction from generated traces
- Test reconstructed objectives match original

**Validation:**
- ✅ Can extract all component information from traces
- ✅ Reconstructed objectives accurate

---

#### 6.5.3 Integration Testing (0.75 days)

- Test full @gen function with Part 2 infrastructure
- Test trace generation and reweighting with Gen.jl
- Test with different component type combinations
- Test likelihood computation consistency

**Validation:**
- ✅ Model integrates with Part 2 components
- ✅ Gen.jl operations work correctly
- ✅ Traces contain all necessary information

**Checkpoint 4 Passing Criteria:**
- ✅ Lean @gen function fully implemented
- ✅ Returns only likelihood as Float64
- ✅ Gen.jl traces all stochasticity
- ✅ No InferenceTrace struct needed
- ✅ No custom Gen.Distribution types
- ✅ Trace extraction utilities working
- ✅ Tests passing with realistic scenarios

---

### 6.6 Phase 5: SMC³ Particle Filter

**Goals:** Implement Sequential Monte Carlo³ for Open-Ended SIPS

#### 6.6.1 Implement Particle State & Initialization (1 day)

**File:** `src/inference/smc3_filter.jl`

- Define `SMC3ParticleState` struct (without warmstart_cache)
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
- Integrate iterative deepening (increase iterations over time)

**Key Details:**
- Use observation choicemap to constrain inference
- Re-trace inference model with new observations
- Update Q-functions efficiently
- Implement effective sample size check
- If iterative_deepening enabled, increase n_iters based on timestep

**Testing:**
- Test on toy 2-action problem with synthetic observations
- Test weights update correctly
- Test resampling when ESS drops
- Test state transitions are valid
- Test iteration count increases when iterative_deepening=true

**Validation:**
- ✅ Update produces valid new state
- ✅ Weights change reasonably with new observations
- ✅ ESS computation correct
- ✅ Resampling preserves high-weight particles
- ✅ Iterative deepening schedule increases smoothly

---

#### 6.6.3 Implement Iterative Deepening (0.5 days)

- Implement `compute_deepening_iterations()` function
- Integrate into `smc3_update!()` to increase iterations over time
- Linear schedule: start at `base_n_iters`, increase over filter steps

**Testing:**
- Test iteration count increases properly
- Test learning improves over filter steps
- Compare final Q-functions at early vs. late steps

**Validation:**
- ✅ Iterations increase smoothly
- ✅ Late particles have better Q-functions than early ones

---

#### 6.6.4 Implement Main Filter Loop (0.5 days)

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

#### 6.6.5 Integrate with Existing Code (1 day)

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
- ✅ Iterative deepening working
- ✅ End-to-end tests passing
- ✅ Iteration schedule correctly increasing over filter steps
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
| 3 | Configuration Structure | 0.75 days | Planning |
| 4 | Generative Inference Model | 3.25 days | Planning |
| 5 | SMC³ Particle Filter | 3.5 days | Planning |
| 6 | Migration & Validation | 5 days | Planning |
| **Total** | | **17.5 days** | **Planning** |

### 9.2 Implementation Order

1. **Start:** Phases 0 → 1 (sequential, foundational)
2. **Parallelize:** Phases 2 & 3 (can work in parallel)
3. **Sequential:** Phase 4 (depends on earlier phases)
4. **Sequential:** Phase 5 (depends on phase 4)
5. **Final:** Phase 6 (depends on phases 4 & 5)

### 9.3 Critical Path

Phases 0 → 1 → 4 → 5 → 6 are on critical path (~13.75 days)

Phases 2-3 can absorb schedule slack.

**Key Reductions:**
- Removed InferenceTrace struct: -0.5 days (no separate type definition needed)
- Eliminated custom Gen.Distribution types: -1 day (use existing Part 2 infrastructure)
- Eliminated component-specific likelihood methods: -1 day (not needed for simple weighting)
- **Phase 4 reduction: 4.5 → 3.25 days (28% improvement)**
- Overall: 18.75 → 17.5 days

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
src/config/inference_config.jl                 (~80 lines)
src/inference/gen_model_continuous.jl          (~300 lines)
src/inference/smc3_filter.jl                   (~350 lines)
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

**New Code:** ~1,200 lines (eliminated InferenceTrace, custom distributions, component-specific methods)
**Modified Code:** ~200 lines (existing files)  
**Archived Code:** ~1,000 lines (preserved for reference)  
**Test Code:** ~1,400 lines (streamlined Phase 4 tests, removed distribution tests)
**Documentation:** ~700 lines  
**Total New Deliverables:** ~5,100 lines

**Key Changes from Original Estimate:**
- Removed InferenceTrace struct: -80 lines
- Eliminated custom Gen.Distribution types: -150 lines
- Eliminated component-specific likelihood methods: -80 lines
- Removed related tests: -150 lines
- Simplified trace extraction utilities: -40 lines net
- Overall reduction: ~500 lines of code and complexity

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
