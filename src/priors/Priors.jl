module Priors

using Parameters: @with_kw
using LinearAlgebra, Statistics, Random
using Distributions: Categorical
using Gen

using MuKumari

import ..Arrodes:
    FourierDiscreteCfg,
    ScoreΠDist,
    PriorDiscreteCfg,
    RBFDiscreteCfg,
    ComponentField,
    RandomFourierField,
    RadialBasisField,
    ZeroField
import ..Utils: randcat

include("generics.jl")
export ComponentField, component_type, make_component, describe_component_params

include("fields.jl")
export make_pomdp_objective_from_field, objective_grid_from_field, objective_grid_from_mdp

include("fourier.jl")
export fourier_params_sampler

include("rbf.jl")
export rbf_params_sampler

include("zero.jl")
export zero_params_sampler

"""
    build_component_param_switch(component_tuples::Vector)

Args:
    - component_tuples: Vector of tuples (ComponentField, sampling_closure) for each component
Returns:
    - (param_switch, component_fields):
        - param_switch: Gen.Switch object that maps component indices to parameter sampling closures
        - component_fields: Vector of ComponentField instances corresponding to the order of the switch
"""
function build_component_param_switch(component_tuples::Vector)
    # Extract component fields and sampling functions
    component_fields = [t[1] for t in component_tuples]
    param_sampling_fns = [t[2] for t in component_tuples]

    # Build Switch from the sampling functions
    param_switch = Gen.Switch(param_sampling_fns...)

    return (param_switch, component_fields)
end

"""
    component_type_sampler(component_fields::Vector)

Default generative function for sampling a component type.
Samples index uniformly from the provided vector of component fields.
Obeys the same ordering as the component switch.
This ensures correct mapping between sampled index and parameter sampling function.

Args:
  - component_fields: Vector of ComponentField instances (should match order of component switch)
Returns:
  - sample_component_type: Gen.@gen function that samples a component index based on the number of fields
    - Note that sample_component_type doesn't take any arguments!
"""
function component_type_sampler(component_fields::Vector)
    Gen.@gen function sample_component_type(c_fields::Vector = component_fields)
        component_idx ~ Gen.categorical(normalize(ones(length(c_fields)), 1))
        return component_idx  # Return index, not the field itself
    end
    return sample_component_type
end

"""
    sample_component_and_params(component_tuples::Vector{Tuple{<:ComponentField, Any}})

Generative function that selects a component type and samples its parameters.

Args:
  - component_switch: Gen.Switch object that maps component indices to parameter sampling functions

Returns:
  - (component_idx, params_dict)
  
Where component_idx can be used to look up the component field from the original vector.
"""
Gen.@gen function sample_component_and_params(
    component_switch::Gen.Switch,
    component_type_sampler::Any,
)
    # Phase 1: Select which component type
    component_idx ~ component_type_sampler()

    # Phase 2: Use Switch to sample parameters for selected component
    params ~ component_switch(component_idx)

    # Phase 3: Return index and parameters
    return (component_idx, params)
end

export build_component_param_switch, component_type_sampler, sample_component_and_params

end
