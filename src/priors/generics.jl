"""
    ComponentField

Abstract supertype for component objective field definitions.

Each concrete component field type (e.g., `RandomFourierField`, `RBFComponent`) 
implements the required interface functions:
- `component_type(::Type{CF}) -> String`
- `sample_component_params(::Type{CF})` (must be @gen function)
- `make_component(::Type{CF}, params::Dict) -> Function`
- `describe_component_params(::Type{CF}) -> String` (optional)
"""
abstract type ComponentField end

"""
    component_type(::ComponentField)

Return the name/identifier of the component field type.

# Arguments
- `CF::Type{<:ComponentField}`: The component field type

# Returns
- `String`: Human-readable name of the component (e.g., "Fourier", "RBF")
"""
function component_type(::ComponentField)
    error("component_type not implemented for this component field type")
end

"""
    make_component(::ComponentField, params::Dict)

Construct the actual component field function from sampled parameters.

# Arguments
- `CF::Type{<:ComponentField}`: The component field type
- `params::Dict`: Dictionary of parameters (from `sample_component_params`)

# Returns
- `Function`: The component field function `f(x, y)::Float64`
"""
function make_component(::ComponentField, params::Dict)
    error("make_component not implemented for this component field type")
end

"""
    describe_component_params(::ComponentField)

Return a human-readable description of the component's parameter structure.

This is an optional function. A default implementation is provided that returns
a generic message.

# Arguments
- `CF::Type{<:ComponentField}`: The component field type

# Returns
- `String`: Description of parameter names and distributions
"""
function describe_component_params(::ComponentField)
    return "No parameter description available for this component type."
end
