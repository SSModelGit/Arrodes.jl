"""
    generics.jl

Abstract type and generic function interface for component field definitions.
Each component type must implement the required interface functions.
"""

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
    component_type(::Type{CF})::String where {CF <: ComponentField}

Return the name/identifier of the component field type.

# Arguments
- `CF::Type{<:ComponentField}`: The component field type

# Returns
- `String`: Human-readable name of the component (e.g., "Fourier", "RBF")

# Example
```julia
component_type(RandomFourierField)  # returns "Fourier"
```
"""
function component_type(::Type{<:ComponentField})::String
    error("component_type not implemented for this component field type")
end

"""
    sample_component_params(::Type{CF})

Generate component parameters using generative tracing via Gen.jl.

This function MUST be implemented as a Gen `@gen` function in concrete component types.
It should use Gen's tracing mechanisms (e.g., `@trace` expressions) to sample parameters
according to the component's prior distribution.

# Arguments
- `CF::Type{<:ComponentField}`: The component field type

# Returns
- A dictionary of sampled parameters (dict structure depends on component type)

# Example
```julia
@gen function sample_component_params(::Type{RandomFourierField})
    amplitude = @trace(Uniform(0, 10), :amplitude)
    frequency = @trace(Uniform(0, π), :frequency)
    phase = @trace(Uniform(0, 2π), :phase)
    return Dict(
        "amplitude" => amplitude,
        "frequency" => frequency,
        "phase" => phase
    )
end
```

# Note
Users implementing a new component type MUST provide a `@gen` version of this function
to enable proper parameter sampling with generative tracing.
"""
function sample_component_params(::Type{<:ComponentField})
    error("sample_component_params not implemented for this component field type")
end

"""
    make_component(::Type{CF}, params::Dict)::Function where {CF <: ComponentField}

Construct the actual component field function from sampled parameters.

# Arguments
- `CF::Type{<:ComponentField}`: The component field type
- `params::Dict`: Dictionary of parameters (from `sample_component_params`)

# Returns
- `Function`: The component field function `f(x, y)::Float64`

# Example
```julia
function make_component(::Type{RandomFourierField}, params::Dict)::Function
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
"""
function make_component(::Type{<:ComponentField}, params::Dict)::Function
    error("make_component not implemented for this component field type")
end

"""
    describe_component_params(::Type{CF})::String where {CF <: ComponentField}

Return a human-readable description of the component's parameter structure.

This is an optional function. A default implementation is provided that returns
a generic message.

# Arguments
- `CF::Type{<:ComponentField}`: The component field type

# Returns
- `String`: Description of parameter names and distributions

# Example
```julia
function describe_component_params(::Type{RandomFourierField})::String
    return \"\"\"
    RandomFourierField parameters:
    - amplitude (float): Sinusoidal amplitude, Uniform[0, 10)
    - frequency (float): Spatial frequency, Uniform[0, π)
    - phase (float): Phase offset, Uniform[0, 2π)
    \"\"\"
end
```
"""
function describe_component_params(::Type{<:ComponentField})::String
    return "No parameter description available for this component type."
end
