"""
    component_type(::ZeroField)

Return the identifier for this component type.
"""
function component_type(::ZeroField)
    return "Zero"
end

"""
    zero_params_sampler(field::ZeroField)

Return a Gen-wrapped closure that samples a "Zero" from the field.
"""
function zero_params_sampler(field::ZeroField)
    Gen.@gen function sample_params()
        return Dict()
    end
    return sample_params
end

"""
    make_component(::ZeroField, params::Dict{String, Any})

Construct a zero scalar field from sampled parameters.

Implements the field formula: f(x, y) = 0

This matches the old discrete implementation's field formula, now with
continuous parameters sampled via Gen.jl instead of discretized bins.

# Arguments
- `params::Dict`: Contents are ignored.

# Returns
- `Function`: Scalar field f(x::Real, y::Real)::Float64
"""
function make_component(::ZeroField, params::Dict)

    # Field formula: f(x,y) = 0
    function zero_field(x::Real, y::Real)::Float64
        return 0.0
    end

    return zero_field
end

"""
    describe_component_params(::ZeroField)

Return documentation for Zero component parameters.
"""
function describe_component_params(::ZeroField)
    return """
    Component Field Type: Zero (ZeroField)
    Description: Constant zero field (replacing old discretized approach)

    Parameters:
      None

    Field Formula: f(x, y) = 0

    Note: This is a simple zero field with no parameters.
    """
end