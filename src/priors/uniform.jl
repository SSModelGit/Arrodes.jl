"""
    component_type(::UniformField)

Return the identifier for this component type.
"""
function component_type(::UniformField)
    return "Uniform"
end

"""
    uniform_params_sampler(field::UniformField)

Return a Gen-wrapped closure that samples Uniform parameters from the given field.
"""
function uniform_params_sampler(field::UniformField)
    Gen.@gen function sample_params()
        constant ~ Gen.uniform(field.constant_min, field.constant_max)
        return Dict("constant" => constant)
    end
    return sample_params
end

"""
    make_component(::UniformField, params::Dict{String, Any})

Construct a Uniform scalar field from sampled parameters.

Implements the field formula: f(x, y) = constant
where the constant is sampled from the uniform distribution.

# Arguments
- `params::Dict`: The sampled constant we will use for the uniform field.

# Returns
- `Function`: Scalar field f(x::Real, y::Real)::Float64
"""
function make_component(::UniformField, params::Dict)
    constant = params["constant"]
    # Sample constant value from uniform distribution defined by field bounds
    function uniform_field(x::Real, y::Real)::Float64
        return constant
    end

    return uniform_field
end

"""
    describe_component_params(::RadialBasisField)

Return documentation for RBF component parameters.
"""
function describe_component_params(::RadialBasisField)
    return """
    Component Field Type: RBF (RadialBasisField)
    Description: Continuous Gaussian RBF field (replacing old discretized approach)

    Parameters:
      center_x ∈ [x_min, x_max]
        Distribution: uniform(x_min, x_max) [continuous, sampled via Gen.jl]
        Impact: x-coordinate of the Gaussian center
        
      center_y ∈ [y_min, y_max]
        Distribution: uniform(y_min, y_max) [continuous, sampled via Gen.jl]
        Impact: y-coordinate of the Gaussian center
        
      amplitude ∈ [amp_min, amp_max]
        Distribution: uniform(amp_min, amp_max) [continuous, sampled via Gen.jl]
        Impact: Peak height of the Gaussian
        
      sigma (σ)
        Value: Fixed from field instance [not sampled]
        Impact: Gaussian bandwidth - controls spread/width

    Field Formula: f(x, y) = A * exp(-(dx² + dy²) / (2σ²))
                  where dx = x - center_x, dy = y - center_y

    Note: This is the continuous version replacing the old discrete binned approach.
    """
end
