

################################################################################
# NEW COMPONENTFIELD-BASED IMPLEMENTATION
################################################################################

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
    amp_max::Float64 = 5.0
    σ::Float64 = 1.0
end

"""
    component_type(::RadialBasisField)

Return the identifier for this component type.
"""
function component_type(::RadialBasisField)
    return "RBF"
end

"""
    sample_rbf_params(field::RadialBasisField)

Sample RBF parameters using specific bounds from field instance via Gen.jl.

Uses Gen's lowercase distribution functions for continuous parameter sampling,
replacing the old discrete bin-based approach while maintaining the ability to
sample center coordinates and amplitude.
"""
Gen.@gen function sample_rbf_params(field::RadialBasisField)
    center_x ~ Gen.uniform(field.x_min, field.x_max)
    center_y ~ Gen.uniform(field.y_min, field.y_max)
    amplitude ~ Gen.uniform(field.amp_min, field.amp_max)
    
    return Dict(
        "center_x" => center_x,
        "center_y" => center_y,
        "amplitude" => amplitude,
        "sigma" => field.σ
    )
end

"""
    make_component(::RadialBasisField, params::Dict{String, Any})

Construct an RBF scalar field from sampled parameters.

Implements the field formula: f(x, y) = A * exp(-(dx² + dy²) / (2σ²))
where dx = x - center_x, dy = y - center_y

This matches the old discrete implementation's field formula, but now with
continuous parameters sampled via Gen.jl instead of discretized bins.

# Arguments
- `params::Dict`: Must contain keys "center_x", "center_y", "amplitude", "sigma"

# Returns
- `Function`: Scalar field f(x::Real, y::Real)::Float64
"""
function make_component(::RadialBasisField, params::Dict{String, Any})
    cx = params["center_x"]
    cy = params["center_y"]
    A = params["amplitude"]
    σ = params["sigma"]
    
    σ_sq = σ^2
    
    # Field formula: f(x,y) = A * exp(-(dx² + dy²) / (2σ²))
    function rbf_field(x::Real, y::Real)::Float64
        dx = x - cx
        dy = y - cy
        r_sq = dx^2 + dy^2
        return A * exp(-r_sq / (2 * σ_sq))
    end
    
    return rbf_field
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
