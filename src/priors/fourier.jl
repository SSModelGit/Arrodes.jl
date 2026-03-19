

################################################################################
# NEW COMPONENTFIELD-BASED IMPLEMENTATION
################################################################################

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
    component_type(::Type{RandomFourierField})::String

Return the identifier for this component type.
"""
function component_type(::Type{RandomFourierField})::String
    return "Fourier"
end

"""
    sample_component_params(field::RandomFourierField)

Sample Fourier parameters using specific bounds from field instance.

Allows customization of parameter bounds per component type instance.
"""
@gen function sample_component_params(field::RandomFourierField)
    amplitude ~ Gen.uniform(0, field.amplitude_max)
    frequency ~ Gen.uniform(0, field.freq_max)
    phase ~ Gen.uniform(0, 2π)
    
    return Dict(
        "amplitude" => amplitude,
        "frequency" => frequency,
        "phase" => phase
    )
end

"""
    make_component(::Type{RandomFourierField}, params::Dict{String, Any})::Function

Construct a Fourier scalar field from sampled parameters.

Implements the field formula: f(x, y) = A * cos(f*x + f*y + ϕ)

This matches the old discrete implementation's field formula, now with
continuous parameters sampled via Gen.jl instead of discretized bins.

# Arguments
- `params::Dict`: Must contain keys "amplitude", "frequency", "phase"

# Returns
- `Function`: Scalar field f(x::Real, y::Real)::Float64
"""
function make_component(::Type{RandomFourierField}, params::Dict{String, Any})::Function
    A = params["amplitude"]
    f = params["frequency"]
    φ = params["phase"]
    
    # Field formula: f(x,y) = A * cos(f*x + f*y + ϕ)
    function fourier_field(x::Real, y::Real)::Float64
        return A * cos(f * x + f * y + φ)
    end
    
    return fourier_field
end

"""
    describe_component_params(::Type{RandomFourierField})::String

Return documentation for Fourier component parameters.
"""
function describe_component_params(::Type{RandomFourierField})::String
    return """
    Component Field Type: Fourier (RandomFourierField)
    Description: Continuous sinusoidal field (replacing old discretized approach)
    
    Parameters:
      amplitude ∈ [0, 10]
        Distribution: uniform(0, 10) [continuous, sampled via Gen.jl]
        Impact: Controls peak magnitude of the cosine component
        
      frequency ∈ [0, π]
        Distribution: uniform(0, π) [continuous, sampled via Gen.jl]
        Impact: Controls spatial frequency/wavelength
        
      phase ∈ [0, 2π)
        Distribution: uniform(0, 2π) [continuous, sampled via Gen.jl]
        Impact: Controls phase shift of the cosine
    
    Field Formula: f(x, y) = A * cos(f * x + f * y + φ)
    
    Note: This is the continuous version replacing the old discrete binned approach.
    """
end