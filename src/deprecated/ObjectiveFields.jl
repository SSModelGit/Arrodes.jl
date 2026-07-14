module ObjectiveFields

using Random

abstract type ComponentField end

Base.@kwdef struct RandomFourierField <: ComponentField
    amplitude_max::Float64 = 10.0
    freq_max::Float64 = pi
end

Base.@kwdef struct RadialBasisField <: ComponentField
    x_min::Float64 = 0.0
    x_max::Float64 = 10.0
    y_min::Float64 = 0.0
    y_max::Float64 = 10.0
    amp_min::Float64 = -10.0
    amp_max::Float64 = 10.0
    σ::Float64 = 1.0
end

component_type(::RandomFourierField) = "Fourier"
component_type(::RadialBasisField) = "RBF"

function fourier_params_sampler(field::RandomFourierField)
    Base.depwarn("generic RFF objectives are retained only for legacy experiments and are not inference hypotheses", :fourier_params_sampler)
    return function (rng::AbstractRNG = Random.default_rng())
        Dict("amplitude" => rand(rng) * field.amplitude_max,
            "frequency_x" => rand(rng) * field.freq_max,
            "frequency_y" => rand(rng) * field.freq_max,
            "phase" => rand(rng) * 2pi)
    end
end

function rbf_params_sampler(field::RadialBasisField)
    Base.depwarn("generic RBF objectives are retained only for legacy experiments and are not inference hypotheses", :rbf_params_sampler)
    return function (rng::AbstractRNG = Random.default_rng())
        Dict("center_x" => field.x_min + rand(rng) * (field.x_max - field.x_min),
            "center_y" => field.y_min + rand(rng) * (field.y_max - field.y_min),
            "amplitude" => field.amp_min + rand(rng) * (field.amp_max - field.amp_min),
            "sigma" => field.σ)
    end
end

function make_component(::RandomFourierField, params::AbstractDict)
    Base.depwarn("RFF objective composition is deprecated and excluded from filtering", :make_component)
    amplitude = params["amplitude"]
    frequency_x = params["frequency_x"]
    frequency_y = params["frequency_y"]
    phase = params["phase"]
    return (x, y) -> amplitude * cos(frequency_x * x + frequency_y * y + phase)
end

function make_component(::RadialBasisField, params::AbstractDict)
    Base.depwarn("RBF objective composition is deprecated and excluded from filtering", :make_component)
    center_x, center_y = params["center_x"], params["center_y"]
    amplitude, sigma = params["amplitude"], params["sigma"]
    return (x, y) -> amplitude * exp(-((x - center_x)^2 + (y - center_y)^2) / (2sigma^2))
end

make_pomdp_objective_from_field(field::Function) =
    state -> Any[field(state.x[1, 1], state.x[1, 2]), false]

function objective_grid_from_field(field::Function, xs, ys)
    return [Float64(field(x, y)) for y in ys, x in xs]
end

describe_component_params(::RandomFourierField) = "Deprecated continuous RFF: amplitude, x/y frequency, phase."
describe_component_params(::RadialBasisField) = "Deprecated Gaussian RBF: center, amplitude, sigma."

export ComponentField, RandomFourierField, RadialBasisField, component_type,
    fourier_params_sampler, rbf_params_sampler, make_component,
    make_pomdp_objective_from_field, objective_grid_from_field, describe_component_params

end
