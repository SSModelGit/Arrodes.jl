export effective_sample_size, conditional_effective_sample_size

function logsumexp(values)
    maximum_value = maximum(values)
    isfinite(maximum_value) || return maximum_value
    maximum_value + log(sum(exp(value - maximum_value) for value in values))
end

function normalize_logweights!(particles)
    normalizer = logsumexp(particle.log_weight for particle in particles)
    isfinite(normalizer) || error("every particle has zero target mass")
    for particle in particles
        particle.log_weight -= normalizer
    end
    normalizer
end

function effective_sample_size(particles)
    weights = exp.([particle.log_weight for particle in particles])
    inv(sum(abs2, weights))
end

effective_sample_size(cloud::ParticleCloud) = effective_sample_size(cloud.particles)
effective_sample_size(result::SMCResult) = effective_sample_size(result.state.cloud)

function conditional_effective_sample_size(log_weights, log_increments)
    log_terms = log_weights .+ log_increments
    numerator = 2logsumexp(log_terms)
    denominator = logsumexp(log_weights .+ 2 .* log_increments)
    exp(numerator - denominator)
end

function systematic_resample(rng, weights, count=length(weights))
    n = length(weights)
    cumulative = cumsum(weights)
    offset = rand(rng) / count
    ancestors = Vector{Int}(undef, count)
    cursor = 1
    for child in 1:count
        target = offset + (child - 1) / count
        while cursor < n && cumulative[cursor] < target
            cursor += 1
        end
        ancestors[child] = cursor
    end
    ancestors
end

function residual_resample(rng, weights)
    n = length(weights)
    counts = floor.(Int, n .* weights)
    ancestors = reduce(vcat, (fill(i, counts[i]) for i in eachindex(counts)); init=Int[])
    remaining = n - length(ancestors)
    if remaining > 0
        residual = n .* weights .- counts
        residual ./= sum(residual)
        append!(ancestors, systematic_resample(rng, residual, remaining))
    end
    shuffle!(rng, ancestors)
end

function weighted_mean_covariance(values, log_weights)
    weights = exp.(log_weights .- logsumexp(log_weights))
    samples = reduce(hcat, values)
    μ = samples * weights
    centered = samples .- μ
    weighted = centered .* sqrt.(weights)'
    Σ = weighted * weighted'
    (mean=μ, covariance=Matrix(Symmetric(Σ)), weights=weights)
end
