using LinearAlgebra
using SCRIBE

# When the observed agent's environmental measurements are available, the
# inverse behavior problem disappears. Fuse those measurements through SCRIBE
# directly; do not route them through Arrodes world inference.
locations = [0.0 0.0; 1.0 0.0; 0.0 1.0; 1.0 1.0]
snapshots = [
    0.0 1.0 0.2 1.1 0.1
    0.2 1.2 0.1 1.0 0.3
    1.0 0.0 1.1 0.1 0.9
    1.2 0.2 1.0 0.0 1.1
]
rank = 2
covariance = 0.5 .* Matrix{Float64}(I, rank, rank)
model = initialize_eof_climate_model(
    snapshots;
    locations=locations,
    rank=rank,
    process_covariance=0.01 .* Matrix{Float64}(I, rank, rank),
    prior_covariance=covariance,
)
prior = SCRIBE.init_agent_info(model.params; prior_covariance=covariance)

observed_locations = [0.0 0.0; 1.0 1.0]
observed_values = [0.35, 0.95]
sensor_covariance = 0.05 .* Matrix{Float64}(I, 2, 2)
effective_covariance = SCRIBE.eof_effective_measurement_covariance(
    model, observed_locations, sensor_covariance,
)
posterior = SCRIBE.condition_on_measurement(
    model, prior, observed_locations, observed_values, effective_covariance,
)

println("Prior coefficient moments: ", SCRIBE.posterior_coefficient_moments(prior))
println("Directly fused moments: ", SCRIBE.posterior_coefficient_moments(posterior))
