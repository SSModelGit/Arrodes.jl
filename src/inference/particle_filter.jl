"""
    particle_filter(observations::Vector{Int}, config::InferenceConfig, 
                    state_data::Matrix{Float64}, n_particles::Int = 50;
                    ess_thresh::Float64 = 0.5, resample_alg::Symbol = :residual,
                    frame_fns = nothing)

# Arguments
- `observations::Vector{Int}`: Sequence of observed actions (indices)
- `config::InferenceConfig`: Complete inference configuration with component tuples, RL config, etc.
- `state_data::Matrix{Float64}`: State data matrix where columns are states at each timestep
- `n_particles::Int`: Number of particles for the filter (default: 50)
- `ess_thresh::Float64`: ESS threshold for resampling as fraction of n_particles (default: 0.5)
- `resample_alg::Symbol`: Resampling algorithm (:residual, :multinomial, :stratified, default: :residual)
- `frame_fns::Union{Vector{Function}, Nothing}`: Frame generation functions.
  - If a Vector of Functions: each called at each timestep, producing separate frame vectors
  - If `nothing`, no frames are generated (default: nothing)

# Returns
- `state`: Particle filter state from Gen with traces and log_weights
- `frames`: 
  - If frame_fns is a Vector: Vector of frame vectors, one per function
  - If frame_fns is nothing: Empty Vector{Any}
"""
function particle_filter(observations::Vector{Int}, config::InferenceConfig, π_dist::ScoreΠDist,
                        state_data::Matrix{Float64}, n_particles::Int = 50;
                        ess_thresh::Float64 = 0.5, resample_alg::Symbol = :residual,
                        frame_fns = nothing)
    
    N = length(observations)
    obs_choices = [choicemap((n => :aidx, observations[n])) for n in 1:N]
    
    # Initialize frame storage: always a vector of vectors, one per frame function
    frames = isnothing(frame_fns) ? Vector{Any}() : [Vector{Any}() for _ in frame_fns]
    
    # ========== Phase 1: Initialize with first observation ==========
    # Note: inference_model_continuous takes cumulative observations[1:t]
    state = pf_initialize(inference_model, 
                         (config, observations[1:1], state_data[:, 1:1], π_dist), 
                         obs_choices[1], n_particles)
    
    # Record first frame if frame functions provided
    if !isnothing(frame_fns)
        for (i, fn) in enumerate(frame_fns)
            p = fn(state_data, 1, state, config)
            push!(frames[i], p)
        end
    end
    
    # ========== Phase 2: Sequential updates ==========
    for n in 2:N
        # Resample if ESS is low
        if effective_sample_size(state) < ess_thresh * n_particles
            pf_resample!(state, resample_alg)

            # Rejuvenation: use MH to refine component selections and parameters
            # Select addresses for component sampling to allow variation
            sels = Any[]
            for k in 1:config.k_components
                # sample_component_and_params is a @gen function with internal traces:
                # - component_idx from component_type_sampler()
                # - params from component_switch(component_idx)
                push!(sels, k => :component)
                push!(sels, k => :component)
            end
            
            # Also allow recent action choices to be refined
            a_lo = max(1, n - 3)  # refine last 3 actions
            for τ in a_lo:(n-1)
                push!(sels, (τ => :aidx))
            end
            
            pf_rejuvenate!(state, mh, (select(sels...),))
        end
        
        # Update with cumulative observations up to timestep n
        pf_update!(state,
                   (config, observations[1:n], state_data[:, 1:n], π_dist),
                   (NoChange(), UnknownChange(), UnknownChange(), NoChange()),
                   obs_choices[n])
        
        # Record frame if frame functions provided
        if !isnothing(frame_fns)
            for (i, fn) in enumerate(frame_fns)
                p = fn(state_data, n, state, config)
                push!(frames[i], p)
            end
        end
    end
    
    return (state, frames)
end

# ============================================================================
# Particle Information Extraction Utilities
# ============================================================================

"""
    extract_particle_component_info(trace::Dict, config::InferenceConfig, 
                                    component_fields::Vector)

Extract component selections and parameters from a particle's trace.

# Returns
- `(component_idxs, component_params)`: Tuple of vectors for each of the K components
  - `component_idxs::Vector{Int}`: Selected component type index for each component
  - `component_params::Vector{Dict}`: Parameter dictionaries for each component
"""
function extract_particle_component_info(trace, config::InferenceConfig,
                                        component_fields::Vector)
    component_idxs = Vector{Int}(undef, config.k_components)
    component_params = Vector{Dict}(undef, config.k_components)
    
    for k in 1:config.k_components
        # Access trace addresses set by inference_model_continuous
        # trace[k => :component] returns tuple (idx, params) from sample_component_and_params
        idx, params = trace[k => :component]
        component_idxs[k] = idx
        component_params[k] = params
    end
    
    return (component_idxs, component_params)
end

"""
    best_particle(pf_state, config::InferenceConfig, component_fields::Vector)

Return the highest-weight particle and its extracted component information.

# Returns
- `(best_idx, best_weight, component_idxs, component_params, objective_fn)`:
  - `best_idx::Int`: Index of the highest-weight particle
  - `best_weight::Float64`: Weight of the best particle
  - `component_idxs::Vector{Int}`: Component type selections from best particle
  - `component_params::Vector{Dict}`: Component parameters from best particle
  - `objective_fn::Function`: Reconstructed aggregate objective function f(x, y) -> Float64
"""
function best_particle(pf_state, config::InferenceConfig, component_fields::Vector)
    # Access traces and weights from particle filter state
    traces = pf_state.traces
    log_weights = pf_state.log_weights
    
    # Find best particle (highest log-weight)
    best_idx = argmax(log_weights)
    best_trace = traces[best_idx]
    best_log_weight = log_weights[best_idx]
    best_weight = exp(best_log_weight)
    
    # Extract component information from best particle
    (idxs, params) = extract_particle_component_info(best_trace, config, component_fields)
    
    # Reconstruct objective from best particle's components
    component_fns = [Priors.make_component(component_fields[idx], p)
                    for (idx, p) in zip(idxs, params)]
    
    objective_fn(x, y) = sum(f(x, y) for f in component_fns)
    
    return (best_idx, best_weight, idxs, params, objective_fn)
end