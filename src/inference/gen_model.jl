@gen function inference_model(config::InferenceConfig,
                                         observations::Vector{Int},
                                         state_data::Matrix{Float64},
                                         π_dist::ScoreΠDist)
    component_switch = config.component_params_switch
    component_sampler = config.component_type_sampler
    # Phase 1: Sample component types and parameters (Gen-traced)
    component_indices = Vector{Int}(undef, config.k_components)
    component_params = Vector{Dict}(undef, config.k_components)
    
    for k in 1:config.k_components
        idx, params = @trace(Priors.sample_component_and_params(component_switch,
                                                                component_sampler), 
                             k => :component)
        component_indices[k] = idx
        component_params[k] = params
    end
    
    # Phase 2: Construct aggregate objective (deterministic)
    component_fields_vec = [config.component_tuples[idx][1] for idx in component_indices]
    component_fns = [Priors.make_component(component_fields_vec[k], component_params[k]) 
                     for k in 1:config.k_components]
    
    aggregate_objective(x::Real, y::Real) = sum([fn(x, y) for fn in component_fns])
    objective_fn = Priors.make_pomdp_objective_from_field(aggregate_objective)
    
    # Generate cache key from component configuration
    config_key = hash((component_indices, [Dict(collect(p)) for p in component_params]))

    # Phase 3: Build MDP and train policy with caching via π_dist
    mdp = RL.ensure_mdp!(π_dist, config_key, objective_fn, config)
    
    # Use π_dist's caching infrastructure via multi-dispatch
    RL.get_π_proposal(π_dist, config_key, mdp, config)

    # Phase 4: Sample actions from learned policy (Gen-traced)
    temperature = config.rl_config.temperature
    
    for n in 1:length(observations)
        s = blindstart_KAgentState(mdp, reshape(state_data[:, n][1:2], (1,2)))
        boltzmann = vec(RL.proposal_boltzmann(π_dist, config_key, config, objective_fn, s; temperature=temperature))
        boltzmann = boltzmann ./ max(sum(boltzmann), 1e-10)
        action_idx = @trace(categorical(boltzmann), n => :aidx)
    end
    
    return component_indices
end

"""
    extract_component_info(trace::Dict, config::InferenceConfig) -> Dict

Extract component indices and parameters from trace.

Returns Dict with :component_indices and :component_params keys.
"""
function extract_component_info(trace, config::InferenceConfig)
    component_indices = Int[]
    component_params = Dict[]
    
    for k in 1:config.k_components
        idx, params = trace[k => :component]
        push!(component_indices, idx)
        push!(component_params, params)
    end
    
    return Dict(
        :component_indices => component_indices,
        :component_params => component_params
    )
end

"""
    reconstruct_objective_from_trace(trace::Dict, config::InferenceConfig) -> Function

Reconstruct aggregate objective (x, y) -> Float64 from trace.
"""
function reconstruct_objective_from_trace(trace, config::InferenceConfig)
    info = extract_component_info(trace, config)
    component_indices = info[:component_indices]
    component_params = info[:component_params]
    component_fields = [config.component_tuples[idx][1] for idx in component_indices]
    
    component_fns = [Priors.make_component(component_fields[k], component_params[k]) 
                     for k in 1:config.k_components]
    
    return (x, y) -> sum([fn(x, y) for fn in component_fns])
end