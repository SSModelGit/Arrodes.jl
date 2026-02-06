"""
    gen_K(cfg::FourierDiscreteCfg)

Generative function to sample number of Fourier features.
"""
@gen function gen_K(cfg::FourierDiscreteCfg)
    K ~ categorical(K_probs(cfg))   # returns 1..Kmax
    return K
end

"""
    gen_mode_indices(cfg::FourierDiscreteCfg)

Generative function to sample f, A, and ϕ for a Fourier feature.
"""
@gen function gen_mode_indices(cfg::FourierDiscreteCfg)
    f_supp, f_w = freq_bin_support_and_probs(cfg)
    a_supp, a_w = amp_bin_support_and_probs(cfg)
    p_supp, p_w = phase_bin_support_and_probs(cfg)

    fx_idx ~ categorical(f_w)
    fy_idx ~ categorical(f_w)
    A_idx  ~ categorical(a_w)
    ϕ_idx  ~ categorical(p_w)

    return (fx_i = f_supp[fx_idx],
            fy_i = f_supp[fy_idx],
            A_i  = a_supp[A_idx],
            ϕ_i  = p_supp[ϕ_idx])
end


"""
    gen_fourier_bank(cfg::FourierDiscreteCfg)

Composes the Fourier feature sampling process:
* First, samples number of features to be used
* Second, samples the parameters for each feature (f, A, ϕ).

Returns a cached set of keys mapping to each feature and associated params.
"""
@gen function gen_fourier_bank_fixed(cfg::FourierDiscreteCfg)
    # K in 1..Kmax
    K = @trace(gen_K(cfg), :K)

    # supports & probs (precompute once)
    f_supp, f_w = freq_bin_support_and_probs(cfg)
    a_supp, a_w = amp_bin_support_and_probs(cfg)
    p_supp, p_w = phase_bin_support_and_probs(cfg)

    # fixed bank of discrete indices (length Kmax)
    fx_i = Vector{Int}(undef, cfg.Kmax)
    fy_i = Vector{Int}(undef, cfg.Kmax)
    A_i  = Vector{Int}(undef, cfg.Kmax)
    ϕ_i  = Vector{Int}(undef, cfg.Kmax)

    for m in 1:cfg.Kmax
        fx_idx = @trace(categorical(f_w), (:mode, m) => :fx_idx)
        fy_idx = @trace(categorical(f_w), (:mode, m) => :fy_idx)
        A_idx  = @trace(categorical(a_w), (:mode, m) => :A_idx)
        ϕ_idx  = @trace(categorical(p_w), (:mode, m) => :ϕ_idx)

        fx_i[m] = f_supp[fx_idx]
        fy_i[m] = f_supp[fy_idx]
        A_i[m]  = a_supp[A_idx]
        ϕ_i[m]  = p_supp[ϕ_idx]
    end

    # continuous params for the full bank
    fx = f_from_i.(fx_i, Ref(cfg))
    fy = f_from_i.(fy_i, Ref(cfg))
    A  = A_from_i.(A_i,  Ref(cfg))
    ϕ  = ϕ_from_i.(ϕ_i,  Ref(cfg))

    # stable cache key uses only the active prefix (1:K)
    key = (K, fx_i[1:K], fy_i[1:K], A_i[1:K], ϕ_i[1:K])

    return (key=key, K=K, fx=fx, fy=fy, A=A, ϕ=ϕ, fx_i=fx_i, fy_i=fy_i, A_i=A_i, ϕ_i=ϕ_i)
end

@gen function inference_model(N::Int, π_dist::ScoreΠDist, agent_params::Dict, state_data::Matrix)
    # sample discretized Fourier parameters (traceable)
    fourier = @trace(gen_fourier_bank_fixed(π_dist.fourier_cfg), :fourier)
    key = fourier.key
    # register for downstream reporting / priors
    register_key_if_new!(π_dist, key)

    # lazy build mdp/policy (side-effecting cache)
    mdp = ensure_mdp!(π_dist, key, fourier, agent_params)
    _   = get_π_proposal(π_dist, key) # only use this to do lazy-loading as needed

    temp = get(agent_params, :policy_temperature, 1.0)
    for n in 1:N
        s = blindstart_KAgentState(mdp, reshape(state_data[:,n][1:2], (1,2)))
        boltzmann = max.(vec(proposal_boltzmann(π_dist, key, s; temperature=temp)), 0.0)
        boltzmann ./= sum(boltzmann)
        _ = {n => :aidx} ~ categorical(boltzmann)
    end

    return key
end