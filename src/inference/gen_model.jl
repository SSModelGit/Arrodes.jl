"""
    gen_K(cfg::FourierDiscreteCfg)

Generative function to sample number of Fourier features.
"""
@gen function gen_K(cfg::FourierDiscreteCfg)
    K ~ categorical(Priors.K_probs(cfg))   # returns 1..Kmax
    return K
end

"""
    gen_mode_indices(cfg::FourierDiscreteCfg)

Generative function to sample f, A, and ϕ for a Fourier feature.
"""
@gen function gen_mode_indices(cfg::FourierDiscreteCfg)
    f_supp, f_w = Priors.freq_bin_support_and_probs(cfg)
    a_supp, a_w = Priors.amp_bin_support_and_probs(cfg)
    p_supp, p_w = Priors.phase_bin_support_and_probs(cfg)
    fx_idx ~ categorical(f_w)
    fy_idx ~ categorical(f_w)
    A_idx  ~ categorical(a_w)
    ϕ_idx  ~ categorical(p_w)

    return (fx_i = f_supp[fx_idx],
            fy_i = f_supp[fy_idx],
            A_i  = a_supp[A_idx],
            ϕ_i  = p_supp[ϕ_idx])
end


@gen function gen_mode_indices(K::Int, cfg::FourierDiscreteCfg)
    f_supp, f_w = Priors.freq_bin_support_and_probs(cfg)
    a_supp, a_w = Priors.amp_bin_support_and_probs(cfg)
    p_supp, p_w = Priors.phase_bin_support_and_probs(cfg)

    fx_i = Vector{Int}(undef, K)
    fy_i = Vector{Int}(undef, K)
    A_i  = Vector{Int}(undef, K)
    ϕ_i  = Vector{Int}(undef, K)

    for m in 1:K
        fx_idx = @trace(categorical(f_w), (:mode, m) => :fx_idx)
        fy_idx = @trace(categorical(f_w), (:mode, m) => :fy_idx)
        A_idx  = @trace(categorical(a_w), (:mode, m) => :A_idx)
        ϕ_idx  = @trace(categorical(p_w), (:mode, m) => :ϕ_idx)

        fx_i[m] = f_supp[fx_idx]
        fy_i[m] = f_supp[fy_idx]
        A_i[m]  = a_supp[A_idx]
        ϕ_i[m]  = p_supp[ϕ_idx]
    end

    # return a vector of per-mode discrete indices
    out = Vector{Any}(undef, K)
    for m in 1:K
        out[m] = (fx_i[m], fy_i[m], A_i[m], ϕ_i[m])
    end
    return out
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
    fx = Priors.f_from_i.(fx_i, Ref(cfg))
    fy = Priors.f_from_i.(fy_i, Ref(cfg))
    A  = Priors.A_from_i.(A_i,  Ref(cfg))
    ϕ  = Priors.ϕ_from_i.(ϕ_i,  Ref(cfg))

    # stable cache key uses only the active prefix (1:K)
    key = (K, fx_i[1:K], fy_i[1:K], A_i[1:K], ϕ_i[1:K])

    return (key=key, K=K, fx=fx, fy=fy, A=A, ϕ=ϕ, fx_i=fx_i, fy_i=fy_i, A_i=A_i, ϕ_i=ϕ_i)
end


@gen function gen_fourier_bank_fixed(K::Int, cfg::FourierDiscreteCfg)
    # supports & probs (precompute once)
    f_supp, f_w = Priors.freq_bin_support_and_probs(cfg)
    a_supp, a_w = Priors.amp_bin_support_and_probs(cfg)
    p_supp, p_w = Priors.phase_bin_support_and_probs(cfg)

    fx_i = Vector{Int}(undef, K)
    fy_i = Vector{Int}(undef, K)
    A_i  = Vector{Int}(undef, K)
    ϕ_i  = Vector{Int}(undef, K)

    for m in 1:K
        fx_idx = @trace(categorical(f_w), (:mode, m) => :fx_idx)
        fy_idx = @trace(categorical(f_w), (:mode, m) => :fy_idx)
        A_idx  = @trace(categorical(a_w), (:mode, m) => :A_idx)
        ϕ_idx  = @trace(categorical(p_w), (:mode, m) => :ϕ_idx)

        fx_i[m] = f_supp[fx_idx]
        fy_i[m] = f_supp[fy_idx]
        A_i[m]  = a_supp[A_idx]
        ϕ_i[m]  = p_supp[ϕ_idx]
    end

    # continuous params for the active prefix
    fx = Priors.f_from_i.(fx_i, Ref(cfg))
    fy = Priors.f_from_i.(fy_i, Ref(cfg))
    A  = Priors.A_from_i.(A_i,  Ref(cfg))
    ϕ  = Priors.ϕ_from_i.(ϕ_i,  Ref(cfg))

    out = Vector{Any}(undef, K)
    for m in 1:K
        out[m] = (fx[m], fy[m], A[m], ϕ[m], fx_i[m], fy_i[m], A_i[m], ϕ_i[m])
    end
    return out
end

@gen function inference_model(N::Int, π_dist::ScoreΠDist, agent_params::Dict, state_data::Matrix)
    # sample discretized Fourier parameters (traceable)
    fourier = @trace(gen_fourier_bank_fixed(π_dist.fourier_cfg), :fourier)
    key = fourier.key
    # register for downstream reporting / priors
    RL.register_key_if_new!(π_dist, key)

    # lazy build mdp/policy (side-effecting cache)
    mdp = RL.ensure_mdp!(π_dist, key, fourier, agent_params)
    _   = RL.get_π_proposal(π_dist, key) # only use this to do lazy-loading as needed

    temp = get(agent_params, :policy_temperature, 1.0)
    for n in 1:N
        s = blindstart_KAgentState(mdp, reshape(state_data[:,n][1:2], (1,2)))
        boltzmann = max.(vec(RL.proposal_boltzmann(π_dist, key, s; temperature=temp)), 0.0)
        boltzmann ./= sum(boltzmann)
        _ = {n => :aidx} ~ categorical(boltzmann)
    end

    return key
end