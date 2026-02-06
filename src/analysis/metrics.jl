export pf_degeneracy, objective_recon_metrics, policy_match_acc

"""
    pf_degeneracy(pf_state, π_dist; n_particles::Int)

Metric evaluating degeneracy of the particle filter.
"""
function pf_degeneracy(pf_state, π_dist; n_particles::Int)
    logw = get_log_weights(pf_state)
    finite = isfinite.(logw)
    all_ninf = !any(finite)

    tops = top_objectives(pf_state, π_dist; topk=5)
    nunique = length(tops)
    collapsed = (nunique == 1) && (!isempty(tops)) && (tops[1].count == n_particles)

    return (all_logw_ninf=all_ninf, nunique=nunique, collapsed=collapsed, ess=effective_sample_size(pf_state))
end

"""
    _zscore(Z)

Compute the Z-score of the objective reconstruction matrix.
"""
function _zscore(Z)
    μ = mean(Z)
    σ = std(vec(Z))
    σ = (σ ≤ 1e-12) ? 1.0 : σ
    return (Z .- μ) ./ σ
end

"""
    objective_recon_metrics(pf_state, π_dist, mdp; gridsize::Int=120)

Compute the RMSE of the objective field reconstruction.
"""
function objective_recon_metrics(pf_state, π_dist, mdp; gridsize::Int=120)
    tops = top_objectives(pf_state, π_dist; topk=1)
    isempty(tops) && return (rmse_z=NaN, corr=NaN)

    key = tops[1].key
    ff = decode_fourier_key(key, π_dist.fourier_cfg)
    field = make_fourier_scalar_field(ff; scaleQ=true)

    lo, hi = mdp.dimensions
    xs = range(lo, hi; length=gridsize)
    ys = range(lo, hi; length=gridsize)

    Zhat  = Matrix{Float64}(undef, length(ys), length(xs))
    Ztrue = Matrix{Float64}(undef, length(ys), length(xs))

    @inbounds for (j,y) in enumerate(ys), (i,x) in enumerate(xs)
        Zhat[j,i] = field(x,y)
        s = blindstart_KAgentState(mdp, [x y])
        Ztrue[j,i] = Float64(mdp.obj(s)[1])
    end

    A = vec(_zscore(Zhat))
    B = vec(_zscore(Ztrue))
    rmse = sqrt(mean((A .- B).^2))
    corr = dot(A,B) / (norm(A)*norm(B) + 1e-12)
    return (rmse_z=rmse, corr=corr)
end

"""
    policy_match_acc(pf_state, π_dist, agent_params, state_data, obs_aidx)

Evaluate the accuracy of the filter's top inferred policy against the true policy.

Currently evaluates using a deterministic approach (greedy argmax) instead of a non-deterministic method.
"""
function policy_match_acc(pf_state, π_dist, agent_params, state_data, obs_aidx)
    tops = top_objectives(pf_state, π_dist; topk=1)
    isempty(tops) && return (acc=NaN, N=0)
    key = tops[1].key
    mdp_hat = ensure_mdp!(π_dist, key)

    temperature = get(agent_params, :policy_temperature, 1.0)

    T = length(obs_aidx)
    pred = Vector{Int}(undef, T)
    @inbounds for t in 1:T
        s = blindstart_KAgentState(mdp_hat, reshape(state_data[:,t][1:2], (1,2)))
        b = vec(proposal_boltzmann(π_dist, key, s; temperature=temperature))
        pred[t] = argmax(b)
    end
    return (acc=mean(pred .== obs_aidx), N=T)
end