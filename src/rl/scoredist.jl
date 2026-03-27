# define getter functions
get_proposal_names(π_dist::ScoreΠDist) = π_dist.prop_names
# get_proposal_component_priors(π_dist::ScoreΠDist) = π_dist.q_objs
# get_proposal_component_objectives(π_dist::ScoreΠDist, proposal) = π_dist.n_compobj_list[proposal]
get_proposal_prior(π_dist::ScoreΠDist, proposal) = π_dist.n_qprop_list[proposal]
get_idxable_proposal_prior_list(π_dist::ScoreΠDist) = [get_proposal_prior(π_dist, p) for p in get_proposal_names(π_dist)]

π_alist(π_dist::ScoreΠDist) = π_dist.mdp_params[1]
π_a_1hot(π_dist::ScoreΠDist) = π_dist.mdp_params[2]
π_a_1hotall(π_dist::ScoreΠDist) = π_dist.mdp_params[3]

# lazy create mdp if missing
ensure_mdp!(π_dist::ScoreΠDist, key) = get!(π_dist.n_propmdp_list, key) do
    @error "ya fucked up, where's the mdp at"
end

# Discrete Fourier API: MDP creation from Fourier field
ensure_mdp!(π_dist::ScoreΠDist, key, ff, agent_params::Dict) = get!(π_dist.n_propmdp_list, key) do
    field = make_fourier_scalar_field(ff; scaleQ=true)
    obj   = make_pomdp_objective_from_field(field)

    build_kagent_pomdp(agent_params, obj; name="fourier_" * string(hash(key)))
end

# Continuous Component API: MDP creation from objective function
ensure_mdp!(π_dist::ScoreΠDist, key, obj::Function, agent_params::Dict, ::InferenceConfig) = get!(π_dist.n_propmdp_list, key) do
    build_kagent_pomdp(agent_params, obj; name="component_" * string(hash(key)))
end

# lazy solver
# Discrete Fourier API: assumes MDP is already cached
get_𝒮_proposal(π_dist::ScoreΠDist, key) = get!(π_dist.n_𝒮_proposals, key) do
    mdp = ensure_mdp!(π_dist, key)
    # specify Deep Q-learning approach; choose Soft-Q learning, for 2000 iterations (empirically selected)
    solver_from_type(mdp, :dql; solver_params=[:softq, 200, 2, 512])
end

# lazy policy
# Discrete Fourier API: uses cached solver and MDP
get_π_proposal(π_dist::ScoreΠDist, key) = get!(π_dist.n_π_proposals, key) do
    𝒮 = get_𝒮_proposal(π_dist, key)
    mdp = ensure_mdp!(π_dist, key)
    solve(𝒮, mdp)
end

# Continuous Component API: returns (solver, policy) tuple
get_π_proposal(π_dist::ScoreΠDist, key, mdp::Any, config::InferenceConfig) = get!(π_dist.n_π_proposals, key) do
    solver, policy = get_𝒮_proposal(π_dist, key, mdp, config)
    (solver, policy)
end

store_π_iql(π_dist::ScoreΠDist, π_iql) = push!(π_dist.n_π_proposals, :iql=>π_iql)
get_π_iql(π_dist::ScoreΠDist) = get(π_dist.n_π_proposals, :iql, nothing)

"""
Register a newly seen key into the proposal set, if absent.
Optionally can set a default prior mass here;
TODO: simplest is uniform mass then renormalize.
"""
function register_key_if_new!(π_dist::ScoreΠDist, key; prior_mass::Float64=1.0)
    if !(key in π_dist.prop_names)
        push!(π_dist.prop_names, key)
        π_dist.n_qprop_list[key] = prior_mass
    end
    return key
end

"""
    top_objectives(pf_state, π_dist; topk=10)

Aggregates posterior mass by objective key (= trace return value).
Returns top-k with:
- key
- prob mass
- count
- decoded Fourier params
"""
function top_objectives(pf_state, π_dist::ScoreΠDist; topk::Int = 10)
    traces = get_traces(pf_state)
    logw   = get_log_weights(pf_state)

    # robust normalization (handles large negative logw); if all -Inf => fallback to counts
    finite = isfinite.(logw)
    if !any(finite)
        # no numeric weights available; return empirical counts only
        counts = Dict{Any,Int}()
        for tr in traces
            key = get_retval(tr)
            counts[key] = get(counts, key, 0) + 1
        end
        keys_sorted = sort(collect(keys(counts)), by=k->counts[k], rev=true)
        Kout = min(topk, length(keys_sorted))
        return [(key=keys_sorted[j],
                 prob=NaN,  # explicitly “unknown”
                 count=counts[keys_sorted[j]],
                 params=decode_fourier_key(keys_sorted[j], π_dist.fourier_cfg))
                for j in 1:Kout]
    end

    lw = logw[finite]
    tr = traces[finite]

    m = maximum(lw)
    w = exp.(lw .- m)
    Z = sum(w)
    p = w ./ Z

    mass   = Dict{Any,Float64}()
    counts = Dict{Any,Int}()

    @inbounds for i in eachindex(tr)
        key = get_retval(tr[i])
        mass[key]   = get(mass, key, 0.0) + p[i]
        counts[key] = get(counts, key, 0) + 1
    end

    keys_sorted = sort(collect(keys(mass)), by=k->mass[k], rev=true)
    Kout = min(topk, length(keys_sorted))

    return [(key=keys_sorted[j],
             prob=mass[keys_sorted[j]],
             count=counts[keys_sorted[j]],
             params=decode_fourier_key(keys_sorted[j], π_dist.fourier_cfg))
            for j in 1:Kout]
end

"""
    top_key(pf_state, π_dist) -> (key, prob)

Convenience accessor for top posterior objective key.
"""
function top_key(pf_state, π_dist::ScoreΠDist)
    tops = top_objectives(pf_state, π_dist; topk=1)
    isempty(tops) && error("top_objectives returned empty; cannot plot.")
    return tops[1].key, tops[1].prob
end