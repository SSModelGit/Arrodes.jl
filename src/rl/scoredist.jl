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

store_π_iql(π_dist::ScoreΠDist, π_iql) = push!(π_dist.n_π_proposals, :iql=>π_iql)
get_π_iql(π_dist::ScoreΠDist) = get(π_dist.n_π_proposals, :iql, nothing)