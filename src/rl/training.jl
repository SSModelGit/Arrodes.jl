"""
    training_budget(prob; schedule=(200,600,1500), τ=(0.03,0.10,0.30))

Map posterior mass -> target training N.

- If prob ≥ τ3 => schedule[3]
- else if prob ≥ τ2 => schedule[2]
- else if prob ≥ τ1 => schedule[1]
- else => 0  (do not train yet)
"""
function training_budget(prob::Real; schedule::NTuple{3,Int}=(200, 600, 1500),
                         τ::NTuple{3,Float64}=(0.03, 0.10, 0.30))
    p = float(prob)
    if !isfinite(p) || p ≤ 0
        return 0
    elseif p ≥ τ[3]
        return schedule[3]
    elseif p ≥ τ[2]
        return schedule[2]
    elseif p ≥ τ[1]
        return schedule[1]
    else
        return 0
    end
end

"""
    surrogate_dataset_from_iql_grid(π_dist, π_iql, mdp;
                                   eval_num=200,
                                   dims=nothing)

Evaluate π_iql on a grid of points across the environment, returning:
- state_data::Matrix{Float64} of size (2, N)  [x;y] per column
- observations::Vector{Int} of length N       action index aidx per point
- eval_locations::Vector{Any} (optional convenience) the original [x y] points

Note: `mdp` should be the `mdp` used for training π_iql! Not another one.
"""
function surrogate_dataset_from_iql_grid(π_dist::ScoreΠDist,
                                        π_iql,
                                        mdp::KAgentPOMDP;
                                        eval_num::Int=200,
                                        dims=nothing)

    # ----- grid sampling (same idea as grid_points in geninf_on_single_trace.jl) -----
    # grid_points(n, dims=(0.,10.)) returns a Vector of 1×2 matrices [x y] :contentReference[oaicite:3]{index=3}
    if isnothing(dims)
        dims = mdp.dimensions
    end
    dim_span  = dims[2] - dims[1]
    dim_shift = dim_span / 20
    nx = ceil(Int, sqrt(eval_num))
    ny = ceil(Int, eval_num / nx)
    xs = range(dims[1] + dim_shift, dims[2] - dim_shift; length=nx)
    ys = range(dims[1] + dim_shift, dims[2] - dim_shift; length=ny)
    eval_locations = collect(Iterators.take(([x y] for y in ys for x in xs), eval_num))

    N = length(eval_locations)

    # ----- build state_data matrix (2 × N) -----
    state_data = Matrix{Float64}(undef, Crux.dim(state_space(mdp))[1], N)
    # @inbounds for i in 1:N
    #     # eval_locations[i] is 1×2; store as x,y rows
    #     state_data[1, i] = Float64(eval_locations[i][1])
    #     state_data[2, i] = Float64(eval_locations[i][2])
    # end

    # ----- map action symbol -> action index (aidx) -----
    alist = π_alist(π_dist)
    a_to_idx = Dict{Any, Int}(a => j for (j, a) in enumerate(alist))

    observations = Vector{Int}(undef, N)

    # ----- evaluate π_iql on each grid state (pattern from expected_recons_err_against_iql) -----
    # expected_recons_err_against_iql does:
    #   s_obs = MuKumari.shape_state_as_obs(mdp, blindstart_KAgentState(mdp, x))
    #   a*    = action(π_iql, s_obs)[1] :contentReference[oaicite:4]{index=4}
    @inbounds for i in 1:N
        s = blindstart_KAgentState(mdp, eval_locations[i])
        obs = MuKumari.shape_state_as_obs(mdp, s)
        state_data[:, i] = copy(obs)
        asymb = action(π_iql, obs)[1]

        idx = get(a_to_idx, asymb, 0)
        idx == 0 && error("π_iql returned action $(asymb) not found in π_alist(π_dist). Check action sets match.")
        observations[i] = idx
    end

    return state_data, observations, eval_locations
end

# Try to get a Flux/Crux model object we can copy params into/out of
"""
    _policy_model(π)

Tries to extract Flux/Crux model object out of a provided policy.

Fails elegantly by returning the value π in case the policy does not have it.
"""
_policy_model(π) = hasproperty(π, :model) ? getproperty(π, :model) : π

"""
    maybe_refine_policies!(π_dist, pf_state, agent_params;
                           topk=5, schedule=(200,600,1500), τ=(0.03,0.10,0.30))

Look at current PF posterior, choose a target budget per key via training_budget,
and call ensure_policy_trained_to! to escalate only those keys.
"""
function maybe_refine_policies!(π_dist::ScoreΠDist, pf_state, agent_params::Dict;
                               topk::Int=5,
                               schedule::NTuple{3,Int}=(200,600,1500),
                               τ::NTuple{3,Float64}=(0.03,0.10,0.30))
    tops = top_objectives(pf_state, π_dist; topk=topk)
    for item in tops
        # item.prob may be NaN during all -Inf weights; skip in that case
        prob = item.prob
        target = training_budget(prob; schedule=schedule, τ=τ)
        target == 0 && continue
        ensure_policy_trained_to!(π_dist, item.key, agent_params;
                                 target_steps=target, warm_start=true)
    end
    return nothing
end

"""
    _warm_start_params!(π_dest, π_src)

Warm-start a policy model by copying Flux model parameters over.

Assumed that the chosen source model is similar enough for this to make sense.
"""
function _warm_start_params!(π_dest, π_src)
    md = _policy_model(π_dest)
    ms = _policy_model(π_src)

    pd = Flux.params(md)
    ps = Flux.params(ms)

    if length(pd) != length(ps)
        @warn "Warm start skipped (param count mismatch)" nd=length(pd) ns=length(ps)
        return π_dest
    end

    for (d, s) in zip(pd, ps)
        if size(d) != size(s)
            @warn "Warm start skipped (param shape mismatch)" sized=size(d) sizes=size(s)
            return π_dest
        end
    end

    for (d, s) in zip(pd, ps)
        d .= s
    end
    return π_dest
end

"""
    ensure_policy_trained_to!(π_dist, key, agent_params;
                              target_steps, warm_start=true,
                              epochs=2, batch_size=512)

Ensures:
- an MDP exists for key (must already be in n_propmdp_list; created via inference_model)
- a SoftQ solver/policy exists
- training has been run up to `target_steps` (in solver N units)

Uses:
- nearest_trained_key(...) and hamming_fourier_key(...) for warm start
- stores trained steps in π_dist.n_𝒮_proposals[:_trained_steps]::Dict{Any,Int}

Returns: policy object (π_dist.n_π_proposals[key])
"""
function ensure_policy_trained_to!(π_dist::ScoreΠDist, key, agent_params::Dict;
                                  target_steps::Int,
                                  warm_start::Bool=true,
                                  epochs::Int=2,
                                  batch_size::Int=512)

    # bookkeeping dict (stored inside n_𝒮_proposals to avoid struct edits)
    trained = get!(π_dist.n_𝒮_proposals, :_trained_steps) do
        Dict{Any,Int}()
    end
    already = get(trained, key, 0)
    if already ≥ target_steps && haskey(π_dist.n_π_proposals, key)
        return π_dist.n_π_proposals[key]
    end

    # MDP must exist (inference_model should have created it)
    if !haskey(π_dist.n_propmdp_list, key)
        # If key hasn't been instantiated yet, we cannot train it here.
        # (The PF will create it once it samples it.)
        return get(π_dist.n_π_proposals, key, nothing)
    end
    mdp = ensure_mdp!(π_dist, key)

    # Build a solver for the *target* budget.
    solver = solver_from_type(mdp, :dql; solver_params=[:softq, target_steps, epochs, batch_size])

    # Warm start policy network parameters from nearest trained neighbor (if requested)
    if warm_start
        nn = nearest_trained_key(π_dist, key; min_trained=1)
        if nn !== nothing && haskey(π_dist.n_π_proposals, nn)
            try
                if hasproperty(solver, :π)
                    _warm_start_params!(getproperty(solver, :π), π_dist.n_π_proposals[nn])
                end
            catch err
                @warn "Warm start failed; training from scratch" err=err
            end
        end
    end

    # Train policy (solve)
    π = solve(solver, mdp)

    # Cache updated solver/policy and trained steps
    π_dist.n_𝒮_proposals[key] = solver
    π_dist.n_π_proposals[key] = π
    trained[key] = target_steps

    return π
end


# Continuous Component API: MDP creation from objective function
ensure_mdp!(π_dist::ScoreΠDist, key, obj::Function, agent_config::InferenceConfig) = get!(π_dist.n_propmdp_list, key) do
    build_kagent_pomdp(agent_config.agent_params, obj; name="component_" * string(key))
end

function get_𝒮_proposal(π_dist::ScoreΠDist, key, mdp::Any, config::InferenceConfig)
    return get!(π_dist.n_𝒮_proposals, key) do
        deep_q_solver(mdp; solver_params=[:softq,
                                          config.rl_config.n_iterations,
                                          config.rl_config.epochs,
                                          config.rl_config.batch_size])
    end
end

# Continuous Component API: returns (solver, policy) tuple
get_π_proposal(π_dist::ScoreΠDist, key, mdp::Any, config::InferenceConfig) = get!(π_dist.n_π_proposals, key) do
    solver = get_𝒮_proposal(π_dist, key, mdp, config)
    solve(solver, mdp)
end