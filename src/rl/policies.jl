"""
    proposal_boltzmann(π_dist::ScoreΠDist, prop_name, loc::KAgentState)


Computes Boltzmann distribution for π_{prop_name}(s).

* `π_dist`: ScoreΠDist object containing all proposal policies and their associated MDPs.
* `prop_name`: Identifier (e.g. symbol or string) for the proposal whose policy should be evaluated.
* `loc`: Current state/location at which the policy is evaluated.
  * Assumed to be in KAgentState form; `MuKumari.shape_state_as_obs` is used internally to convert it.
  * TODO: Bad behavior to use a non-exported function! Either export or choose different approach.

The policy is evaluated using `Crux.value` on the one-hot action set, producing unnormalized action scores.
These are normalized with a softmax (after subtracting the maximum for numerical stability) to obtain a Boltzmann distribution.

Returns: `boltzmann`
* `boltzmann`::Matrix{Float64} giving the Boltzmann action distribution.
  * Rows correspond to states (only one).
  * Columns correspond to actions.
  * Values are cast to `Float64` (from `Float32`, e.g. when computed on GPU) for compatibility with Gen’s tracing and scoring machinery.
"""
function proposal_boltzmann(π_dist::ScoreΠDist, prop_name, loc; temperature::Float64=1.0)
    mdp = ensure_mdp!(π_dist, prop_name)
    π_prop = get_π_proposal(π_dist, prop_name)
    all_a_onehot = π_a_1hotall(π_dist)

    # assume state location already in obs vec form, otherwise need to use MuKumari.shape_state_as_obs(loc)
    q = Crux.value(π_prop, MuKumari.shape_state_as_obs(mdp, loc), all_a_onehot)
    # Stability + temperature
    T = max(temperature, 1e-6)
    logits = (q .- maximum(q, dims=2)) ./ T

    boltzmann = softmax(logits, dims=2)

    # cast boltzmann distribution into Float64 form, from as the GPU operates in Float32
    return Float64.(boltzmann)
end

"""
    greedy_action_symbol_from_boltzmann(π_dist, key, s) -> (a_sym, a_idx)

Non-deterministic choice of action by Boltzmann over Q-values.

Uses `proposal_boltzmann(...)` machinery to compute a Boltzmann
distribution over actions and then selects argmax (greedy).

This gives a deterministic rollout for visual comparison.

Returns (a::Symbol, aidx::Int)
"""
function greedy_action_symbol_from_boltzmann(π_dist::ScoreΠDist, key, s::KAgentState)
    b = vec(proposal_boltzmann(π_dist, key, s))
    # guard against tiny negatives/nans
    b = max.(b, 0.0)
    if !(isfinite(sum(b))) || sum(b) <= 0
        # fall back to uniform if something went wrong numerically
        b .= 1.0
    end
    aidx = argmax(b)
    asymb = π_alist(π_dist)[aidx]
    return asymb, aidx
end

"""
    rollout_greedy_policy(π_dist, key; start_state, T) -> (xs, ys, states)

Rolls out the policy induced by the proposal's Q-function on its own MDP.
Uses greedy selection from the Boltzmann distribution (argmax over actions).
"""
function rollout_greedy_policy(π_dist::ScoreΠDist, key;
                               start_state::KAgentState,
                               T::Int)
    mdp = ensure_mdp!(π_dist, key)
    s = copy(start_state)
    xs = Vector{Float64}(undef, T)
    ys = Vector{Float64}(undef, T)
    states = Vector{KAgentState}(undef, T)

    @inbounds for t in 1:T
        xs[t] = Float64(s.x[1,1])
        ys[t] = Float64(s.x[1,2])
        states[t] = copy(s)

        asymb, _ = greedy_action_symbol_from_boltzmann(π_dist, key, s)

        # Transition using the POMDP generative step like in inference_model
        s = POMDPs.@gen(:sp)(mdp, s, asymb)
    end

    return xs, ys, states
end

"""
    qpolicy_action(π, mdp, s; temperature=1.0, rng=...)

Non-deterministic choice of action by Boltzmann over Q-values.
Returns (a::Symbol, aidx::Int, probs::Vector{Float64})
"""
function qpolicy_action(π, mdp::KAgentPOMDP, s::KAgentState;
                        temperature::Real=1.0,
                        rng=Random.default_rng())

    as = actions(mdp)
    all_a_onehot = Flux.onehotbatch(as, as)
    obs = MuKumari.shape_state_as_obs(mdp, s)

    q = vec(Crux.value(π, obs, all_a_onehot))
    qmax = maximum(q)
    logits = (q .- qmax) ./ temperature
    p = exp.(logits)
    p ./= sum(p)

    aidx = randcat(rng, p)
    return as[aidx], aidx, Float64.(p)
end

"""
    softq_policy(mdp; N=2000, epochs=2, batch_size=256)

Trains SoftQ via deep_q_solver and returns (solver, policy).
"""
function softq_policy(mdp::KAgentPOMDP; N::Int=2000, epochs::Int=2, batch_size::Int=256)
    𝒮 = deep_q_solver(mdp; solver_params=[:softq, N, epochs, batch_size])
    π = solve(𝒮, mdp)
    return 𝒮, π
end

"""
    rollout_experience_buffer(mdp, π; T=20, temperature=1.0, rng=...)
"""
function rollout_experience_buffer(mdp::KAgentPOMDP, π;
                                   T::Int=20,
                                   temperature::Real=1.0,
                                   rng=Random.default_rng())

    as = actions(mdp)
    na = length(as)

    # Determine obs dimension robustly
    s0 = rand(initialstate(mdp))
    obs0 = MuKumari.shape_state_as_obs(mdp, s0)
    obs_dim = length(obs0)

    data = alloc_buffer_dict(obs_dim, na, T)

    s = s0
    for t in 1:T
        a, aidx, _ = qpolicy_action(π, mdp, s; temperature=temperature, rng=rng)

        # transition
        nt = POMDPs.gen(mdp, s, a, rng)
        sp = nt.sp
        r  = nt.r

        # store onehot action column (Bool matrix)
        data[:a][:, t] .= false
        data[:a][aidx, t] .= true

        # store obs-shaped state, next-state
        data[:s][:, t]  .= Float64.(shape_state_as_obs(mdp, s))
        data[:sp][:, t] .= Float64.(shape_state_as_obs(mdp, sp))

        # store reward (1×T)
        data[:r][1, t] = Float64(r)

        s = sp
    end

    return mk_experience_buffer(data)
end