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

function proposal_boltzmann(π_dist::ScoreΠDist, mdp_key, config::InferenceConfig, obj::Function,
                            loc; temperature::Float64=1.0)
    mdp = ensure_mdp!(π_dist, mdp_key, obj, config)
    π_prop = get_π_proposal(π_dist, mdp_key, mdp, config)
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