################
# RBF Discretization
################

"""
    center_bin_support_and_probs(cfg::RBFDiscreteCfg; n_bins::Int=20)

Constructs support and uniform probabilities for discretized spatial bins.
Returns separate supports for x and y coordinates.
"""
function center_bin_support_and_probs(cfg::RBFDiscreteCfg; n_bins::Int=20)
    x_supp = collect(range(cfg.x_min, cfg.x_max; length=n_bins))
    y_supp = collect(range(cfg.y_min, cfg.y_max; length=n_bins))
    
    x_w = fill(1.0 / n_bins, n_bins)
    y_w = fill(1.0 / n_bins, n_bins)
    
    return x_supp, y_supp, x_w, y_w
end

"""
    amplitude_bin_support_and_probs(cfg::RBFDiscreteCfg; n_amps::Int=10)

Constructs support and uniform probabilities for RBF amplitudes.
Amplitudes are positive and span a reasonable range.
"""
function amplitude_bin_support_and_probs(cfg::RBFDiscreteCfg; n_amps::Int=10)
    amp_supp = collect(range(0.1, 5.0; length=n_amps))
    amp_w = fill(1.0 / n_amps, n_amps)
    return amp_supp, amp_w
end

################################
# RBF Feature Key Operations
################################

"""
    sample_rbf_key(cfg::RBFDiscreteCfg;
                    K_override::Union{Nothing,Int}=nothing,
                    n_bins::Int=20,
                    n_amps::Int=10,
                    rng=Random.default_rng())

Sample a discrete RBF key specifying centers, amplitudes, and bandwidth.

Returns tuple: (K, x_indices, y_indices, amp_indices)
where each index vector has length K.
"""
function sample_rbf_key(cfg::RBFDiscreteCfg;
                        K_override::Union{Nothing,Int}=nothing,
                        n_bins::Int=20,
                        n_amps::Int=10,
                        rng=Random.default_rng())

    # Supports/probs
    Kp = K_probs(cfg)
    x_supp, y_supp, x_w, y_w = center_bin_support_and_probs(cfg; n_bins=n_bins)
    amp_supp, amp_w = amplitude_bin_support_and_probs(cfg; n_amps=n_amps)

    K = isnothing(K_override) ? rand(rng, Categorical(Kp)) : K_override
    K = clamp(K, 1, cfg.Kmax)

    x_idx = Vector{Int}(undef, K)
    y_idx = Vector{Int}(undef, K)
    amp_idx = Vector{Int}(undef, K)

    for m in 1:K
        x_idx[m] = randcat(rng, x_w)
        y_idx[m] = randcat(rng, y_w)
        amp_idx[m] = randcat(rng, amp_w)
    end

    key = (K, x_idx, y_idx, amp_idx)
    return key
end

"""
    decode_rbf_key(key, cfg::RBFDiscreteCfg;
                    n_bins::Int=20,
                    n_amps::Int=10)

Decode RBF key of the form (K, x_idx, y_idx, amp_idx) into continuous params.

Returns NamedTuple with fields:
- `K::Int`: Number of RBF centers
- `x::Vector{Float64}`: x-coordinates of centers
- `y::Vector{Float64}`: y-coordinates of centers
- `amp::Vector{Float64}`: Amplitude of each Gaussian
- `x_idx, y_idx, amp_idx::Vector{Int}`: Discrete indices
"""
function decode_rbf_key(key, cfg::RBFDiscreteCfg;
                        n_bins::Int=20,
                        n_amps::Int=10)

    K, x_idx, y_idx, amp_idx = key

    x_supp, y_supp, _, _ = center_bin_support_and_probs(cfg; n_bins=n_bins)
    amp_supp, _ = amplitude_bin_support_and_probs(cfg; n_amps=n_amps)

    # Extract only active prefix (1:K)
    x = x_supp[x_idx[1:K]]
    y = y_supp[y_idx[1:K]]
    amp = amp_supp[amp_idx[1:K]]

    return (
        K=K,
        x=x,
        y=y,
        amp=amp,
        x_idx=x_idx[1:K],
        y_idx=y_idx[1:K],
        amp_idx=amp_idx[1:K]
    )
end

"""
    hamming_rbf_key(k1, k2) -> Int

Hamming distance on RBF discrete key representation.

Key format assumed:
    (K::Int, x_idx::Vector{Int}, y_idx::Vector{Int}, amp_idx::Vector{Int})

Only compare active prefixes (1:K), and add abs(K1-K2).
Each mode has 3 discrete indices (x, y, amp), so unmatched modes count as 3.
"""
function hamming_rbf_key(k1, k2)
    K1, x1, y1, a1 = k1
    K2, x2, y2, a2 = k2
    d = abs(K1 - K2)

    K = min(K1, K2)
    @inbounds for m in 1:K
        d += (x1[m] != x2[m])
        d += (y1[m] != y2[m])
        d += (a1[m] != a2[m])
    end

    # treat unmatched tail entries as mismatches
    if K1 != K2
        Kbig = max(K1, K2)
        d += 3 * (Kbig - K)  # each extra RBF has 3 discrete indices
    end
    return d
end

"""
    nearest_trained_key(π_dist, key; min_trained=1)

Returns the closest RBF key among those already in π_dist.n_π_proposals and
whose training steps record indicates ≥ min_trained.
Returns `nothing` if none exist.
"""
function nearest_trained_key_rbf(π_dist::ScoreΠDist, key; min_trained::Int=1)
    best = nothing
    best_d = typemax(Int)

    # fall back if training bookkeeping not present yet
    steps = get!(π_dist.n_𝒮_proposals, :_trained_steps) do
        Dict{Any,Int}()
    end

    for k in keys(π_dist.n_π_proposals)
        # skip non-keys (e.g. :iql)
        k isa Tuple || continue
        get(steps, k, 0) ≥ min_trained || continue

        # Try RBF hamming distance; skip if key format mismatch
        try
            d = hamming_rbf_key(key, k)
            if d < best_d
                best = k
                best_d = d
            end
        catch
            continue
        end
    end
    return best
end

##############################
# RBF Field Construction
##############################

"""
    make_rbf_scalar_field(bank; σ::Float64=1.0)

Returns: field(x::Real, y::Real)::Float64

Definition:
  field(x,y) = Σ_{m=1..K} amp[m] * exp(-((x - x[m])² + (y - y[m])²) / (2σ²))

The bank can be either:
- A NamedTuple with fields: K, x, y, amp
- A Vector of per-center tuples (x, y, amp)

# Arguments
- `bank`: RBF parameters (NamedTuple or Vector of tuples)
- `σ`: Gaussian bandwidth (can override or use from bank if present)
"""
function make_rbf_scalar_field(bank; σ::Float64=1.0)
    # accept either the named-tuple bank or a Vector of per-center tuples
    if bank isa AbstractVector
        centers = bank
        K = length(centers)
        x = [c[1] for c in centers]
        y = [c[2] for c in centers]
        amp = [c[3] for c in centers]
    else
        K = bank.K
        x = bank.x
        y = bank.y
        amp = bank.amp
        # Use σ from bank if available, otherwise use parameter
        σ = haskey(bank, :σ) ? bank.σ : σ
    end

    σ_sq = σ^2

    field = function (xi::Real, yi::Real)
        acc = 0.0
        @inbounds for m in 1:K
            dx = xi - x[m]
            dy = yi - y[m]
            r_sq = dx^2 + dy^2
            acc += amp[m] * exp(-r_sq / (2 * σ_sq))
        end
        return acc
    end

    return field
end

"""
    objective_grid_from_rbf_key(key, cfg::RBFDiscreteCfg, xs, ys;
                                 n_bins::Int=20,
                                 n_amps::Int=10)

Build objective scalar field from RBF key+cfg and evaluate on grid.
"""
function objective_grid_from_rbf_key(key, cfg::RBFDiscreteCfg, xs, ys;
                                      n_bins::Int=20,
                                      n_amps::Int=10)
    bank = decode_rbf_key(key, cfg; n_bins=n_bins, n_amps=n_amps)
    field = make_rbf_scalar_field(bank; σ=cfg.σ)
    return objective_grid_from_field(field, xs, ys)
end
