################
# Discretization
################

"""
    K_probs(cfg::FourierDiscreteCfg)

Constructs categorical vector mapping k-count to exponential decay distribution.
"""
function K_probs(cfg::FourierDiscreteCfg)
    ws = exp.(-cfg.λK .* (0:(cfg.Kmax-1)))
    ws ./= sum(ws)
    return ws
end

"""
    freq_bin_support_and_probs(cfg::FourierDiscreteCfg)

TODO: Constructs categorical vector mapping freq to exp decay??
"""
function freq_bin_support_and_probs(cfg::FourierDiscreteCfg)
    supp = collect(-cfg.Fmax_i:cfg.Fmax_i)
    if cfg.freq_mag_decay <= 0
        ws = fill(1.0, length(supp))
    else
        ws = exp.(-cfg.freq_mag_decay .* abs.(supp))
    end
    ws ./= sum(ws)
    return supp, ws
end

"""
    amp_bin_support_and_probs(cfg::FourierDiscreteCfg)

TODO: Constructs categorical vector mapping amplitude to uniform??
"""
function amp_bin_support_and_probs(cfg::FourierDiscreteCfg)
    supp = collect(0:cfg.Amax_i)
    ws = fill(1.0, length(supp))
    ws ./= sum(ws)
    return supp, ws
end

"""
    phase_bin_support_and_probs(cfg::FourierDiscreteCfg)

TODO: Constructs categorical vector mapping ϕ to uniform??
"""
function phase_bin_support_and_probs(cfg::FourierDiscreteCfg)
    supp = collect(0:(cfg.P-1))
    ws = fill(1.0, length(supp))
    ws ./= sum(ws)
    return supp, ws
end

################################
# Fourier Feature Key Operations
################################

"""
Internal: sample discrete Fourier indices with an override for K and with controllable supports.
Returns (key, ff_namedtuple, sweep_tag, sweep_level, cfg_used)
"""
function sample_fourier_key(cfg::FourierDiscreteCfg;
                            K_override::Union{Nothing,Int}=nothing,
                            rng=Random.default_rng())

    # Supports/probs
    Kp = K_probs(cfg)
    freq_supp, freq_w = freq_bin_support_and_probs(cfg)
    amp_supp, amp_w   = amp_bin_support_and_probs(cfg)

    K = isnothing(K_override) ? rand(rng, Categorical(Kp)) : K_override
    K = clamp(K, 1, cfg.Kmax)

    fx_idx = Vector{Int}(undef, K)
    fy_idx = Vector{Int}(undef, K)
    A_idx  = Vector{Int}(undef, K)
    ϕ_idx  = Vector{Int}(undef, K)

    for m in 1:K
        fx_idx[m] = freq_supp[randcat(rng, freq_w)]
        fy_idx[m] = freq_supp[randcat(rng, freq_w)]
        A_idx[m]  = amp_supp[randcat(rng, amp_w)]
        ϕ_idx[m]  = rand(rng, 0:cfg.P-1)
    end

    key = (K, fx_idx, fy_idx, A_idx, ϕ_idx)
    return key
end

"""
Decode Fourier key of the form (K, fx_i, fy_i, A_i, ϕ_i) into continuous params.
Assumes fx_i etc are integer vectors of length K (or length Kmax, if fixed-bank).
"""
function decode_fourier_key(key, cfg::FourierDiscreteCfg)
    K, fx_i, fy_i, A_i, ϕ_i = key
    # use only active prefix if vectors are longer
    fx = f_from_i.(fx_i[1:K], Ref(cfg))
    fy = f_from_i.(fy_i[1:K], Ref(cfg))
    A  = A_from_i.(A_i[1:K],  Ref(cfg))
    ϕ  = ϕ_from_i.(ϕ_i[1:K],  Ref(cfg))
    return (K=K, fx=fx, fy=fy, A=A, ϕ=ϕ, fx_i=fx_i[1:K], fy_i=fy_i[1:K], A_i=A_i[1:K], ϕ_i=ϕ_i[1:K])
end

"""
    hamming_fourier_key(k1, k2) -> Int

Hamming distance on Fourier *discrete* key representation.

Key format assumed:
    (K::Int, fx_i::Vector{Int}, fy_i::Vector{Int}, A_i::Vector{Int}, ϕ_i::Vector{Int})

Only compare the active prefixes (1:K), and add abs(K1-K2).
"""
function hamming_fourier_key(k1, k2)
    K1, fx1, fy1, A1, ϕ1 = k1
    K2, fx2, fy2, A2, ϕ2 = k2
    d = abs(K1 - K2)

    K = min(K1, K2)
    @inbounds for m in 1:K
        d += (fx1[m] != fx2[m])
        d += (fy1[m] != fy2[m])
        d += (A1[m]  != A2[m])
        d += (ϕ1[m]  != ϕ2[m])
    end

    # treat unmatched tail entries as mismatches
    if K1 != K2
        Kbig = max(K1, K2)
        d += 4 * (Kbig - K)  # each extra mode has 4 discrete indices
    end
    return d
end

"""
    nearest_trained_key(π_dist, key; min_trained=1)

Returns the closest key among those already in π_dist.n_π_proposals and
whose training steps record indicates ≥ min_trained.
Returns `nothing` if none exist.
"""
function nearest_trained_key(π_dist::ScoreΠDist, key; min_trained::Int=1)
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

        d = hamming_fourier_key(key, k)
        if d < best_d
            best = k
            best_d = d
        end
    end
    return best
end

##############################
# Arbitrary Field Construction
##############################

"""
    make_fourier_scalar_field(bank; normalize=true)

Returns:
- field(x::Real, y::Real)::Float64

Definition:
  field(x,y) = Σ_{m=1..K} A[m] * cos(fx[m]*x + fy[m]*y + ϕ[m])

If `scaleQ=true`, divides by max(1,K) so magnitude doesn't explode with K.
"""
function make_fourier_scalar_field(bank; scaleQ::Bool=true)
    K  = bank.K
    fx = bank.fx
    fy = bank.fy
    A  = bank.A
    ϕ  = bank.ϕ
    invK = scaleQ ? (1.0 / max(1, K)) : 1.0

    field = function (x::Real, y::Real)
        acc = 0.0
        @inbounds for m in 1:K
            acc += A[m] * cos(fx[m]*x + fy[m]*y + ϕ[m])
        end
        return invK * acc
    end

    return field
end

"""
    objective_grid_from_key(key, cfg, xs, ys)

Build objective scalar field from Fourier key+cfg and evaluate on grid.
"""
function objective_grid_from_key(key, cfg, xs, ys)
    bank = decode_fourier_key(key, cfg)
    field = make_fourier_scalar_field(bank; scaleQ=true)
    return objective_grid_from_field(field, xs, ys)
end