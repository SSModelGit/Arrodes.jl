# Hybrid Objective Architecture: Detailed Requirements & Design Scaffolding

**Document Version:** 1.0  
**Date:** March 18, 2026  
**Purpose:** Design specification for mixed-modal (Fourier + RBF) objective function architecture  
**Status:** Pre-implementation (architectural scaffold for future development sessions)

---

## Table of Contents

1. [Executive Overview](#executive-overview)
2. [Architecture Goals](#architecture-goals)
3. [Capability Requirements](#capability-requirements)
4. [Data Flow & Type System](#data-flow--type-system)
5. [Functional Specifications](#functional-specifications)
6. [Integration Points](#integration-points)
7. [Backward Compatibility](#backward-compatibility)
8. [Suggested Implementation Approach](#suggested-implementation-approach)

---

## Executive Overview

### Current State
The Arrodes.jl inference system currently supports **Fourier-only** objective functions:
- Keys encode as: `(K, fx_i, fy_i, A_i, ϕ_i)` 
- Sampling through: `gen_fourier_bank_fixed()` generative function
- Field evaluation: sum of cosines with phase and amplitude modulation
- Configuration: single `FourierDiscreteCfg` in `ScoreΠDist`

### Desired End State
A **mixed-modal** system where each objective function component can independently be:
- **Fourier feature:** `A·cos(fx·x + fy·y + ϕ)` (frequency domain representation)
- **RBF (Radial Basis Function):** `A·exp(-(r²/(2σ²)))` (spatial Gaussian basis)

Components are sampled from a **Bernoulli distribution** at inference time, allowing:
- Per-component type selection (Fourier XOR RBF for each mode)
- Unified field composition through superposition
- Flexible prior configuration (pure Fourier, pure RBF, or mixed)
- Backward-compatible operation with existing Fourier-only code

### Key Insight: Composability
The fundamental design principle is that objectives are **composable bags of components**, each independently:
- Encoded (via discrete indices)
- Decoded (to continuous parameters)
- Evaluated (to scalar contribution)
- Ranked (via distance metrics)

This enables:
- Runtime flexibility in component selection
- Extensibility to future modal types
- Clean separation of concerns in inference/visualization code

---

## Architecture Goals

### Primary Goals
1. **Modal Flexibility:** Support both Fourier and RBF components in the same objective
2. **Component Independence:** Each mode operates independently; no cross-contamination
3. **Unified Interface:** All objectives produce `(x, y) → Float64` scalar fields
4. **Extensibility:** Architecture supports future modal types (Wavelets, Polynomials, etc.)
5. **Inference Integrity:** Particle filter operates unchanged; posterior inference remains valid

### Secondary Goals
1. **Backward Compatibility:** Fourier-only code continues to work
2. **Performance:** No overhead for pure Fourier runs (monomorphic behavior when `mode=FOURIER`)
3. **Debuggability:** Component types are explicit; easy to diagnose mixed objectives
4. **Serialization:** Keys remain serializable for caching and offline analysis

### Non-Goals
- Real-time component adaptation during inference (fixed at sampling time)
- Automatic mode selection (explicit configuration)
- Cross-component parameter coupling (independent per component)

---

## Capability Requirements

### REQ-1: Configuration Management

**Requirement:** System must support configuration of three distinct operating modes.

**Specification:**

```julia
# Mode 1: Pure Fourier (current behavior)
cfg_fourier = FourierDiscreteCfg(Kmax=10, λK=0.35, ...)
π_dist_fourier = ScoreΠDist(fourier_cfg=cfg_fourier, prior_mode=FOURIER)

# Mode 2: Pure RBF (new)
cfg_rbf = RBFDiscreteCfg(Kmax=5, λK=0.5, σ=1.0, ...)
π_dist_rbf = ScoreΠDist(rbf_cfg=cfg_rbf, prior_mode=RBF)

# Mode 3: Hybrid (new)
cfg_hybrid = HybridCfg(
    fourier_cfg = FourierDiscreteCfg(...),
    rbf_cfg = RBFDiscreteCfg(...),
    p_fourier = 0.5,  # Bernoulli parameter for component type
    mode = HYBRID
)
π_dist_hybrid = ScoreΠDist(hybrid_cfg=cfg_hybrid, prior_mode=HYBRID)
```

**Concrete Capabilities:**
- [x] `HybridCfg` struct bundles both `FourierDiscreteCfg` and `RBFDiscreteCfg`
- [x] `ComponentMode` enum: `FOURIER | RBF | HYBRID` for runtime dispatch
- [x] `ScoreΠDist` accepts `prior_mode::ComponentMode` field
- [x] Each mode has independent configuration parameters
- [x] Bernoulli probability `p_fourier` controls per-component type selection in hybrid mode

---

### REQ-2: Key Encoding & Decoding

**Requirement:** System must encode/decode objective parameters in a mode-agnostic way.

**Specification:**

#### Current Fourier Key:
```julia
fourier_key = (K, fx_i, fy_i, A_i, ϕ_i)
# where:
#   K::Int - number of active components
#   fx_i::Vector{Int} - frequency indices (x-component)
#   fy_i::Vector{Int} - frequency indices (y-component)
#   A_i::Vector{Int} - amplitude indices
#   ϕ_i::Vector{Int} - phase indices
```

#### New Hybrid Key Format:
```julia
hybrid_key = (K, component_types, components)
# where:
#   K::Int - number of active components
#   component_types::Vector{ComponentMode} - type of each component [FOURIER | RBF]
#   components::Vector{Union{FourierComponent, RBFComponent}} - per-component data

# Component structures:
struct FourierComponent
    fx_i::Int
    fy_i::Int
    A_i::Int
    ϕ_i::Int
end

struct RBFComponent
    x_idx::Int
    y_idx::Int
    amp_idx::Int
end
```

**Concrete Capabilities:**
- [x] Fourier keys remain unchanged (backward compatible)
- [x] Hybrid keys explicitly track component types in `component_types` vector
- [x] Component data is encapsulated in component-specific structs
- [x] `decode_fourier_key(key, cfg)` returns `(K, fx, fy, A, ϕ)` continuous parameters
- [x] `decode_rbf_key(key, cfg)` returns `(K, x, y, amp)` continuous parameters
- [x] `decode_hybrid_key(key, cfg)` unpacks to mixed continuous parameters
- [x] Conversion functions: `to_hybrid_key(fourier_key, cfg)` and `from_hybrid_key(hybrid_key, cfg)`

---

### REQ-3: Generative Model (Sampling)

**Requirement:** Inference model must sample objectives according to configured mode.

**Specification:**

#### Mode-Specific Sampling:

**Fourier Mode:**
```julia
@gen function gen_fourier_bank_fixed(cfg::FourierDiscreteCfg)
    K ~ categorical(K_probs(cfg))
    # ... sample K Fourier components
    # each: (fx_i, fy_i, A_i, ϕ_i)
    return (key=(K, fx_i, fy_i, A_i, ϕ_i), ...)
end
```

**RBF Mode (new):**
```julia
@gen function gen_rbf_bank_fixed(cfg::RBFDiscreteCfg)
    K ~ categorical(K_probs(cfg))
    # ... sample K RBF components
    # each: (x_idx, y_idx, amp_idx)
    return (key=(K, x_idx, y_idx, amp_idx), ...)
end
```

**Hybrid Mode (new):**
```julia
@gen function gen_component_modes(K::Int, cfg::HybridCfg)
    modes = Vector{ComponentMode}(undef, K)
    for m in 1:K
        modes[m] ~ bernoulli(cfg.p_fourier) ? FOURIER : RBF
    end
    return modes
end

@gen function gen_hybrid_components(K::Int, modes::Vector{ComponentMode}, cfg::HybridCfg)
    components = Vector{Union{FourierComponent, RBFComponent}}(undef, K)
    for m in 1:K
        if modes[m] == FOURIER
            # Sample Fourier: (fx_i, fy_i, A_i, ϕ_i)
            components[m] = FourierComponent(...)
        else  # RBF
            # Sample RBF: (x_idx, y_idx, amp_idx)
            components[m] = RBFComponent(...)
        end
    end
    return components
end

@gen function gen_hybrid_bank_fixed(cfg::HybridCfg)
    K ~ categorical(K_probs(cfg.fourier_cfg))  # Use Fourier K distribution
    modes ~ gen_component_modes(K, cfg)
    components ~ gen_hybrid_components(K, modes, cfg)
    key = (K, modes, components)
    return (key=key, K=K, modes=modes, components=components)
end
```

**Dispatcher (in inference_model):**
```julia
@gen function inference_model(N::Int, π_dist::ScoreΠDist, agent_params::Dict, state_data::Matrix)
    if π_dist.prior_mode == FOURIER
        bank ~ gen_fourier_bank_fixed(π_dist.fourier_cfg)
    elseif π_dist.prior_mode == RBF
        bank ~ gen_rbf_bank_fixed(π_dist.rbf_cfg)
    else  # HYBRID
        bank ~ gen_hybrid_bank_fixed(π_dist.hybrid_cfg)
    end
    # ... rest unchanged
end
```

**Concrete Capabilities:**
- [x] Each mode has independent sampling function
- [x] Hybrid mode samples K from Fourier distribution (arbitrary choice, could use RBF)
- [x] Hybrid mode samples per-component type via Bernoulli
- [x] Hybrid mode samples component parameters according to sampled type
- [x] Returned bank contains: `key`, `K`, `modes`, `components` (hybrid-specific fields optional)
- [x] `inference_model` dispatches based on `π_dist.prior_mode`
- [x] All samplers integrate with Gen.jl's trace infrastructure

---

### REQ-4: Field Construction & Evaluation

**Requirement:** System must construct scalar fields from any objective key and evaluate them on (x,y) points.

**Specification:**

#### Fourier Field:
```julia
function make_fourier_scalar_field(bank; scaleQ::Bool=true)
    K, fx, fy, A, ϕ = bank  # or named tuple
    invK = scaleQ ? (1.0 / max(1, K)) : 1.0
    return function (x::Real, y::Real)
        acc = 0.0
        for m in 1:K
            acc += A[m] * cos(fx[m]*x + fy[m]*y + ϕ[m])
        end
        return invK * acc
    end
end
```

#### RBF Field (new):
```julia
function make_rbf_scalar_field(bank; σ::Float64=1.0, scaleQ::Bool=true)
    K, x_centers, y_centers, amplitudes = bank  # or named tuple
    invK = scaleQ ? (1.0 / max(1, K)) : 1.0
    return function (x::Real, y::Real)
        acc = 0.0
        for m in 1:K
            r_sq = (x - x_centers[m])^2 + (y - y_centers[m])^2
            acc += amplitudes[m] * exp(-r_sq / (2 * σ^2))
        end
        return invK * acc
    end
end
```

#### Hybrid Field (new):
```julia
function make_hybrid_scalar_field(bank; cfg::HybridCfg, scaleQ::Bool=true)
    K, modes, components = bank
    invK = scaleQ ? (1.0 / max(1, K)) : 1.0
    
    return function (x::Real, y::Real)
        acc = 0.0
        for m in 1:K
            if modes[m] == FOURIER
                comp = components[m]
                fx = Priors.f_from_i(comp.fx_i, cfg.fourier_cfg)
                fy = Priors.f_from_i(comp.fy_i, cfg.fourier_cfg)
                A = Priors.A_from_i(comp.A_i, cfg.fourier_cfg)
                ϕ = Priors.ϕ_from_i(comp.ϕ_i, cfg.fourier_cfg)
                acc += A * cos(fx*x + fy*y + ϕ)
            else  # RBF
                comp = components[m]
                x_c = Priors.x_from_i(comp.x_idx, cfg.rbf_cfg)
                y_c = Priors.y_from_i(comp.y_idx, cfg.rbf_cfg)
                A = Priors.A_from_i(comp.amp_idx, cfg.rbf_cfg)
                r_sq = (x - x_c)^2 + (y - y_c)^2
                acc += A * exp(-r_sq / (2 * cfg.rbf_cfg.σ^2))
            end
        end
        return invK * acc
    end
end
```

**Concrete Capabilities:**
- [x] Each mode produces `(x::Real, y::Real) → Float64` closure
- [x] All fields support optional normalization via `scaleQ` parameter
- [x] Hybrid field iterates through components, dispatching per-type logic
- [x] Field evaluation is **deterministic** and **differentiable** (no randomness)
- [x] Fields compose via superposition: all components sum into single value
- [x] Grid evaluation: `objective_grid_from_field(field, xs, ys)` works for any field type

---

### REQ-5: Distance Metrics for Key Similarity

**Requirement:** System must compute distance between keys for nearest-neighbor lookups.

**Specification:**

#### Fourier Distance:
```julia
function hamming_fourier_key(k1, k2)
    K1, fx1, fy1, A1, ϕ1 = k1
    K2, fx2, fy2, A2, ϕ2 = k2
    
    d = abs(K1 - K2) * 10  # Heavy penalty for K mismatch
    K = min(K1, K2)
    for m in 1:K
        d += (fx1[m] != fx2[m]) + (fy1[m] != fy2[m])
        d += (A1[m] != A2[m]) + (ϕ1[m] != ϕ2[m])
    end
    return d
end
```

#### RBF Distance (new):
```julia
function hamming_rbf_key(k1, k2)
    K1, x1, y1, a1 = k1
    K2, x2, y2, a2 = k2
    
    d = abs(K1 - K2) * 10
    K = min(K1, K2)
    for m in 1:K
        d += (x1[m] != x2[m]) + (y1[m] != y2[m]) + (a1[m] != a2[m])
    end
    return d
end
```

#### Hybrid Distance (new):
```julia
function hamming_hybrid_key(k1, k2)
    K1, modes1, comps1 = k1
    K2, modes2, comps2 = k2
    
    d = abs(K1 - K2) * 10  # K mismatch penalty
    K = min(K1, K2)
    
    for m in 1:K
        # Component type mismatch costs high
        d += (modes1[m] != modes2[m]) ? 5 : 0
        
        # Within-type parameter distance
        if modes1[m] == FOURIER && modes2[m] == FOURIER
            c1, c2 = comps1[m], comps2[m]
            d += (c1.fx_i != c2.fx_i) + (c1.fy_i != c2.fy_i)
            d += (c1.A_i != c2.A_i) + (c1.ϕ_i != c2.ϕ_i)
        elseif modes1[m] == RBF && modes2[m] == RBF
            c1, c2 = comps1[m], comps2[m]
            d += (c1.x_idx != c2.x_idx) + (c1.y_idx != c2.y_idx)
            d += (c1.amp_idx != c2.amp_idx)
        end
    end
    return d
end
```

**Concrete Capabilities:**
- [x] Each mode has dedicated distance function
- [x] K mismatches heavily penalized (encourages same-K matches)
- [x] Component type mismatches in hybrid have high cost (5 units vs 1 for parameter)
- [x] Used by `nearest_trained_key()` for policy refinement
- [x] Integer-valued output for discrete comparison
- [x] Symmetric: `d(k1, k2) = d(k2, k1)` (for consistency)

---

### REQ-6: MDP Caching & Proposal Management

**Requirement:** System must cache and manage MDPs/policies per objective key.

**Specification:**

#### Current Flow:
```julia
function ensure_mdp!(π_dist::ScoreΠDist, key, bank, agent_params::Dict)
    get!(π_dist.n_propmdp_list, key) do
        field = make_fourier_scalar_field(bank; scaleQ=true)
        obj = make_pomdp_objective_from_field(field)
        mdp = build_kagent_pomdp(agent_params, obj)
        return mdp
    end
end
```

#### New Mode-Dispatched Flow:
```julia
function ensure_mdp!(π_dist::ScoreΠDist, key, bank, agent_params::Dict)
    get!(π_dist.n_propmdp_list, key) do
        if π_dist.prior_mode == FOURIER
            field = make_fourier_scalar_field(bank; scaleQ=true)
        elseif π_dist.prior_mode == RBF
            field = make_rbf_scalar_field(bank; σ=π_dist.rbf_cfg.σ)
        else  # HYBRID
            field = make_hybrid_scalar_field(bank; cfg=π_dist.hybrid_cfg, scaleQ=true)
        end
        obj = make_pomdp_objective_from_field(field)
        mdp = build_kagent_pomdp(agent_params, obj)
        return mdp
    end
end
```

**Concrete Capabilities:**
- [x] MDP lazy-loaded on first request for key
- [x] Dispatch based on `π_dist.prior_mode`
- [x] Field type automatically selected from prior_mode
- [x] Cache remains agnostic to field type (stores MDPs, not fields)
- [x] Policy proposals cached separately via `get_π_proposal(π_dist, key)`
- [x] Training step tracking: `π_dist.n_𝒮_proposals[key]` records training iterations

---

### REQ-7: Particle Filter Integration

**Requirement:** Particle filter must support inference under any mode without logic changes.

**Specification:**

#### Current Trace Structure (Fourier):
```julia
# Choices are traced at paths like:
(:fourier, :K)                      # total component count
(:fourier, :mode, m) => :fx_idx     # for component m, index choices
(:fourier, :mode, m) => :fy_idx
(:fourier, :mode, m) => :A_idx
(:fourier, :mode, m) => :ϕ_idx
```

#### New Mode-Aware Trace Paths:
```julia
# FOURIER mode: unchanged
(:fourier, :K)
(:fourier, :mode, m) => :fx_idx

# RBF mode: new path structure
(:rbf, :K)
(:rbf, :mode, m) => :x_idx
(:rbf, :mode, m) => :y_idx
(:rbf, :mode, m) => :amp_idx

# HYBRID mode: mixed paths
(:hybrid, :K)
(:hybrid, :modes, m) => :mode       # which type is component m?
# Then conditionally:
(:hybrid, :components, m, :fourier) => :fx_idx  # if FOURIER
(:hybrid, :components, m, :rbf) => :x_idx       # if RBF
```

#### Particle Filter Rejuvenation:
```julia
function particle_filter(observations::Vector{Int}, π_dist::ScoreΠDist, ...)
    # ... setup ...
    
    for n in 2:N
        if should_resample(state)
            pf_resample!(state)
            
            # Build selection based on prior_mode
            sels = Any[]
            if π_dist.prior_mode == FOURIER
                push!(sels, :fourier => :K)
                for m in 1:M
                    push!(sels, (:fourier, :mode, m) => :fx_idx)
                    # ... other indices
                end
            elseif π_dist.prior_mode == RBF
                push!(sels, :rbf => :K)
                for m in 1:M
                    push!(sels, (:rbf, :mode, m) => :x_idx)
                    # ... other indices
                end
            else  # HYBRID
                push!(sels, :hybrid => :K)
                # Note: mode choices are deterministic after sampling,
                # can only rejuvenate component parameters
                for m in 1:M
                    # Check trace to see which type this component is
                    # then add appropriate selectors
                end
            end
            
            pf_rejuvenate!(state, mh, (select(sels...),))
            RL.maybe_refine_policies!(π_dist, state, agent_params; topk=5)
        end
        pf_update!(state, (n, π_dist, agent_params, state_data), obs_choices[n], n_particles)
    end
    
    return state
end
```

**Concrete Capabilities:**
- [x] Trace paths prefixed by mode identifier (`:fourier`, `:rbf`, `:hybrid`)
- [x] Rejuvenation selectors adapted to mode-specific trace structure
- [x] No changes to core particle filter algorithm (initialization, update, resample)
- [x] Policy refinement queries remain unchanged (uses `top_objectives()`)
- [x] Inference produces same posterior interface regardless of mode

---

### REQ-8: Visualization & Analysis

**Requirement:** All visualization functions must work with any objective mode.

**Specification:**

#### Key Decoding Dispatcher:
```julia
function decode_key(key, π_dist::ScoreΠDist)
    """Dispatch to appropriate decoder based on prior_mode"""
    if π_dist.prior_mode == FOURIER
        return decode_fourier_key(key, π_dist.fourier_cfg)
    elseif π_dist.prior_mode == RBF
        return decode_rbf_key(key, π_dist.rbf_cfg)
    else  # HYBRID
        return decode_hybrid_key(key, π_dist.hybrid_cfg)
    end
end
```

#### Plotting Functions Updated:
```julia
# OLD (hardcoded Fourier):
function plot_top_objective_with_trajectories(pf_state, π_dist::ScoreΠDist, ...)
    key, prob = RL.top_key(pf_state, π_dist)
    ff = Priors.decode_fourier_key(key, π_dist.fourier_cfg)  # HARDCODED
    field = Priors.make_fourier_scalar_field(ff; scaleQ=true)  # HARDCODED
    # ...
end

# NEW (dispatched):
function plot_top_objective_with_trajectories(pf_state, π_dist::ScoreΠDist, ...)
    key, prob = RL.top_key(pf_state, π_dist)
    bank = decode_key(key, π_dist)  # DISPATCHED
    field = make_field(bank, π_dist)  # DISPATCHED (new helper)
    # ...
end

function make_field(bank, π_dist::ScoreΠDist)
    """Construct field from decoded bank based on prior_mode"""
    if π_dist.prior_mode == FOURIER
        return make_fourier_scalar_field(bank; scaleQ=true)
    elseif π_dist.prior_mode == RBF
        return make_rbf_scalar_field(bank; σ=π_dist.rbf_cfg.σ)
    else  # HYBRID
        return make_hybrid_scalar_field(bank; cfg=π_dist.hybrid_cfg, scaleQ=true)
    end
end
```

**Concrete Capabilities:**
- [x] All plotting functions use `decode_key()` dispatcher
- [x] All field construction uses `make_field()` dispatcher
- [x] Heatmaps, trajectories, comparisons work for any mode
- [x] Barplots and metric visualizations unchanged (work on keys/posteriors only)
- [x] Export functions updated to include RBF/Hybrid visualization support

---

### REQ-9: Ablation & Analysis Pipeline

**Requirement:** Ablation study pipeline must support all modes for systematic evaluation.

**Specification:**

#### Ablation Objective Generation:
```julia
function build_ablation_objectives_hybrid(;
    rng = Random.default_rng(),
    base_cfg::HybridCfg = HybridCfg(),
    levels::Int = 10,
    sweeps::Vector{Symbol} = [:K, :p_fourier, :fourier_freq, :rbf_sigma]
)
    """
    Generate ablation objectives varying:
    - K: number of components (1..Kmax)
    - p_fourier: Bernoulli parameter (0.0..1.0)
    - fourier_freq: frequency range (Fmax_i)
    - rbf_sigma: RBF width (σ)
    """
    out = NamedTuple[]
    
    # Sweep 1: K (total components)
    for k in 1:base_cfg.fourier_cfg.Kmax
        cfg = deepcopy(base_cfg)
        key = Priors.sample_hybrid_key(cfg; K_override=k, rng=rng)
        bank = Priors.decode_hybrid_key(key, cfg)
        field = Priors.make_hybrid_scalar_field(bank; cfg=cfg, scaleQ=true)
        obj = Priors.make_pomdp_objective_from_field(field)
        push!(out, (sweep=:K, level=k, cfg=cfg, key=key, field=field, obj=obj))
    end
    
    # Sweep 2: p_fourier (component type distribution)
    p_values = range(0.0, 1.0; length=levels)
    for p in p_values
        cfg = deepcopy(base_cfg)
        cfg.p_fourier = p
        key = Priors.sample_hybrid_key(cfg; rng=rng)
        bank = Priors.decode_hybrid_key(key, cfg)
        field = Priors.make_hybrid_scalar_field(bank; cfg=cfg, scaleQ=true)
        obj = Priors.make_pomdp_objective_from_field(field)
        push!(out, (sweep=:p_fourier, level=p, cfg=cfg, key=key, field=field, obj=obj))
    end
    
    # Sweep 3+: mode-specific parameters...
    
    return out
end
```

**Concrete Capabilities:**
- [x] Ablation objectives generatable for any prior_mode
- [x] Each sweep varies one parameter; others fixed
- [x] K sweep varies total component count
- [x] p_fourier sweep explores Fourier vs RBF balance
- [x] Mode-specific sweeps (Fmax_i, σ, etc.) per configuration
- [x] Returned objectives stored with cfg, key, field for analysis

---

### REQ-10: Backward Compatibility

**Requirement:** Existing Fourier-only code must work unchanged.

**Specification:**

#### Existing Code Patterns (Must Still Work):
```julia
# Pattern 1: ScoreΠDist with explicit fourier_cfg
cfg = FourierDiscreteCfg(Kmax=10)
π_dist = ScoreΠDist(fourier_cfg=cfg)  # Should default prior_mode=FOURIER

# Pattern 2: Key sampling and decoding
key = sample_fourier_key(cfg; K_override=3, rng=rng)
bank = decode_fourier_key(key, cfg)

# Pattern 3: Field construction
field = make_fourier_scalar_field(bank; scaleQ=true)

# Pattern 4: Visualization
plot_top_objective_with_trajectories(pf_state, π_dist, ...)
```

#### Backward Compatibility Mechanisms:
```julia
# DEFAULT: prior_mode should default to FOURIER
@with_kw struct ScoreΠDist
    fourier_cfg::FourierDiscreteCfg = FourierDiscreteCfg()
    prior_mode::ComponentMode = FOURIER  # DEFAULT to Fourier
    # ... other fields
end

# Fourier-specific functions remain unchanged
sample_fourier_key(cfg; K_override=nothing, rng=default_rng())  # unchanged
decode_fourier_key(key, cfg)  # unchanged
make_fourier_scalar_field(bank; scaleQ=true)  # unchanged

# Visualization functions dispatch:
# If π_dist.prior_mode == FOURIER, paths through original code
# If π_dist.prior_mode == RBF or HYBRID, paths through new code
```

**Concrete Capabilities:**
- [x] Default `prior_mode=FOURIER` preserves existing behavior
- [x] No changes required to existing scripts using Fourier
- [x] Fourier-specific functions remain 100% unchanged
- [x] Visualization functions detect Fourier mode and use original paths (zero overhead)
- [x] New users can explicitly opt into RBF/HYBRID modes

---

## Data Flow & Type System

### Type Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                    PriorDiscreteCfg (abstract)                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
├─ FourierDiscreteCfg                                             │
│  └─ Kmax, λK, Δf, Fmax_i, ΔA, Amax_i, P, freq_mag_decay       │
│                                                                 │
├─ RBFDiscreteCfg                                                 │
│  └─ Kmax, λK, σ, x_min/max, y_min/max                          │
│                                                                 │
└─ HybridCfg                                                      │
   └─ fourier_cfg, rbf_cfg, p_fourier, mode                       │
```

### Component Structures

```
┌─────────────────────────────────────────────────────────────────┐
│             Union{FourierComponent, RBFComponent}               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
├─ FourierComponent                                               │
│  └─ fx_i::Int, fy_i::Int, A_i::Int, ϕ_i::Int                  │
│                                                                 │
└─ RBFComponent                                                   │
   └─ x_idx::Int, y_idx::Int, amp_idx::Int                       │
```

### Key Structures

```
Fourier Key:          (K::Int, fx_i::Vec{Int}, fy_i::Vec{Int}, A_i::Vec{Int}, ϕ_i::Vec{Int})
RBF Key:              (K::Int, x_idx::Vec{Int}, y_idx::Vec{Int}, amp_idx::Vec{Int})
Hybrid Key:           (K::Int, modes::Vec{ComponentMode}, components::Vec{Union{...}})
```

### Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         INFERENCE PIPELINE                                   │
└──────────────────────────────────────────────────────────────────────────────┘

1. CONFIGURATION
   ├─ FourierDiscreteCfg / RBFDiscreteCfg / HybridCfg
   └─ ScoreΠDist(fourier_cfg, rbf_cfg, prior_mode, hybrid_cfg)

2. SAMPLING (Gen.jl)
   ├─ π_dist.prior_mode == FOURIER
   │  └─ gen_fourier_bank_fixed(fourier_cfg) → (K, fx_i, fy_i, A_i, ϕ_i)
   ├─ π_dist.prior_mode == RBF
   │  └─ gen_rbf_bank_fixed(rbf_cfg) → (K, x_idx, y_idx, amp_idx)
   └─ π_dist.prior_mode == HYBRID
      └─ gen_hybrid_bank_fixed(hybrid_cfg)
         ├─ gen_K(fourier_cfg.λK) → K
         ├─ gen_component_modes(K, p_fourier) → modes::Vec{ComponentMode}
         ├─ gen_hybrid_components(K, modes) → components
         └─ (K, modes, components)

3. KEY REGISTRATION
   └─ register_key_if_new!(π_dist, key)

4. MDP CONSTRUCTION
   └─ ensure_mdp!(π_dist, key, bank)
      ├─ detect π_dist.prior_mode
      ├─ decode_key(key, π_dist) → continuous bank
      ├─ make_field(bank, π_dist) → (x,y) → Float64
      ├─ make_pomdp_objective_from_field(field) → obj function
      └─ build_kagent_pomdp(agent_params, obj) → mdp

5. POLICY INFERENCE & CACHING
   ├─ get_𝒮_proposal(π_dist, key) → solver
   └─ get_π_proposal(π_dist, key) → policy

6. PARTICLE FILTERING
   ├─ pf_initialize/update/resample/rejuvenate
   └─ Trace paths adapt to prior_mode ((:fourier|:rbf|:hybrid, ...))

7. POSTERIOR ANALYSIS
   ├─ top_objectives(pf_state, π_dist) → sorted list of (key, prob, ...)
   └─ Posterior agnostic to mode (works with any key format)

8. VISUALIZATION / ANALYSIS
   ├─ decode_key(top_key, π_dist) → dispatches to correct decoder
   ├─ make_field(decoded_bank, π_dist) → dispatches to correct field constructor
   └─ plot/evaluate on grid → uniform interface
```

---

## Functional Specifications

### Function Signatures (New Required Functions)

#### Type Definitions (types.jl)
```julia
@enum ComponentMode begin
    FOURIER = 1
    RBF = 2
    HYBRID = 3
end

@with_kw struct HybridCfg <: PriorDiscreteCfg
    fourier_cfg::FourierDiscreteCfg = FourierDiscreteCfg()
    rbf_cfg::RBFDiscreteCfg = RBFDiscreteCfg()
    p_fourier::Float64 = 0.5
    mode::ComponentMode = HYBRID
end

@with_kw struct FourierComponent
    fx_i::Int
    fy_i::Int
    A_i::Int
    ϕ_i::Int
end

@with_kw struct RBFComponent
    x_idx::Int
    y_idx::Int
    amp_idx::Int
end

# Extended ScoreΠDist
@with_kw struct ScoreΠDist
    # ... existing fields ...
    fourier_cfg::FourierDiscreteCfg = FourierDiscreteCfg()
    rbf_cfg::RBFDiscreteCfg = RBFDiscreteCfg()
    prior_mode::ComponentMode = FOURIER
    hybrid_cfg::HybridCfg = HybridCfg()
end
```

#### Hybrid Priors Module (priors/hybrid.jl)
```julia
# Sampling
sample_hybrid_key(cfg::HybridCfg; K_override=nothing, rng=default_rng())::Tuple
decode_hybrid_key(key::Tuple, cfg::HybridCfg)::NamedTuple
hamming_hybrid_key(k1::Tuple, k2::Tuple)::Int

# Field Construction
make_hybrid_scalar_field(bank::NamedTuple; cfg::HybridCfg, scaleQ::Bool=true)::Function
objective_grid_from_hybrid_key(key::Tuple, cfg::HybridCfg, xs, ys)::Matrix

# Conversion (backward compatibility)
to_hybrid_key(fourier_key::Tuple, cfg::FourierDiscreteCfg)::Tuple
from_hybrid_key(hybrid_key::Tuple, cfg::HybridCfg)::Union{Tuple, Nothing}
```

#### Inference Model Extensions (inference/gen_model.jl)
```julia
@gen function gen_rbf_bank_fixed(cfg::RBFDiscreteCfg)::NamedTuple
@gen function gen_component_modes(K::Int, cfg::HybridCfg)::Vector{ComponentMode}
@gen function gen_hybrid_components(K::Int, modes::Vector{ComponentMode}, cfg::HybridCfg)::Vector
@gen function gen_hybrid_bank_fixed(cfg::HybridCfg)::NamedTuple

# Updated dispatcher
# (inference_model modified to dispatch based on π_dist.prior_mode)
```

#### Visualization Dispatchers (viz/objectives.jl)
```julia
decode_key(key::Tuple, π_dist::ScoreΠDist)::NamedTuple
make_field(bank::NamedTuple, π_dist::ScoreΠDist)::Function

# Updated existing functions (minimal changes):
plot_top_objective_with_trajectories(pf_state, π_dist::ScoreΠDist, ...)  # use dispatchers
plot_objective_side_by_side(pf_state, π_dist::ScoreΠDist, ...)  # use dispatchers
# ... etc for all viz functions
```

#### RL Module Extensions (rl/scoredist.jl)
```julia
# Updated (mode-aware):
ensure_mdp!(π_dist::ScoreΠDist, key::Tuple, bank::NamedTuple, agent_params::Dict)::MDP

# Updated (mode-aware):
nearest_trained_key(π_dist::ScoreΠDist, key::Tuple; min_trained::Int=1)::Union{Tuple, Nothing}
```

#### Particle Filter (inference/particle_filter.jl)
```julia
# Updated to dispatch trace paths based on π_dist.prior_mode
# Core algorithm unchanged; only trace path prefixes adapt
```

#### Ablation Studies (analysis/ablations.jl)
```julia
build_ablation_objectives_hybrid(;
    rng=default_rng(),
    base_cfg::HybridCfg=HybridCfg(),
    levels::Int=10
)::Vector{NamedTuple}
```

---

## Integration Points

### 1. Type System Integration
- **File:** `src/types.jl`
- **Changes:** Add `ComponentMode`, `HybridCfg`, `FourierComponent`, `RBFComponent` types
- **Impact:** Exported from `Arrodes.jl`; available to all modules

### 2. Priors Module Integration
- **File:** `src/priors/Priors.jl`, `src/priors/hybrid.jl` (new)
- **Changes:** Include hybrid.jl; export all hybrid functions
- **Dependencies:** Uses `FourierDiscreteCfg`, `RBFDiscreteCfg`, `HybridCfg`, existing Fourier/RBF functions

### 3. Inference Module Integration
- **File:** `src/inference/gen_model.jl`
- **Changes:** Add `gen_rbf_bank_fixed`, `gen_component_modes`, `gen_hybrid_components`, `gen_hybrid_bank_fixed`; update `inference_model` dispatcher
- **Dependencies:** Uses `gen_K`, Priors module, Gen.jl framework

### 4. RL Module Integration
- **File:** `src/rl/scoredist.jl`
- **Changes:** Update `ensure_mdp!`, `nearest_trained_key` to dispatch based on prior_mode
- **Dependencies:** Uses new `make_field` dispatcher, field construction functions

### 5. Visualization Integration
- **File:** `src/viz/objectives.jl`
- **Changes:** Add `decode_key`, `make_field` dispatchers; update all plotting functions to use dispatchers
- **Dependencies:** Uses new decoder/field constructor, existing Plots.jl

### 6. Particle Filter Integration
- **File:** `src/inference/particle_filter.jl`
- **Changes:** Adapt trace path construction based on `π_dist.prior_mode`
- **Dependencies:** Uses Gen.jl, GenParticleFilters.jl

### 7. Visualization Module Export
- **File:** `src/viz/Visualizations.jl`
- **Changes:** Re-export new dispatcher functions from objectives.jl
- **Dependencies:** Already imports objectives.jl

### 8. Analysis Module Integration
- **File:** `src/analysis/ablations.jl`
- **Changes:** Add `build_ablation_objectives_hybrid` function
- **Dependencies:** Uses Priors module, existing ablation infrastructure

### 9. Main Package Exports
- **File:** `src/Arrodes.jl`
- **Changes:** Export new types (`ComponentMode`, `HybridCfg`, component types) and new functions
- **Impact:** All new functionality accessible from `using Arrodes`

---

## Backward Compatibility

### Guarantee: Zero Breaking Changes to Fourier-Only Code

#### Principle 1: Default to FOURIER Mode
```julia
# Old code continues to work because:
cfg = FourierDiscreteCfg(...)
π_dist = ScoreΠDist(fourier_cfg=cfg)  # prior_mode defaults to FOURIER

# All downstream code sees prior_mode == FOURIER and routes through original paths
```

#### Principle 2: Fourier Functions Unchanged
```julia
# These functions are 100% identical:
sample_fourier_key(cfg; K_override=nothing, rng=default_rng())
decode_fourier_key(key, cfg)
make_fourier_scalar_field(bank; scaleQ=true)
hamming_fourier_key(k1, k2)

# No changes, no performance overhead
```

#### Principle 3: Dispatch on Mode Prevents Fragmentation
```julia
# Visualization functions dispatch:
if π_dist.prior_mode == FOURIER
    # Execute original optimized path (compiled as before)
    bank = decode_fourier_key(key, π_dist.fourier_cfg)
    field = make_fourier_scalar_field(bank; scaleQ=true)
else
    # Execute new dispatcher
    bank = decode_key(key, π_dist)
    field = make_field(bank, π_dist)
end

# Fourier users see no performance difference (monomorphic)
```

#### Principle 4: Key Format Stability (Fourier)
```julia
# Fourier keys never change:
# Old: (K, fx_i::Vec{Int}, fy_i::Vec{Int}, A_i::Vec{Int}, ϕ_i::Vec{Int})
# New: identical for FOURIER mode

# Code serializing/deserializing Fourier keys unaffected
```

#### Principle 5: Particle Filter Trace Structure Stability (Fourier)
```julia
# Fourier trace paths unchanged:
(:fourier, :K)
(:fourier, :mode, m) => :fx_idx
# ... etc

# Code hardcoding trace paths for Fourier unaffected
```

### Migration Path for New Code (to Hybrid)

#### Step 1: Use Explicit Mode
```julia
# Old: implicit FOURIER
cfg = FourierDiscreteCfg(...)
π_dist = ScoreΠDist(fourier_cfg=cfg)

# New: explicit, same behavior
cfg = FourierDiscreteCfg(...)
π_dist = ScoreΠDist(fourier_cfg=cfg, prior_mode=FOURIER)
```

#### Step 2: Switch to RBF
```julia
# Pure RBF mode
cfg_rbf = RBFDiscreteCfg(...)
π_dist = ScoreΠDist(rbf_cfg=cfg_rbf, prior_mode=RBF)

# Inference, visualization, analysis: identical interface
```

#### Step 3: Enable Hybrid
```julia
# Mixed objectives
cfg_hybrid = HybridCfg(
    fourier_cfg=FourierDiscreteCfg(...),
    rbf_cfg=RBFDiscreteCfg(...),
    p_fourier=0.5
)
π_dist = ScoreΠDist(hybrid_cfg=cfg_hybrid, prior_mode=HYBRID)

# All existing analysis code works unchanged
```

---

## Suggested Implementation Approach

### Phase 1: Foundation (Type System & RBF Completion)

**Objective:** Establish type hierarchy and complete RBF implementation for reuse

**Tasks:**
1. **Add types in `src/types.jl`:**
   - Add `ComponentMode` enum
   - Add `HybridCfg` struct
   - Add `FourierComponent` and `RBFComponent` structs
   - Export all new types

2. **Complete RBF discretization in `src/priors/rbf.jl`:**
   - Fix `N_probs` → ensure `K_probs` works for RBF
   - Verify `sample_rbf_key()`, `decode_rbf_key()` functions (already exist but need testing)
   - Add `x_from_i()`, `y_from_i()` converter functions (currently missing)
   - Ensure all parameter converters work symmetrically

3. **Test RBF module:**
   - `sample_rbf_key()` → `decode_rbf_key()` round-trip
   - Grid evaluation: `objective_grid_from_rbf_key()` produces correct heatmaps
   - Distance metric: `hamming_rbf_key()` behaves correctly

**Files Modified:**
- `src/types.jl` (additions only)
- `src/priors/rbf.jl` (verify/complete converters)
- `test/test_fields.jl` (add RBF round-trip test)

**Backward Compatibility:** ✅ No changes to existing code paths

---

### Phase 2: Hybrid Priors Module

**Objective:** Implement hybrid objective sampling and field construction

**Tasks:**
1. **Create `src/priors/hybrid.jl`:**
   - `sample_hybrid_key()` - samples K, component types, component parameters
   - `decode_hybrid_key()` - converts indices to continuous bank
   - `hamming_hybrid_key()` - distance metric for hybrid keys
   - `make_hybrid_scalar_field()` - field construction from mixed components
   - Conversion functions for backward compatibility

2. **Update `src/priors/Priors.jl`:**
   - Include hybrid.jl
   - Export hybrid functions
   - Ensure proper imports of types

3. **Write tests:**
   - Hybrid key sampling consistency
   - Field evaluation correctness
   - Conversion round-trips

**Files Modified:**
- `src/priors/hybrid.jl` (new file)
- `src/priors/Priors.jl` (include + exports)
- `test/test_fields.jl` (add hybrid tests)

**Backward Compatibility:** ✅ New module; no changes to existing

---

### Phase 3: Generative Model Extension

**Objective:** Extend inference model to support RBF and hybrid sampling

**Tasks:**
1. **Create RBF generative functions in `src/inference/gen_model.jl`:**
   - `gen_rbf_bank_fixed(cfg::RBFDiscreteCfg)`
   - Parallel structure to Fourier version
   - Integrates with Gen.jl tracing

2. **Create hybrid generative functions:**
   - `gen_component_modes(K, cfg)` - sample component types
   - `gen_hybrid_components(K, modes, cfg)` - sample per-component parameters
   - `gen_hybrid_bank_fixed(cfg)` - orchestrator

3. **Update `inference_model()` dispatcher:**
   - Check `π_dist.prior_mode`
   - Call appropriate bank function
   - Rest of model unchanged

4. **Update particle filter trace paths in `src/inference/particle_filter.jl`:**
   - Construct selectors based on `π_dist.prior_mode`
   - `:fourier`, `:rbf`, `:hybrid` prefixes

5. **Write tests:**
   - Trace generation for each mode
   - Inference on simple toy problem

**Files Modified:**
- `src/inference/gen_model.jl` (additions + dispatcher update)
- `src/inference/particle_filter.jl` (trace path adaptation)
- `test/test_filter.jl` (add RBF/hybrid trace tests)

**Backward Compatibility:** ✅ Default mode = FOURIER; original paths untouched

---

### Phase 4: MDP Caching & RL Integration

**Objective:** Enable mode-aware MDP construction and caching

**Tasks:**
1. **Update `src/rl/scoredist.jl`:**
   - Modify `ensure_mdp!()` to dispatch based on `π_dist.prior_mode`
   - Create helper: `make_field_from_bank(bank, π_dist)` dispatcher
   - Update `nearest_trained_key()` to use mode-aware distance functions

2. **Add fields to `ScoreΠDist` (if not already done):**
   - `rbf_cfg::RBFDiscreteCfg`
   - `prior_mode::ComponentMode`
   - `hybrid_cfg::HybridCfg`

3. **Write tests:**
   - MDP construction for each mode
   - Policy caching consistency
   - Nearest-neighbor lookups

**Files Modified:**
- `src/rl/scoredist.jl` (ensure_mdp! + dispatcher)
- `src/types.jl` (ScoreΠDist extensions, if needed)
- `test/test_rl.jl` (add mode-specific MDP tests)

**Backward Compatibility:** ✅ Default dispatching to Fourier paths

---

### Phase 5: Visualization & Analysis Dispatchers

**Objective:** Make all visualization functions mode-agnostic

**Tasks:**
1. **Add dispatcher functions in `src/viz/objectives.jl`:**
   - `decode_key(key, π_dist)` - routes to correct decoder
   - `make_field(bank, π_dist)` - routes to correct field constructor

2. **Update visualization functions to use dispatchers:**
   - `plot_top_objective_with_trajectories()` - use dispatchers
   - `plot_objective_side_by_side()` - use dispatchers
   - `make_final_inference_figures()` - use dispatchers
   - (All heatmap/field-based plots)

3. **Update `src/viz/Visualizations.jl`:**
   - Export new dispatchers
   - Update docstrings

4. **Write tests:**
   - Visualization correctness for each mode
   - Integration with ablation pipeline

**Files Modified:**
- `src/viz/objectives.jl` (dispatchers + function updates)
- `src/viz/Visualizations.jl` (exports)
- `test/test_viz.jl` (add mode-specific viz tests)

**Backward Compatibility:** ✅ Fourier mode uses original code paths

---

### Phase 6: Ablation Studies Integration

**Objective:** Enable ablation studies for all modes

**Tasks:**
1. **Update `src/analysis/ablations.jl`:**
   - Add `build_ablation_objectives_hybrid()` function
   - Support sweeps: K, p_fourier, Fourier parameters, RBF parameters
   - Integrate with existing evaluation pipeline

2. **Update `src/analysis/Analysis.jl`:**
   - Export hybrid ablation function

3. **Update example script `iq_sips_ablation.jl`:**
   - Add optional parameter for prior_mode
   - Document hybrid mode usage

4. **Write tests:**
   - Ablation generation for each mode
   - Pipeline integration

**Files Modified:**
- `src/analysis/ablations.jl` (add hybrid ablation function)
- `src/analysis/Analysis.jl` (exports)
- `examples/ablations/iq_sips_ablation.jl` (documentation)

**Backward Compatibility:** ✅ New function; existing pipeline unchanged

---

### Phase 7: Documentation & Testing

**Objective:** Document architecture and validate all paths

**Tasks:**
1. **Update docstrings:**
   - ScoreΠDist: document prior_mode, new config fields
   - Sample functions: document mode-specific behavior
   - Field constructors: document when each is called
   - Visualization: document dispatcher behavior

2. **Create integration test:**
   - End-to-end Fourier pipeline (original)
   - End-to-end RBF pipeline (new)
   - End-to-end Hybrid pipeline (new)
   - Each should produce posterior, visualizations, metrics

3. **Performance benchmarks:**
   - Fourier mode overhead (should be ~0% vs current)
   - RBF mode throughput
   - Hybrid mode scaling with K

4. **Migration guide:**
   - Document how to switch modes
   - Provide example configurations

**Files Modified:**
- All modified files (docstring updates)
- `test/test_integration.jl` (new comprehensive test)
- `docs/HYBRID_USAGE.md` (new usage guide)

**Backward Compatibility:** ✅ 100% guaranteed (validated by integration tests)

---

### Implementation Order Rationale

1. **Start with Phase 1 (Types):** Quick win; unblocks everything else. No risk.

2. **Then Phase 2 (Hybrid Priors):** Core logic independent; can be tested in isolation.

3. **Then Phase 3 (Gen Model):** Integrates types + priors; critical path for inference.

4. **Then Phase 4 (RL Integration):** Builds on inference; enables actual MDP construction.

5. **Then Phase 5 (Visualization):** Depends on MDP construction; enables validation/debugging.

6. **Then Phase 6 (Ablation):** Highest level; depends on everything else; least critical.

7. **Finally Phase 7 (Testing/Docs):** Comprehensive validation and knowledge capture.

---

### Risk Mitigation

**Risk 1:** Breaking existing Fourier code
- **Mitigation:** Default `prior_mode=FOURIER`; Fourier functions unchanged; dispatch-based routing
- **Validation:** Phase 7 integration test for Fourier mode

**Risk 2:** Gen.jl trace incompatibility
- **Mitigation:** Test trace generation early (Phase 3); use mode prefixes to avoid collisions
- **Validation:** Phase 3 trace generation test

**Risk 3:** Performance regression for Fourier
- **Mitigation:** Use dispatch for monomorphic specialization; benchmark Phase 7
- **Validation:** Performance benchmarks show < 1% overhead

**Risk 4:** Incomplete RBF implementation
- **Mitigation:** Complete RBF in Phase 1 with comprehensive tests
- **Validation:** Phase 1 round-trip tests

**Risk 5:** Particle filter incompatibility
- **Mitigation:** Test PF early in Phase 3; trace paths carefully designed
- **Validation:** Phase 3 PF test; Phase 7 integration test

---

### Testing Strategy

#### Unit Tests (by phase)
- Phase 1: Type construction, enum behavior
- Phase 2: Key sampling, decoding, field evaluation, conversion
- Phase 3: Trace generation, key encoding in traces
- Phase 4: MDP construction, distance metrics
- Phase 5: Decoder dispatch, field dispatch, visualization
- Phase 6: Ablation generation, pipeline integration
- Phase 7: End-to-end mode testing

#### Integration Tests (Phase 7)
- Fourier-only pipeline (baseline)
- RBF-only pipeline
- Hybrid pipeline (p_fourier=0.5)
- Hybrid pipeline (p_fourier=0.0 → pure RBF via hybrid mechanism)
- Hybrid pipeline (p_fourier=1.0 → pure Fourier via hybrid mechanism)

#### Regression Tests (Phase 7)
- Run existing test suite with new code
- Verify no change in Fourier-only results
- Verify no performance regression

---

### Checkpoints for Sign-Off

1. **Phase 1 Checkpoint:** Types defined, tests pass, no compilation errors
2. **Phase 2 Checkpoint:** Hybrid module functional, round-trip tests pass
3. **Phase 3 Checkpoint:** Inference model produces valid traces for all modes
4. **Phase 4 Checkpoint:** MDPs constructed correctly; policies cache properly
5. **Phase 5 Checkpoint:** All visualizations produce output for all modes
6. **Phase 6 Checkpoint:** Ablation sweeps generate valid objectives
7. **Phase 7 Checkpoint:** Full integration tests pass; backward compatibility confirmed; documentation complete

---

## Summary of Concrete Requirements

### New Types Required
- [x] `ComponentMode` enum (FOURIER | RBF | HYBRID)
- [x] `HybridCfg` struct
- [x] `FourierComponent` struct
- [x] `RBFComponent` struct
- [x] Extended `ScoreΠDist` with `rbf_cfg`, `prior_mode`, `hybrid_cfg` fields

### New Functions Required
- [x] `sample_hybrid_key()`, `decode_hybrid_key()`, `hamming_hybrid_key()`
- [x] `make_hybrid_scalar_field()`, `objective_grid_from_hybrid_key()`
- [x] `gen_rbf_bank_fixed()`, `gen_component_modes()`, `gen_hybrid_components()`, `gen_hybrid_bank_fixed()`
- [x] `decode_key()` dispatcher, `make_field()` dispatcher
- [x] Conversion functions: `to_hybrid_key()`, `from_hybrid_key()`
- [x] Updated: `ensure_mdp!()`, `nearest_trained_key()`, `inference_model()`, particle filter path adapters
- [x] Updated: All visualization functions (plot_top_objective..., plot_objective_side_by_side, make_final_inference_figures)
- [x] `build_ablation_objectives_hybrid()`

### Files to Create/Modify
- **Create:** `src/priors/hybrid.jl` (new module)
- **Create:** `test/test_integration.jl` (comprehensive integration test)
- **Modify:** `src/types.jl` (add types)
- **Modify:** `src/priors/Priors.jl` (include hybrid module)
- **Modify:** `src/priors/rbf.jl` (complete implementation)
- **Modify:** `src/inference/gen_model.jl` (add RBF/hybrid generators)
- **Modify:** `src/inference/particle_filter.jl` (adapt trace paths)
- **Modify:** `src/rl/scoredist.jl` (mode dispatch)
- **Modify:** `src/viz/objectives.jl` (add dispatchers, update functions)
- **Modify:** `src/viz/Visualizations.jl` (exports)
- **Modify:** `src/analysis/ablations.jl` (add hybrid ablation)
- **Modify:** `test/test_fields.jl`, `test/test_filter.jl`, `test/test_rl.jl`, `test/test_viz.jl` (add tests)

### Backward Compatibility Guarantees
- [x] Default `prior_mode=FOURIER`
- [x] Fourier functions unchanged (sample_fourier_key, decode_fourier_key, make_fourier_scalar_field)
- [x] Fourier trace paths unchanged ((:fourier, :K), (:fourier, :mode, m) => :...)
- [x] Fourier key format unchanged ((K, fx_i, fy_i, A_i, ϕ_i))
- [x] Existing scripts require zero modifications
- [x] Visualization functions detect Fourier mode and use original paths

---

**End of Document**

*This document serves as a design scaffold for implementation sessions. Refer to this for comprehensive requirements, concrete specifications, and suggested phased approach.*

