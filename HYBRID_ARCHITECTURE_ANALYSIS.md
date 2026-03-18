# Hybrid Objective Architecture: Analysis & Implementation Details

**Document Version:** 1.0  
**Date:** March 18, 2026  
**Purpose:** Deep technical analysis of downstream implementation impact  
**Status:** Supporting document for requirements (see HYBRID_ARCHITECTURE_REQUIREMENTS.md for specs)

---

## Table of Contents

1. [Current Inference Pipeline](#current-inference-pipeline)
2. [Downstream Impact Analysis](#downstream-impact-analysis)
3. [Data Structure Changes](#data-structure-changes)
4. [Function Modification Strategy](#function-modification-strategy)
5. [Outdated Functions Analysis](#outdated-functions-analysis)

---

## Current Inference Pipeline

### 1. Configuration Setup
```julia
cfg = FourierDiscreteCfg(Kmax=10, λK=0.35, ...)
π_dist = ScoreΠDist(fourier_cfg=cfg)
agent_params = Dict(:start=>[...], :dimensions=>(0.0, 10.0), :menv=>menv, ...)
```

### 2. Generative Model Sampling
```julia
@gen function gen_K(cfg) → K ~ categorical(K_probs(cfg))
@gen function gen_mode_indices(K, cfg) → (fx_i, fy_i, A_i, ϕ_i)
@gen function gen_fourier_bank_fixed(cfg) → (key, K, fx, fy, A, ϕ, ...)
@gen function inference_model(N, π_dist, agent_params, state_data)
    bank ~ gen_fourier_bank_fixed(π_dist.fourier_cfg)
    key = bank.key
    # ...register key, build MDP...
    # ...trace actions...
    return key
```

### 3. Particle Filter Execution
```julia
state = pf_initialize(inference_model, ...)
for n in 2:N
    if should_resample(state)
        pf_resample!(state)
        # Trace paths: (:fourier, :K), (:fourier, :mode, m) => :...
        pf_rejuvenate!(state, mh, (select(...),))
    end
    pf_update!(state, ...)
end
```

### 4. Posterior Aggregation
```julia
tops = top_objectives(pf_state, π_dist; topk=10)
# Returns: (key, prob, count, params)
```

### 5. Visualization & Analysis
```julia
key = tops[1].key
bank = decode_fourier_key(key, π_dist.fourier_cfg)
field = make_fourier_scalar_field(bank; scaleQ=true)
obj = make_pomdp_objective_from_field(field)
# ... build MDP, plot, evaluate metrics
```

---

## Downstream Impact Analysis

### REQ-D1: Configuration Impact

**Current State:**
```julia
π_dist.fourier_cfg::FourierDiscreteCfg  # Only Fourier configuration
```

**Required Changes:**
```julia
# Option 1: Fourier mode (backward compatible)
π_dist.fourier_cfg::FourierDiscreteCfg = FourierDiscreteCfg()
π_dist.prior_mode::ComponentMode = FOURIER  # Default

# Option 2: RBF mode (new)
π_dist.rbf_cfg::RBFDiscreteCfg = RBFDiscreteCfg()
π_dist.prior_mode::ComponentMode = RBF

# Option 3: Hybrid mode (new)
π_dist.hybrid_cfg::HybridCfg = HybridCfg(...)
π_dist.prior_mode::ComponentMode = HYBRID
```

**Downstream Consequences:**
1. All code accessing `π_dist.fourier_cfg` must check `prior_mode` first
2. Field construction code must dispatch on `prior_mode`
3. Visualization assumes Fourier; must be updated
4. MDP construction hardcoded to Fourier; must be updated

**Impact Radius:** 
- 🔴 HIGH: `inference_model`, `ensure_mdp!`, all visualization functions
- 🟡 MEDIUM: Particle filter, ablation studies
- 🟢 LOW: Top-level API, test infrastructure

---

### REQ-D2: Key Encoding Impact

**Current State:**
```julia
key = (K, fx_i, fy_i, A_i, ϕ_i)  # Fourier-specific tuple
# Used in: registration, caching, posterior analysis, serialization
```

**Required Changes:**
```julia
# Fourier mode (backward compatible):
key_fourier = (K, fx_i, fy_i, A_i, ϕ_i)

# RBF mode (new):
key_rbf = (K, x_idx, y_idx, amp_idx)

# Hybrid mode (new):
key_hybrid = (K, modes::Vec{ComponentMode}, components::Vec{Union{...}})
```

**Downstream Consequences:**
1. `π_dist.prop_names` now contains heterogeneous key types
2. Distance metrics (`hamming_fourier_key`) must dispatch on key type
3. Nearest-neighbor logic must know key type
4. Serialization must handle three formats
5. Cache lookup must be type-aware

**Impact Radius:**
- 🔴 HIGH: `ScoreΠDist` key storage, nearest_trained_key lookup
- 🟡 MEDIUM: Posterior aggregation, ablation cache storage
- 🟢 LOW: Public API (keys are opaque to users)

---

### REQ-D3: Generative Model Impact

**Current State:**
```
inference_model()
  └─ gen_fourier_bank_fixed()
       ├─ gen_K() 
       └─ gen_mode_indices() × K
```

**Required Changes:**
```
inference_model() with dispatcher:
  ├─ if prior_mode == FOURIER: gen_fourier_bank_fixed()
  ├─ if prior_mode == RBF: gen_rbf_bank_fixed()
  └─ if prior_mode == HYBRID: gen_hybrid_bank_fixed()

gen_hybrid_bank_fixed():
  ├─ gen_K()
  ├─ gen_component_modes(K)  ← NEW
  ├─ gen_hybrid_components(K, modes)  ← NEW
  └─ compose key = (K, modes, components)
```

**Downstream Consequences:**
1. Trace structure changes per mode
2. Particle filter rejuvenation must adapt trace paths
3. Three separate generative models to maintain
4. Backward compatibility: Fourier path unchanged

**Impact Radius:**
- 🔴 HIGH: Entire inference pipeline
- 🟡 MEDIUM: Particle filter, trace debugging
- 🟢 LOW: Ablation studies (only calls sampling functions)

---

### REQ-D4: Field Construction Impact

**Current State:**
```julia
bank = decode_fourier_key(key, cfg)
field = make_fourier_scalar_field(bank; scaleQ=true)
# Used in: ensure_mdp!, visualization, analysis
```

**Required Changes:**
```julia
# Mode dispatcher needed at call sites:
if π_dist.prior_mode == FOURIER
    bank = decode_fourier_key(key, π_dist.fourier_cfg)
    field = make_fourier_scalar_field(bank; scaleQ=true)
elseif π_dist.prior_mode == RBF
    bank = decode_rbf_key(key, π_dist.rbf_cfg)
    field = make_rbf_scalar_field(bank; σ=π_dist.rbf_cfg.σ)
else  # HYBRID
    bank = decode_hybrid_key(key, π_dist.hybrid_cfg)
    field = make_hybrid_scalar_field(bank; cfg=π_dist.hybrid_cfg, scaleQ=true)
end
```

**Downstream Consequences:**
1. Every function constructing fields must add dispatch logic
2. Visualization functions need refactoring
3. MDP construction needs refactoring
4. New field constructor must be implemented (`make_hybrid_scalar_field`)

**Affected Functions:**
- `ensure_mdp!()` - 🔴 CRITICAL
- `plot_top_objective_with_trajectories()` - 🔴 CRITICAL
- `plot_objective_side_by_side()` - 🔴 CRITICAL
- `make_final_inference_figures()` - 🔴 CRITICAL
- All grid evaluation functions - 🟡 MEDIUM
- Metric computation functions - 🟡 MEDIUM

**Impact Radius:**
- 🔴 HIGH: 5+ visualization functions
- 🟡 MEDIUM: MDP construction, analysis metrics
- 🟢 LOW: Test infrastructure

---

### REQ-D5: Distance Metric Impact

**Current State:**
```julia
d = hamming_fourier_key(k1, k2)  # Distance between Fourier keys
# Used in: nearest_trained_key lookup
```

**Required Changes:**
```julia
function hamming_key(k1, k2, π_dist)
    if π_dist.prior_mode == FOURIER
        return hamming_fourier_key(k1, k2)
    elseif π_dist.prior_mode == RBF
        return hamming_rbf_key(k1, k2)
    else  # HYBRID
        return hamming_hybrid_key(k1, k2)
    end
end
```

**Downstream Consequences:**
1. `nearest_trained_key()` must dispatch
2. Lookup reliability depends on correct distance metric
3. Hybrid metric must weight type changes heavily
4. Three separate distance functions to maintain/debug

**Impact Radius:**
- 🟡 MEDIUM: Policy refinement, nearest-neighbor lookups
- 🟢 LOW: Not critical to core inference (optional optimization)

---

### REQ-D6: Particle Filter Trace Impact

**Current State:**
```julia
# Trace paths:
(:fourier, :K)
(:fourier, :mode, m) => :fx_idx
(:fourier, :mode, m) => :fy_idx
(:fourier, :mode, m) => :A_idx
(:fourier, :mode, m) => :ϕ_idx
```

**Required Changes:**
```julia
# Mode FOURIER: unchanged
(:fourier, :K)
(:fourier, :mode, m) => :fx_idx
# ...

# Mode RBF: new paths
(:rbf, :K)
(:rbf, :mode, m) => :x_idx
(:rbf, :mode, m) => :y_idx
(:rbf, :mode, m) => :amp_idx

# Mode HYBRID: new paths
(:hybrid, :K)
(:hybrid, :modes, m) => :mode  # which type is component m?
# Then conditional:
(:hybrid, :components, m, :fourier) => :fx_idx  # if FOURIER
(:hybrid, :components, m, :rbf) => :x_idx       # if RBF
```

**Downstream Consequences:**
1. Trace path selection in rejuvenation must be mode-aware
2. Cannot use hardcoded selectors
3. Must dynamically construct selector list
4. More complex rejuvenation logic

**Modified Function:**
- `particle_filter()` - 🔴 CRITICAL (rejuvenation section)

**Backward Compatibility:**
- Fourier mode uses original paths → no change
- RBF/hybrid modes add new paths → no collision

**Impact Radius:**
- 🔴 HIGH: Particle filter rejuvenation
- 🟢 LOW: Rest of particle filter (init/update/resample unchanged)

---

## Data Structure Changes

### ScoreΠDist Extension

**Current:**
```julia
@with_kw struct ScoreΠDist
    prop_names::Vector = []
    n_qprop_list::Dict{Any, Float64} = Dict{Any, Float64}()
    n_propmdp_list::Dict{Any, Any} = Dict{Any, Any}()
    n_𝒮_proposals::Dict{Any, Any} = Dict{Any, Any}()
    n_π_proposals::Dict{Any, Any} = Dict{Any, Any}()
    mdp_params::Vector = []
    fourier_cfg::FourierDiscreteCfg = FourierDiscreteCfg()
end
```

**Extended:**
```julia
@with_kw struct ScoreΠDist
    # ... existing fields (unchanged) ...
    
    # NEW fields:
    fourier_cfg::FourierDiscreteCfg = FourierDiscreteCfg()  # KEEP for backward compat
    rbf_cfg::RBFDiscreteCfg = RBFDiscreteCfg()             # NEW
    prior_mode::ComponentMode = FOURIER                    # NEW
    hybrid_cfg::HybridCfg = HybridCfg()                    # NEW
end
```

**Backward Compatibility:**
- All existing code sees `fourier_cfg` (unchanged)
- New fields have sensible defaults
- Existing serialized `ScoreΠDist` still deserializes (new fields initialize to defaults)

---

### Key Storage Heterogeneity

**Current State:**
```julia
π_dist.prop_names::Vector = [key1, key2, ...]  # All keys format: (K, fx_i, fy_i, A_i, ϕ_i)
```

**New State:**
```julia
π_dist.prop_names::Vector = [
    (K, fx_i, fy_i, A_i, ϕ_i),                              # Fourier
    (K, x_idx, y_idx, amp_idx),                            # RBF
    (K, modes, components),                                # Hybrid
    ...
]
```

**Implication:**
- `π_dist.prop_names` is now heterogeneous
- Code cannot assume key format
- Must dispatch on `π_dist.prior_mode` to interpret keys
- **Not a breaking change:** code already doesn't interpret keys (only caches them)

---

## Function Modification Strategy

### Strategy 1: Dispatcher Pattern (Recommended)

```julia
# Define dispatcher once
function decode_key(key, π_dist::ScoreΠDist)
    if π_dist.prior_mode == FOURIER
        return decode_fourier_key(key, π_dist.fourier_cfg)
    elseif π_dist.prior_mode == RBF
        return decode_rbf_key(key, π_dist.rbf_cfg)
    else
        return decode_hybrid_key(key, π_dist.hybrid_cfg)
    end
end

# Use in visualization functions:
function plot_top_objective_with_trajectories(pf_state, π_dist, ...)
    key, prob = RL.top_key(pf_state, π_dist)
    bank = decode_key(key, π_dist)  # ← USE DISPATCHER
    field = make_field(bank, π_dist)  # ← USE DISPATCHER
    # ... rest unchanged
end
```

**Advantages:**
- ✅ Centralized dispatch logic
- ✅ Easy to extend with new modes
- ✅ Minimal changes to existing functions
- ✅ Fourier path remains unchanged (monomorphic for Fourier users)

---

### Strategy 2: Mode-Specific Branches (Alternative)

```julia
function plot_top_objective_with_trajectories(pf_state, π_dist, ...)
    key, prob = RL.top_key(pf_state, π_dist)
    
    if π_dist.prior_mode == FOURIER
        bank = decode_fourier_key(key, π_dist.fourier_cfg)
        field = make_fourier_scalar_field(bank; scaleQ=true)
    elseif π_dist.prior_mode == RBF
        bank = decode_rbf_key(key, π_dist.rbf_cfg)
        field = make_rbf_scalar_field(bank; σ=π_dist.rbf_cfg.σ)
    else  # HYBRID
        bank = decode_hybrid_key(key, π_dist.hybrid_cfg)
        field = make_hybrid_scalar_field(bank; cfg=π_dist.hybrid_cfg, scaleQ=true)
    end
    
    # ... rest unchanged
end
```

**Disadvantages:**
- ❌ Code duplication in every visualization function
- ❌ Difficult to extend with new modes
- ✅ Might be faster (inlines dispatch)

---

**Recommendation:** Use Strategy 1 (dispatchers) for maintainability.

---

## Outdated Functions Analysis

### Tier 1: Definitely Remove

| Function | Module | Reason | Alternative |
|----------|--------|--------|-------------|
| `nearest_trained_key()` | fourier.jl | Never called in ablation pipeline; `top_objectives()` is better | `top_objectives()` |
| `hamming_fourier_key()` | fourier.jl | Only used by `nearest_trained_key()`; can be inlined | Inlined |
| `gen_mode_indices()` | gen_model.jl | Parallel design; `gen_fourier_bank_fixed` is main path | `gen_fourier_bank_fixed` |
| `policy_match_acc()` | metrics.jl | Defined for policy accuracy, never called in ablation | Unused |

### Tier 2: Probably Remove

| Function | Module | Reason | Status |
|----------|--------|--------|--------|
| `objective_grid_from_key()` | fourier.jl | Wrapper around `decode_fourier_key()` + `objective_grid_from_field()` | Redundant; combine into one |
| `eval_ablation_on_dataset()` | ablations.jl | Complex variant; replaced by `ablation_main()` | Superseded |
| `surrogate_dataset_from_iql_grid()` | RL | Marked with TODO in ablations.jl:543 | Incomplete |

### Tier 3: Keep (Used Indirectly)

| Function | Module | Reason |
|----------|--------|--------|
| `freq_bin_support_and_probs()` | fourier.jl | Internal to gen_model sampling |
| `amp_bin_support_and_probs()` | fourier.jl | Internal to gen_model sampling |
| `phase_bin_support_and_probs()` | fourier.jl | Internal to gen_model sampling |
| `f_from_i()` | fourier.jl | Parameter converter, used in decode + field |
| `A_from_i()` | fourier.jl | Parameter converter, used in decode + field |
| `ϕ_from_i()` | fourier.jl | Parameter converter, used in decode + field |

### Functions with RBF Equivalents (Keep for Hybrid)

| Function | Module | Status | Keep? |
|----------|--------|--------|-------|
| `sample_rbf_key()` | rbf.jl | Needed for hybrid sampling | ✅ YES |
| `decode_rbf_key()` | rbf.jl | Needed for hybrid decoding | ✅ YES |
| `hamming_rbf_key()` | rbf.jl | Needed for hybrid distance | ✅ YES |
| `nearest_trained_key_rbf()` | rbf.jl | Parallel to Fourier version | ⚠️ MAYBE |
| `make_rbf_scalar_field()` | rbf.jl | Needed for hybrid fields | ✅ YES |
| `objective_grid_from_rbf_key()` | rbf.jl | Needed for analysis | ✅ YES |

**Recommendation:** Keep all RBF functions; they're essential for hybrid implementation.

---

## Return Value Consistency

### Key Guarantee: Unified Field Interface

**Requirement:** All field constructors return `(x::Real, y::Real) → Float64` closure.

**Specification:**

```julia
# All these return identical closure signature:
field_fourier = make_fourier_scalar_field(bank_fourier; scaleQ=true)::Function
field_rbf = make_rbf_scalar_field(bank_rbf; σ=1.0, scaleQ=true)::Function
field_hybrid = make_hybrid_scalar_field(bank_hybrid; cfg=cfg, scaleQ=true)::Function

# Usage is identical:
value = field_fourier(5.0, 3.0)  # Float64
value = field_rbf(5.0, 3.0)      # Float64
value = field_hybrid(5.0, 3.0)   # Float64

# Downstream code unchanged:
obj = make_pomdp_objective_from_field(field)  # Works for any field
Z = objective_grid_from_field(field, xs, ys) # Works for any field
```

**Benefit:** All downstream code treating fields uniformly; no dispatch needed after field construction.

---

## Summary of Return Value Changes

### Functions Returning Mode-Aware Values

| Function | Old Return | New Return | Dispatch Needed? |
|----------|-----------|-----------|------------------|
| `decode_key` | N/A | NamedTuple (mixed) | ✅ In function |
| `make_field` | N/A | Function (uniform) | ✅ In function |
| `hamming_key` | Int | Int | ✅ In function |
| `inference_model` | key | key (mode-specific format) | ❌ Upstream handles |
| `gen_hybrid_bank_fixed` | N/A | NamedTuple (with modes) | N/A |

**Principle:** Dispatch happens inside these functions; callers see uniform interface.

---

**End of Document**

*This document provides technical depth for implementation decisions. Refer to HYBRID_ARCHITECTURE_REQUIREMENTS.md for the full specification.*

