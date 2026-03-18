# Hybrid Architecture: Quick Reference Checklist

**Purpose:** Single-page checklist and quick lookup for hybrid architecture requirements

---

## Architecture at a Glance

```
Current:  Fourier-only objectives
Goal:     Mixed Fourier + RBF objectives per component
Mechanism: Bernoulli selection of component type during sampling
Result:   Unified scalar fields via superposition
```

---

## Core Types Needed

| Type | Purpose | Location |
|------|---------|----------|
| `ComponentMode` enum | Runtime mode selector (FOURIER\|RBF\|HYBRID) | types.jl |
| `HybridCfg` struct | Bundles both Fourier and RBF configs | types.jl |
| `FourierComponent` | Single Fourier component: fx_i, fy_i, A_i, ϕ_i | types.jl |
| `RBFComponent` | Single RBF component: x_idx, y_idx, amp_idx | types.jl |
| Extended `ScoreΠDist` | Adds: rbf_cfg, prior_mode, hybrid_cfg fields | types.jl |

---

## Key Encoding Formats

### Fourier (unchanged)
```julia
(K::Int, fx_i::Vec{Int}, fy_i::Vec{Int}, A_i::Vec{Int}, ϕ_i::Vec{Int})
```

### RBF (new)
```julia
(K::Int, x_idx::Vec{Int}, y_idx::Vec{Int}, amp_idx::Vec{Int})
```

### Hybrid (new)
```julia
(K::Int, modes::Vec{ComponentMode}, components::Vec{Union{FourierComponent, RBFComponent}})
```

---

## Critical Functions

### Sampling (Gen.jl functions)
- `gen_fourier_bank_fixed(cfg)` - unchanged
- `gen_rbf_bank_fixed(cfg)` - **new**
- `gen_component_modes(K, cfg)` - **new** (Bernoulli per component)
- `gen_hybrid_components(K, modes, cfg)` - **new** (sample per-component params)
- `gen_hybrid_bank_fixed(cfg)` - **new** (orchestrator)

### Decoding (Priors module)
- `decode_fourier_key(key, cfg)` - unchanged
- `decode_rbf_key(key, cfg)` - **new**
- `decode_hybrid_key(key, cfg)` - **new**
- `decode_key(key, π_dist)` - **new dispatcher**

### Field Construction
- `make_fourier_scalar_field(bank; scaleQ)` - unchanged
- `make_rbf_scalar_field(bank; σ, scaleQ)` - **new**
- `make_hybrid_scalar_field(bank; cfg, scaleQ)` - **new**
- `make_field(bank, π_dist)` - **new dispatcher**

### Distance Metrics
- `hamming_fourier_key(k1, k2)` - unchanged
- `hamming_rbf_key(k1, k2)` - **new**
- `hamming_hybrid_key(k1, k2)` - **new** (penalizes type mismatch heavily)

### MDP Integration
- `ensure_mdp!(π_dist, key, bank, agent_params)` - **updated to dispatch**
- `nearest_trained_key(π_dist, key)` - **updated to use mode-aware distance**

---

## Inference Flow by Mode

### FOURIER Mode (unchanged)
```
ScoreΠDist(fourier_cfg, prior_mode=FOURIER)
    ↓
gen_fourier_bank_fixed()
    ↓
(K, fx_i, fy_i, A_i, ϕ_i) key
    ↓
decode_fourier_key() → continuous params
    ↓
make_fourier_scalar_field() → (x,y)→Float64
```

### RBF Mode (new)
```
ScoreΠDist(rbf_cfg, prior_mode=RBF)
    ↓
gen_rbf_bank_fixed()
    ↓
(K, x_idx, y_idx, amp_idx) key
    ↓
decode_rbf_key() → continuous params
    ↓
make_rbf_scalar_field() → (x,y)→Float64
```

### HYBRID Mode (new)
```
ScoreΠDist(hybrid_cfg, prior_mode=HYBRID)
    ↓
gen_hybrid_bank_fixed()
    ├─ gen_K() → K
    ├─ gen_component_modes(K) → [FOURIER|RBF, ...]
    └─ gen_hybrid_components(K, modes) → components
    ↓
(K, modes, components) key
    ↓
decode_hybrid_key() → mixed continuous params
    ↓
make_hybrid_scalar_field() → (x,y)→Float64
```

---

## Particle Filter Trace Paths

### FOURIER Mode (unchanged)
```julia
(:fourier, :K)
(:fourier, :mode, m) => :fx_idx
(:fourier, :mode, m) => :fy_idx
(:fourier, :mode, m) => :A_idx
(:fourier, :mode, m) => :ϕ_idx
```

### RBF Mode (new)
```julia
(:rbf, :K)
(:rbf, :mode, m) => :x_idx
(:rbf, :mode, m) => :y_idx
(:rbf, :mode, m) => :amp_idx
```

### HYBRID Mode (new)
```julia
(:hybrid, :K)
(:hybrid, :modes, m) => :mode        # which type?
# Then conditional on mode[m]:
(:hybrid, :components, m, :fourier) => :fx_idx
(:hybrid, :components, m, :rbf) => :x_idx
```

---

## Dispatch Pattern (Template)

```julia
# DECODER DISPATCHER
function decode_key(key, π_dist::ScoreΠDist)
    if π_dist.prior_mode == FOURIER
        return decode_fourier_key(key, π_dist.fourier_cfg)
    elseif π_dist.prior_mode == RBF
        return decode_rbf_key(key, π_dist.rbf_cfg)
    else  # HYBRID
        return decode_hybrid_key(key, π_dist.hybrid_cfg)
    end
end

# FIELD DISPATCHER
function make_field(bank::NamedTuple, π_dist::ScoreΠDist)
    if π_dist.prior_mode == FOURIER
        return make_fourier_scalar_field(bank; scaleQ=true)
    elseif π_dist.prior_mode == RBF
        return make_rbf_scalar_field(bank; σ=π_dist.rbf_cfg.σ)
    else  # HYBRID
        return make_hybrid_scalar_field(bank; cfg=π_dist.hybrid_cfg, scaleQ=true)
    end
end
```

---

## Files to Create/Modify

### Create (new files)
- ✅ `src/priors/hybrid.jl` - hybrid module (sampling, decoding, field construction)
- ✅ `test/test_integration.jl` - comprehensive integration tests

### Modify (existing files)
- ✅ `src/types.jl` - add component types and enums
- ✅ `src/priors/Priors.jl` - include hybrid module
- ✅ `src/priors/rbf.jl` - complete/verify RBF implementation
- ✅ `src/inference/gen_model.jl` - add RBF/hybrid generators + dispatcher
- ✅ `src/inference/particle_filter.jl` - adapt trace paths to prior_mode
- ✅ `src/rl/scoredist.jl` - update ensure_mdp, nearest_trained_key with dispatch
- ✅ `src/viz/objectives.jl` - add dispatchers, update visualization functions
- ✅ `src/viz/Visualizations.jl` - export dispatchers
- ✅ `src/analysis/ablations.jl` - add build_ablation_objectives_hybrid
- ✅ `test/test_fields.jl` - add RBF/hybrid tests
- ✅ `test/test_filter.jl` - add mode-specific trace tests
- ✅ `test/test_rl.jl` - add mode-specific MDP tests
- ✅ `test/test_viz.jl` - add mode-specific visualization tests

---

## Implementation Phases

| Phase | Goal | Duration |
|-------|------|----------|
| 1 | Types + RBF completion | 1-2 days |
| 2 | Hybrid priors module | 1 day |
| 3 | Gen model + particle filter | 2-3 days |
| 4 | MDP caching + RL dispatch | 1 day |
| 5 | Visualization dispatchers | 1-2 days |
| 6 | Ablation integration | 0.5 day |
| 7 | Testing + documentation | 2-3 days |
| **Total** | | **9-13 days** |

---

## Backward Compatibility Guarantees

| Item | Guarantee | How |
|------|-----------|-----|
| Fourier code | Zero changes needed | Default `prior_mode=FOURIER` |
| Fourier functions | Unchanged API | No modifications to sample/decode/make_field |
| Fourier keys | Unchanged format | (K, fx_i, fy_i, A_i, ϕ_i) identical |
| Trace paths | Unchanged | (:fourier, ...) paths identical |
| Performance | No overhead | Dispatch-based monomorphic specialization |
| Serialization | Fully compatible | Fourier keys serialize identically |

---

## Critical Tests (Minimum)

1. **Type construction:** Create instances of all new types
2. **Fourier round-trip:** sample → decode → identical continuous params
3. **RBF round-trip:** sample → decode → identical continuous params
4. **Hybrid round-trip:** sample → decode → identical continuous and type info
5. **Fourier field:** Evaluate Fourier field, check values finite and reasonable
6. **RBF field:** Evaluate RBF field, check values finite and reasonable
7. **Hybrid field:** Evaluate hybrid field, check superposition works
8. **Trace generation:** Gen.jl traces valid for all modes
9. **MDP construction:** MDP created successfully for all modes
10. **Visualization:** Plots generate without errors for all modes
11. **Particle filter:** PF runs on toy problem for all modes
12. **Backward compat:** Run existing Fourier test suite, results identical

---

## Configuration Examples

### Pure Fourier (default, backward compatible)
```julia
cfg = FourierDiscreteCfg(Kmax=10, λK=0.35, ...)
π_dist = ScoreΠDist(fourier_cfg=cfg)  # prior_mode defaults to FOURIER
```

### Pure RBF (new)
```julia
cfg = RBFDiscreteCfg(Kmax=5, λK=0.5, σ=1.0, ...)
π_dist = ScoreΠDist(rbf_cfg=cfg, prior_mode=RBF)
```

### 50/50 Hybrid (new)
```julia
cfg = HybridCfg(
    fourier_cfg=FourierDiscreteCfg(...),
    rbf_cfg=RBFDiscreteCfg(...),
    p_fourier=0.5
)
π_dist = ScoreΠDist(hybrid_cfg=cfg, prior_mode=HYBRID)
```

### 80/20 Hybrid: More Fourier (new)
```julia
cfg = HybridCfg(
    fourier_cfg=FourierDiscreteCfg(...),
    rbf_cfg=RBFDiscreteCfg(...),
    p_fourier=0.8
)
π_dist = ScoreΠDist(hybrid_cfg=cfg, prior_mode=HYBRID)
```

---

## Key Insight: Composability

**All objectives are composition of independent components:**

```
Objective = Σ_{m=1..K} Component_m(x, y)

where Component_m ∈ {
    Fourier: A·cos(fx·x + fy·y + ϕ)
    RBF:     A·exp(-(r²/(2σ²)))
}
```

**Enables:**
- Per-component type selection (Fourier XOR RBF)
- Unified field interface (all produce scalars)
- Future extensibility (add more component types)
- Clean inference logic (sample + evaluate)

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────┐
│           Configuration Layer (types.jl)            │
│  FourierDiscreteCfg | RBFDiscreteCfg | HybridCfg   │
│           + ComponentMode enum                      │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│         Sampling Layer (gen_model.jl)               │
│  gen_fourier_bank_fixed | gen_rbf_bank_fixed |      │
│  gen_hybrid_bank_fixed (dispatcher)                 │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│    Key Encoding (Tuple / Component structure)       │
│  Fourier | RBF | Hybrid                             │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│    Decoding Layer (Priors / dispatchers)            │
│  decode_fourier_key | decode_rbf_key |             │
│  decode_hybrid_key (dispatcher)                     │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│  Field Construction (make_field dispatcher)         │
│  make_fourier_scalar_field | make_rbf_scalar_field │
│  make_hybrid_scalar_field                           │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│        Field Evaluation (x,y) → Float64             │
│         Used by: MDP, visualization, metrics        │
└─────────────────────────────────────────────────────┘
```

---

## Document References

- **Full Requirements:** `HYBRID_ARCHITECTURE_REQUIREMENTS.md` (this file)
- **Analysis Document:** `HYBRID_ARCHITECTURE_ANALYSIS.md` (downstrea implementation details)

---

**Status:** Design scaffold complete. Ready for implementation sessions.

**Last Updated:** March 18, 2026

