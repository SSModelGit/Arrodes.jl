# Hybrid Modal Objective Architecture: Documentation Index

**Created:** March 18, 2026  
**Purpose:** Design scaffolding for mixed Fourier + RBF objective function system  
**Status:** Complete architectural specification (pre-implementation)

---

## Documentation Structure

This design comprises **three complementary documents**, each serving a specific purpose:

### 1. **HYBRID_ARCHITECTURE_REQUIREMENTS.md** (Primary Document)
**→ Start here if:**
- You're implementing the architecture
- You need the complete specification
- You want concrete function signatures
- You're planning development timeline

**Contents:**
- Executive overview and goals
- 10 detailed capability requirements (REQ-1 through REQ-10)
- Complete type system design with examples
- Functional specifications for all new functions
- Integration points across the codebase
- Backward compatibility guarantees
- 7-phase implementation approach with checkpoints
- Risk mitigation strategies

**Length:** ~1,400 lines | **Read Time:** 30-40 minutes

**Key Sections:**
- Section 3: "Capability Requirements" — detailed specs for each component
- Section 8: "Suggested Implementation Approach" — phased rollout plan
- Section 4: "Data Flow & Type System" — architecture diagrams

---

### 2. **HYBRID_ARCHITECTURE_ANALYSIS.md** (Technical Details)
**→ Reference this when:**
- You need to understand downstream impacts
- You're planning specific code changes
- You want to see before/after code patterns
- You're analyzing which functions to modify

**Contents:**
- Current inference pipeline walkthrough
- Detailed downstream impact analysis (REQ-D1 through REQ-D6)
- Data structure extension details
- Function modification strategies with pros/cons
- Outdated function removal analysis (3 tiers)
- Return value consistency principles

**Length:** ~530 lines | **Read Time:** 15-20 minutes

**Key Sections:**
- Section 2: "Downstream Impact Analysis" — impact radius by component
- Section 5: "Outdated Functions Analysis" — which functions can be removed
- Section 4: "Function Modification Strategy" — dispatcher vs branch patterns

---

### 3. **HYBRID_ARCHITECTURE_QUICK_REFERENCE.md** (Cheat Sheet)
**→ Use this for:**
- Quick lookups during implementation
- Remembering key formats and trace paths
- Configuration examples
- Implementation checklists
- Architecture diagrams

**Contents:**
- Architecture at a glance
- Core types needed (table)
- Key encoding formats (all three modes)
- Critical functions (sampling, decoding, field, distance)
- Inference flow by mode (visual)
- Particle filter trace paths
- Dispatch pattern template
- Files to create/modify
- Implementation phases summary
- Backward compatibility guarantees
- Configuration examples

**Length:** ~360 lines | **Read Time:** 5-10 minutes

**Key Sections:**
- "Core Types Needed" — quick lookup table
- "Critical Functions" — function signature reference
- "Implementation Phases" — timeline summary
- "Configuration Examples" — copy-paste ready

---

## Reading Paths

### Path 1: Full Implementation Planning
1. Read **REQUIREMENTS.md** (Section 1-2: Executive Overview + Goals)
2. Read **REQUIREMENTS.md** (Section 3: Capability Requirements)
3. Read **QUICK_REFERENCE.md** (for context)
4. Read **ANALYSIS.md** (for implementation details)
5. Read **REQUIREMENTS.md** (Section 8: Implementation Approach)

**Time:** ~60 minutes | **Outcome:** Ready to start Phase 1

### Path 2: Understand Current Impact
1. Read **ANALYSIS.md** (Section 2: Downstream Impact Analysis)
2. Skim **REQUIREMENTS.md** (Section 3: Capability Requirements)
3. Use **QUICK_REFERENCE.md** for specific lookups

**Time:** ~25 minutes | **Outcome:** Know what code changes where

### Path 3: Quick Reference During Coding
1. Use **QUICK_REFERENCE.md** as primary reference
2. Jump to **REQUIREMENTS.md** for detailed specs
3. Check **ANALYSIS.md** for modification strategies

**Time:** Variable | **Outcome:** On-demand lookup while coding

### Path 4: Backward Compatibility Verification
1. Read **REQUIREMENTS.md** (Section 7: Backward Compatibility)
2. Check **QUICK_REFERENCE.md** (Backward Compatibility Guarantees table)
3. Reference **ANALYSIS.md** (Section 1: Current Pipeline)

**Time:** ~20 minutes | **Outcome:** Confidence in compatibility

---

## Key Decisions Made

### 1. Mode-Based Dispatch
**Decision:** Use runtime `prior_mode` enum to select between Fourier/RBF/Hybrid  
**Rationale:** Clean separation; enables future extensions; zero overhead for monomorphic Fourier users  
**Location:** REQUIREMENTS.md REQ-1, QUICK_REFERENCE.md "Dispatch Pattern"

### 2. Backward Compatibility via Default
**Decision:** Default `prior_mode=FOURIER` in ScoreΠDist  
**Rationale:** Existing code continues unchanged; no migration required  
**Location:** REQUIREMENTS.md Section 7, QUICK_REFERENCE.md "Backward Compatibility"

### 3. Unified Field Interface
**Decision:** All field constructors return `(x, y) → Float64` closure  
**Rationale:** Downstream code uniform; no dispatch after construction  
**Location:** REQUIREMENTS.md REQ-4, ANALYSIS.md "Return Value Consistency"

### 4. Component-Based Encoding
**Decision:** Hybrid keys explicitly track component types  
**Rationale:** Enables per-component diagnostics; simplifies distance metrics  
**Location:** REQUIREMENTS.md REQ-2, QUICK_REFERENCE.md "Key Encoding Formats"

### 5. Bernoulli Selection
**Decision:** Each component independently selected as Fourier or RBF  
**Rationale:** Simple probabilistic model; enables all-Fourier/all-RBF/mixed through single framework  
**Location:** REQUIREMENTS.md REQ-3, QUICK_REFERENCE.md "Hybrid Mode (new)"

---

## Implementation Checklist

### Pre-Implementation
- [ ] Read HYBRID_ARCHITECTURE_REQUIREMENTS.md Sections 1-2
- [ ] Read HYBRID_ARCHITECTURE_ANALYSIS.md Section 2
- [ ] Understand type system (REQUIREMENTS.md Section 4)
- [ ] Review current pipeline (ANALYSIS.md Section 1)

### Phase 1: Types (1-2 days)
- [ ] Read REQUIREMENTS.md Section 4 (Data Flow)
- [ ] Add ComponentMode enum to types.jl
- [ ] Add HybridCfg struct to types.jl
- [ ] Add FourierComponent, RBFComponent structs
- [ ] Extend ScoreΠDist fields
- [ ] Run type construction tests
- [ ] Verify exports

### Phase 2: Hybrid Priors (1 day)
- [ ] Read REQUIREMENTS.md REQ-2
- [ ] Create src/priors/hybrid.jl
- [ ] Implement sample_hybrid_key()
- [ ] Implement decode_hybrid_key()
- [ ] Implement make_hybrid_scalar_field()
- [ ] Implement hamming_hybrid_key()
- [ ] Write round-trip tests

### Phase 3: Generative Model (2-3 days)
- [ ] Read REQUIREMENTS.md REQ-3
- [ ] Create gen_rbf_bank_fixed()
- [ ] Create gen_component_modes()
- [ ] Create gen_hybrid_components()
- [ ] Create gen_hybrid_bank_fixed()
- [ ] Update inference_model() dispatcher
- [ ] Update particle_filter() trace paths
- [ ] Test trace generation

### Phase 4: MDP Caching (1 day)
- [ ] Read ANALYSIS.md Section 2 (ensure_mdp impact)
- [ ] Create make_field() dispatcher
- [ ] Update ensure_mdp!() to dispatch
- [ ] Update nearest_trained_key() if used
- [ ] Write MDP construction tests

### Phase 5: Visualization (1-2 days)
- [ ] Read ANALYSIS.md Section 4 (Function modification strategies)
- [ ] Create decode_key() dispatcher
- [ ] Update plot_top_objective_with_trajectories()
- [ ] Update plot_objective_side_by_side()
- [ ] Update make_final_inference_figures()
- [ ] Update all grid evaluation functions
- [ ] Write visualization tests

### Phase 6: Ablation Integration (0.5 day)
- [ ] Add build_ablation_objectives_hybrid()
- [ ] Test sweep generation
- [ ] Verify pipeline integration

### Phase 7: Testing & Docs (2-3 days)
- [ ] Create comprehensive integration test
- [ ] Test Fourier mode (baseline)
- [ ] Test RBF mode
- [ ] Test Hybrid modes (p=0.0, 0.5, 1.0)
- [ ] Verify backward compatibility
- [ ] Update docstrings
- [ ] Performance benchmarks

---

## Quick Navigation

### By Topic

**Type System**
- REQUIREMENTS.md Section 4 (Data Flow & Type System)
- QUICK_REFERENCE.md "Core Types Needed"
- ANALYSIS.md Section 3 (Data Structure Changes)

**Sampling & Generative Model**
- REQUIREMENTS.md REQ-3
- QUICK_REFERENCE.md "Inference Flow by Mode"
- ANALYSIS.md Section 2 (REQ-D3)

**Field Construction**
- REQUIREMENTS.md REQ-4
- QUICK_REFERENCE.md "Critical Functions"
- ANALYSIS.md Section 2 (REQ-D4)

**Particle Filter**
- REQUIREMENTS.md REQ-7
- QUICK_REFERENCE.md "Particle Filter Trace Paths"
- ANALYSIS.md Section 2 (REQ-D6)

**Visualization**
- REQUIREMENTS.md REQ-8
- ANALYSIS.md Section 2 (REQ-D4)
- ANALYSIS.md Section 4 (Function Modification Strategy)

**Backward Compatibility**
- REQUIREMENTS.md Section 7
- QUICK_REFERENCE.md "Backward Compatibility Guarantees"
- ANALYSIS.md Section 1 (Current State)

**Implementation Timeline**
- REQUIREMENTS.md Section 8
- QUICK_REFERENCE.md "Implementation Phases"

---

## Key Acronyms & Terms

| Term | Definition |
|------|-----------|
| Fourier | Frequency-domain basis: `A·cos(fx·x + fy·y + ϕ)` |
| RBF | Radial Basis Function (Gaussian): `A·exp(-(r²/(2σ²)))` |
| Hybrid | Mixed objectives with both Fourier and RBF components |
| Component | Single objective function term (either Fourier or RBF) |
| ComponentMode | Enum selector (FOURIER \| RBF \| HYBRID) for runtime dispatch |
| Bank | Decoded objective parameters (continuous values ready for evaluation) |
| Key | Discrete/indexed objective representation (what gets cached/serialized) |
| prior_mode | Field in ScoreΠDist selecting operating mode |
| HybridCfg | Configuration bundle containing both fourier_cfg and rbf_cfg |

---

## Related Documents in Repository

- `HYBRID_ARCHITECTURE_REQUIREMENTS.md` - This document's primary specification
- `HYBRID_ARCHITECTURE_ANALYSIS.md` - Technical analysis and downstream impacts
- `HYBRID_ARCHITECTURE_QUICK_REFERENCE.md` - Quick lookup and checklists
- `examples/ablations/iq_sips_ablation.jl` - Example pipeline (reference implementation)

---

## Questions & Troubleshooting

### "How do I start?"
→ Read REQUIREMENTS.md Sections 1-2, then start Phase 1 in Section 8

### "What changes are backward compatible?"
→ Check REQUIREMENTS.md Section 7 or QUICK_REFERENCE.md "Backward Compatibility"

### "Which function do I need to modify?"
→ Use ANALYSIS.md Section 2 "Downstream Impact Analysis" to find impact radius

### "What's the key format for [mode]?"
→ Check QUICK_REFERENCE.md "Key Encoding Formats"

### "How do I add mode dispatch to a function?"
→ See ANALYSIS.md Section 4 "Function Modification Strategy" + QUICK_REFERENCE.md "Dispatch Pattern"

### "What are trace paths for [mode]?"
→ Check QUICK_REFERENCE.md "Particle Filter Trace Paths"

### "How long will implementation take?"
→ See QUICK_REFERENCE.md "Implementation Phases" or REQUIREMENTS.md Section 8

### "Will existing Fourier code break?"
→ No. See REQUIREMENTS.md Section 7: "Backward Compatibility"

---

## Design Principles (Recap)

1. **Composability:** Objectives are sums of independent components
2. **Modal Flexibility:** Runtime selection via prior_mode enum
3. **Backward Compatible:** Fourier mode is default; existing code unchanged
4. **Type Safe:** Component types explicit; no runtime format confusion
5. **Extensible:** New modal types can be added in future phases
6. **Unified Interface:** All fields produce (x,y)→Float64; downstream uniform
7. **Debuggable:** Component types visible for diagnostics

---

## Success Criteria

- ✅ All three documents complete and self-contained
- ✅ Zero ambiguity on specifications
- ✅ Clear implementation path with phases
- ✅ Backward compatibility guaranteed
- ✅ Examples provided for all patterns
- ✅ Quick reference for fast lookup
- ✅ Ready to start Phase 1

---

**Status:** ✅ Architecture scaffold complete and ready for implementation  
**Next Step:** Begin Phase 1 implementation (Type System)  
**Reference:** Start with HYBRID_ARCHITECTURE_REQUIREMENTS.md

