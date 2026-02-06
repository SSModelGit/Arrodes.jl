export eval_ablation_mdp, eval_ablation_from_cache, run_ablation_suite, ablation_main

export reconstruct_mdp_from_cache, generate_and_cache_ablation_data, load_ablation_cache

"""
BSON payload structure:
cache = Dict(
  :meta => ...,
  :muenv_spec => MuEnvSpec(...),
  :records => Vector{Dict} with per-objective:
      id, sweep, level,
      cfg (FourierDiscreteCfg serialized ok),
      key (Tuple K, fx_i, fy_i, A_i, ϕ_i),
      agent_params_core (Dict without :menv / :start_state),
      skeleton_ref,
      full_data (Dict{Symbol,Matrix}),
      anon_data (Dict{Symbol,Matrix})
)
"""

function reconstruct_mdp_from_cache(rec::Dict, muenv_spec::MuEnvSpec)
    cfg = rec[:cfg]
    key = rec[:key]

    bank = decode_fourier_key(key, cfg)                 # returns bank with fx, fy, A, ϕ
    field = make_fourier_scalar_field(bank; scaleQ=true)
    obj = make_pomdp_objective_from_field(field)

    menv = build_shared_menv(muenv_spec)

    agent_params = deepcopy(rec[:agent_params_core])
    agent_params[:menv]  = menv
    agent_params[:goals] = Any[]
    # Start state should be consistent and avoid BSON-loaded mdp.menv:
    x0 = agent_params[:start]
    agent_params[:start_state] = KAgentState(x0, [predict_env(menv, x0)], Matrix[])

    mdp = build_kagent_pomdp(agent_params, obj; name="abl_$(rec[:id])")
    return mdp, agent_params
end

"""
    build_ablation_objectives(; rng=..., base_cfg=FourierDiscreteCfg(), levels=10)

Creates 30 objectives total:
- sweep=:K (10 objs): K in [1..10], with narrow freq/amp ranges
- sweep=:freq_range (10 objs): Fmax_i increases, K fixed at 2
- sweep=:amp_range (10 objs): Amax_i increases, K fixed at 2

Returns Vector of NamedTuples with fields:
(id, sweep, level, cfg, key, field, obj)
"""
function build_ablation_objectives(; rng=Random.default_rng(),
                                   base_cfg::FourierDiscreteCfg=FourierDiscreteCfg(),
                                   levels::Int=10)

    out = NamedTuple[]

    # Sweep A: number of features K (keep freq/amp “similar”: small ranges)
    # Choose narrow supports by using small Fmax_i and small Amax_i.
    cfgK = FourierDiscreteCfg(; Kmax=10,
                             λK=base_cfg.λK,
                             Δf=base_cfg.Δf, Fmax_i=3, freq_mag_decay=0.0,
                             ΔA=base_cfg.ΔA, Amax_i=1,
                             P=base_cfg.P)

    for i in 1:levels
        K = i  # 1..10
        key = sample_fourier_key(cfgK; K_override=K, rng=rng)
        # decode indices -> actual values (fx, fy, A, ϕ)
        ff = decode_fourier_key(key, cfgK)
        field = make_fourier_scalar_field(ff; scaleQ=true)
        obj   = make_pomdp_objective_from_field(field)
        push!(out, (id=length(out)+1, sweep=:K, level=K, cfg=cfgK, key=key, field=field, obj=obj))
    end

    # Sweep B: frequency range (keep K=2, amplitude range fixed)
    # “Similar freq values” -> small Fmax_i; “very different” -> large Fmax_i.
    cfgF_base = FourierDiscreteCfg(; Kmax=10,
                                  λK=base_cfg.λK,
                                  Δf=base_cfg.Δf,
                                  Fmax_i=3, freq_mag_decay=0.0,
                                  ΔA=base_cfg.ΔA, Amax_i=base_cfg.Amax_i,  # keep amplitude range fixed
                                  P=base_cfg.P)

    F_levels = round.(Int, range(2, 30; length=levels))  # monotone increase
    for Fmax in F_levels
        cfgF = FourierDiscreteCfg(; Kmax=cfgF_base.Kmax, λK=cfgF_base.λK,
                                 Δf=cfgF_base.Δf, Fmax_i=Fmax, freq_mag_decay=cfgF_base.freq_mag_decay,
                                 ΔA=cfgF_base.ΔA, Amax_i=cfgF_base.Amax_i,
                                 P=cfgF_base.P)
        key = sample_fourier_key(cfgF; K_override=2, rng=rng)
        ff = decode_fourier_key(key, cfgF)
        field = make_fourier_scalar_field(ff; scaleQ=true)
        obj   = make_pomdp_objective_from_field(field)
        push!(out, (id=length(out)+1, sweep=:freq_range, level=Fmax, cfg=cfgF, key=key, field=field, obj=obj))
    end

    # Sweep C: amplitude range (keep K=2, frequency range fixed)
    cfgA_base = FourierDiscreteCfg(; Kmax=10,
                                  λK=base_cfg.λK,
                                  Δf=base_cfg.Δf, Fmax_i=base_cfg.Fmax_i, freq_mag_decay=base_cfg.freq_mag_decay,
                                  ΔA=base_cfg.ΔA, Amax_i=3,
                                  P=base_cfg.P)

    A_levels = round.(Int, range(2, 50; length=levels))
    for Amax in A_levels
        cfgA = FourierDiscreteCfg(; Kmax=cfgA_base.Kmax, λK=cfgA_base.λK,
                                 Δf=cfgA_base.Δf, Fmax_i=cfgA_base.Fmax_i, freq_mag_decay=cfgA_base.freq_mag_decay,
                                 ΔA=cfgA_base.ΔA, Amax_i=Amax,
                                 P=cfgA_base.P)
        key = sample_fourier_key(cfgA; K_override=2, rng=rng)
        ff = decode_fourier_key(key, cfgA)
        field = make_fourier_scalar_field(ff; scaleQ=true)
        obj   = make_pomdp_objective_from_field(field)
        push!(out, (id=length(out)+1, sweep=:amp_range, level=Amax, cfg=cfgA, key=key, field=field, obj=obj))
    end

    return out
end

############################
# 3) Synthesize 30 MDPs from skeletons + shared MuEnv + empty goals
############################

"""
    synthesize_ablation_mdps(skeleton_packs, objectives; shared_menv=build_shared_menv(), rng=...)

For each objective:
- sample one skeleton pack at random from the 25
- extract agent_params_from_mdp(skeleton.mdp)
- override :menv and :goals
- build_kagent_pomdp(agent_params, obj)

Returns Vector of NamedTuples:
(id, sweep, level, mdp, agent_params, skeleton_ref, objrec)
"""
function synthesize_ablation_mdps(skeleton_packs::Vector{RunPack},
                                  objectives::Vector{<:NamedTuple};
                                  shared_menv=build_shared_menv(),
                                  rng=Random.default_rng())

    out = NamedTuple[]
    for objrec in objectives
        sk = rand(rng, skeleton_packs)
        agent_params = agent_params_from_mdp(sk.mdp)

        agent_params[:menv]  = shared_menv
        agent_params[:goals] = Any[]

        mdp_new = build_kagent_pomdp(agent_params, objrec.obj; name="abl_$(objrec.id)")

        push!(out, (id=objrec.id,
                    sweep=objrec.sweep,
                    level=objrec.level,
                    mdp=mdp_new,
                    agent_params=agent_params,
                    skeleton_ref=(run_id=sk.run_id, agent=sk.agent, inst=sk.inst, num_obstacles=sk.ann.num_obstacles),
                    objrec=objrec))
    end
    return out
end

function generate_and_cache_ablation_data(bson_path::String;
                                          cache_path::String,
                                          rng::AbstractRNG,
                                          shared_muenv_spec::MuEnvSpec=MuEnvSpec(),
                                          nbins::Int=5, per_bin::Int=5,
                                          levels::Int=10,
                                          T::Int=20)

    packs_all, skeletons, bin_info = select_skeleton_mdps(bson_path; nbins=nbins, per_bin=per_bin, rng=rng)

    objectives = build_ablation_objectives(; rng=rng, levels=levels)

    # IMPORTANT: do NOT call agent_params_from_mdp in a way that touches BSON-loaded mdp.menv
    mdprecs = synthesize_ablation_mdps(skeletons, objectives;
                                       shared_menv=build_shared_menv(shared_muenv_spec),
                                       rng=rng)

    records = Vector{Dict}(undef, length(mdprecs))

    for (i, rec) in enumerate(mdprecs)
        mdp = rec.mdp

        # Train SoftQ for generation
        _, π_softq = softq_policy(mdp; N=2000, epochs=2, batch_size=256)

        temperature = get(rec.agent_params, :policy_temperature, 2.0)
        full_buf = rollout_experience_buffer(mdp, π_softq; T=T, temperature=temperature, rng=rng)

        # Create anon_buf by copying data with Dict{Symbol,Matrix} typing
        full_data = full_buf.data
        anon_data = Dict{Symbol, Matrix}(k => copy(v) for (k,v) in full_data)
        # zero out first two rows of :s and :sp
        anon_data[:s][1:2, :] .= 0.0
        anon_data[:sp][1:2, :] .= 0.0

        # Store a BSON-safe “core” agent_params (NO :menv and NO :start_state)
        ap = deepcopy(rec.agent_params)
        pop!(ap, :menv, nothing)
        pop!(ap, :start_state, nothing)

        records[i] = Dict(
            :id => rec.id,
            :sweep => rec.sweep,
            :level => rec.level,
            :cfg => rec.objrec.cfg,
            :key => rec.objrec.key,
            :agent_params_core => ap,
            :skeleton_ref => rec.skeleton_ref,
            :full_data => Dict{Symbol, Matrix}(k => copy(v) for (k,v) in full_data),
            :anon_data => anon_data,
        )
    end

    cache = Dict(
        :meta => Dict(
            :source_bson => bson_path,
            :nbins => nbins, :per_bin => per_bin,
            :n_skeletons => length(skeletons),
            :n_objectives => length(objectives),
            :T => T,
            :bin_info => bin_info,
        ),
        :muenv_spec => shared_muenv_spec,
        :records => records,
    )

    BSON.@save cache_path cache
    return cache
end

function load_ablation_cache(cache_path::String)
    d = BSON.load(cache_path)
    @assert haskey(d, :cache) "Expected BSON to contain key :cache"
    return d[:cache]
end

"""
    eval_ablation_mdp(rec; n_particles=..., iql_gridN=..., minN=20, ...)

rec is one element from synthesize_ablation_mdps output.
Returns NamedTuple with all metrics for modeA and modeB plus identifiers.
"""
function eval_ablation_mdp(rec; n_particles::Int=50, ess_thresh::Float64=0.7, refine_every::Int=5, refine_topk::Int=5,
                           iql_gridN::Int=100, minN::Int=20, gridsize::Int=120, rng=Random.default_rng())

    mdp = rec.mdp

    # 1) Train SoftQ for data generation (Mode A “real” dataset)
    _, π_softq = softq_policy(mdp; N=2000, epochs=2, batch_size=256)

    # 2) Generate experience (full + anon identical here)
    temperature = get(rec.agent_params, :policy_temperature, 2.0)
    full_buf = rollout_experience_buffer(mdp, π_softq; T=minN, temperature=temperature, rng=rng)

    anon_data = Dict{Symbol,Matrix}(k => copy(v) for (k,v) in full_buf.data)
    anon_buf  = ExperienceBuffer(anon_data, size(anon_data[:s], 2), 1, Array{Int64}[], nothing, 0)
    anonymize_buffer_location!(anon_buf)

    # 3) Train IQL (Mode B surrogate driver)
    π_iql, 𝒟_iql, _ = quick_IQL(mdp, anon_buf)  # uses helper

    # 4) Build π_dist with action mappings
    as = actions(mdp)
    action_list = [as, a->Flux.onehot(a, as), Flux.onehotbatch(as, as)]
    π_dist = ScoreΠDist(; mdp_params=action_list)

    # 5) Mode A PF
    T = size(full_buf.data[:s], 2)
    data_slices = (T ≤ minN) ? collect(1:T) : [rand(rng, 1:T) for _ in 1:minN]
    state_dataA = full_buf.data[:s][:, data_slices]
    obs_aidxA   = onehot_cols_to_aidx(full_buf.data[:a][:, data_slices])

    pfA = particle_filter(obs_aidxA, π_dist, rec.agent_params, state_dataA, n_particles;
                          ess_thresh=ess_thresh, refine_every=refine_every, refine_topk=refine_topk)

    # 6) Mode B PF (IQL grid surrogate)
    iql_state_data, iql_obs_aidx, _ = surrogate_dataset_from_iql_grid(π_dist, π_iql, mdp; eval_num=iql_gridN)

    pfB = particle_filter(iql_obs_aidx, π_dist, rec.agent_params, iql_state_data, n_particles;
                          ess_thresh=ess_thresh, refine_every=refine_every, refine_topk=refine_topk)

    # 7) Metrics for both modes
    degA = pf_degeneracy(pfA, π_dist; n_particles=n_particles)
    objA = objective_recon_metrics(pfA, π_dist, mdp; gridsize=gridsize)
    polA = policy_match_acc(pfA, π_dist, rec.agent_params, state_dataA, obs_aidxA)

    degB = pf_degeneracy(pfB, π_dist; n_particles=n_particles)
    objB = objective_recon_metrics(pfB, π_dist, mdp; gridsize=gridsize)
    polB = policy_match_acc(pfB, π_dist, rec.agent_params, iql_state_data, iql_obs_aidx)

    return (
        id=rec.id, sweep=rec.sweep, level=rec.level,
        skeleton_ref=rec.skeleton_ref,
        # Mode A:
        A=(deg=degA, obj=objA, pol=polA),
        # Mode B:
        B=(deg=degB, obj=objB, pol=polB),
    )
end


"""
Run PF + metrics only, using cached buffers.
This reruns quick_IQL (Mode B) from anon_data, but avoids regenerating the trajectories.
"""
function eval_ablation_from_cache(cache::Dict;
                                  n_particles::Int=50,
                                  ess_thresh::Float64=0.7,
                                  refine_every::Int=5,
                                  refine_topk::Int=5,
                                  iql_gridN::Int=120,
                                  gridsize::Int=120,
                                  ess_min_frac::Float64=0.25,   # NEW
                                  rng::AbstractRNG=Random.default_rng())

    muenv_spec = cache[:muenv_spec]
    records = cache[:records]

    evals = Vector{Any}(undef, length(records))
    ess_min = ess_min_frac * n_particles

    for (i, rec) in enumerate(records)
        mdp, agent_params = reconstruct_mdp_from_cache(rec, muenv_spec)

        full_data = Dict{Symbol, Matrix}(rec[:full_data])
        anon_data = Dict{Symbol, Matrix}(rec[:anon_data])

        full_buf = ExperienceBuffer(full_data, size(full_data[:s],2), 1, Array{Int64}[], nothing, 0)
        anon_buf = ExperienceBuffer(anon_data, size(anon_data[:s],2), 1, Array{Int64}[], nothing, 0)

        π_iql, 𝒮_iql, _ = quick_IQL(mdp, anon_buf)

        as = actions(mdp)
        action_list = [as, a->Flux.onehot(a, as), Flux.onehotbatch(as, as)]
        π_dist = ScoreΠDist(; mdp_params=action_list)

        # Mode A PF inputs
        state_dataA = full_buf.data[:s]
        obs_aidxA   = onehot_cols_to_aidx(full_buf.data[:a])
        lobs = Int64(length(obs_aidxA) * 0.1)

        pfA = particle_filter(obs_aidxA[1:lobs], π_dist, agent_params, state_dataA[:, 1:lobs], n_particles;
                              ess_thresh=ess_thresh, refine_every=refine_every, refine_topk=refine_topk)

        # Mode B PF inputs: TODO!!!
        # iql_state_data, iql_obs_aidx, _ = surrogate_dataset_from_iql_grid(π_dist, π_iql, mdp; eval_num=iql_gridN)

        pfB = particle_filter(obs_aidxA, π_dist, agent_params, state_dataA, n_particles*3;
                              ess_thresh=ess_thresh, refine_every=refine_every, refine_topk=refine_topk)

        # Degeneracy first
        degA = pf_degeneracy(pfA, π_dist; n_particles=n_particles)
        degB = pf_degeneracy(pfB, π_dist; n_particles=n_particles)

        badA = degA.collapsed || (degA.ess < ess_min)
        badB = degB.collapsed || (degB.ess < ess_min)

        # Only compute other metrics if not degenerate; else NaN them
        objA = badA ? (rmse_z=NaN, corr=NaN) : objective_recon_metrics(pfA, π_dist, mdp; gridsize=gridsize)
        polA = badA ? (acc=NaN,)            : policy_match_acc(pfA, π_dist, agent_params, state_dataA, obs_aidxA)

        objB = badB ? (rmse_z=NaN, corr=NaN) : objective_recon_metrics(pfB, π_dist, mdp; gridsize=gridsize)
        polB = badB ? (acc=NaN,)             : policy_match_acc(pfB, π_dist, agent_params, state_dataA, obs_aidxA)

        keyA, probA = badA ? (nothing, NaN) : top_key(pfA, π_dist)
        keyB, probB = badB ? (nothing, NaN) : top_key(pfB, π_dist)


        evals[i] = (
            id=rec[:id], sweep=rec[:sweep], level=rec[:level],
            skeleton_ref=rec[:skeleton_ref],
            A=(deg=degA, bad=badA, obj=objA, pol=polA, top_key=keyA, top_prob=probA),
            B=(deg=degB, bad=badB, obj=objB, pol=polB, top_key=keyB, top_prob=probB),
        )
    end

    return evals
end

############################
# 6) Run full ablation + aggregate + plots
############################

"""
    run_ablation_suite(bson_path; ...)

End-to-end:
1) select 25 skeletons from bins
2) build 30 objectives
3) synthesize 30 MDPs
4) eval each (Mode A vs Mode B metrics)
Returns:
- meta info
- eval records (vector)
- grouped summaries
"""
function run_ablation_suite(bson_path::AbstractString;
                            nbins::Int=5,
                            per_bin::Int=5,
                            rng=Random.default_rng(),
                            shared_menv=build_shared_menv(),
                            n_particles::Int=50,
                            ess_thresh::Float64=0.7,
                            refine_every::Int=5,
                            refine_topk::Int=5,
                            iql_gridN::Int=120,
                            minN::Int=20,
                            gridsize::Int=120)

    packs_all, skeletons, bin_info = select_skeleton_mdps(bson_path; nbins=nbins, per_bin=per_bin, rng=rng)

    objectives = build_ablation_objectives(; rng=rng, levels=10)
    mdprecs = synthesize_ablation_mdps(skeletons, objectives; shared_menv=shared_menv, rng=rng)

    evals = Vector{Any}(undef, length(mdprecs))
    for (i, rec) in enumerate(mdprecs)
        evals[i] = eval_ablation_mdp(rec;
                                     n_particles=n_particles,
                                     ess_thresh=ess_thresh,
                                     refine_every=refine_every,
                                     refine_topk=refine_topk,
                                     iql_gridN=iql_gridN,
                                     minN=minN,
                                     gridsize=gridsize,
                                     rng=rng)
    end

    return (meta=(bin_info=bin_info,
                  n_skeletons=length(skeletons),
                  n_objectives=length(objectives),
                  n_mdps=length(mdprecs)),
            evals=evals)
end

function eval_all(packs::Vector{RunPack}; max_tests::Int=1000,
                  kwargs...)
    packs2 = diversify_packs(packs)
    N = min(length(packs2), max_tests)
    out = Vector{Any}(undef, N)
    for i in 1:N
        out[i] = eval_pack(packs2[i]; kwargs...)
    end
    return out
end

function multi_run_test(bson_path::AbstractString;
                        max_tests::Int=1000,
                        n_particles::Int=50,
                        ess_thresh::Float64=0.7,
                        refine_every::Int=5,
                        refine_topk::Int=5,
                        minN::Int=20,
                        iql_gridN::Int=80,
                        gridsize::Int=120)

    packs = load_runpacks(bson_path)
    evals = eval_all(packs;
                     max_tests=max_tests,
                     n_particles=n_particles,
                     ess_thresh=ess_thresh,
                     refine_every=refine_every,
                     refine_topk=refine_topk,
                     minN=minN,
                     iql_gridN=iql_gridN)

    return summarize_eval(evals; n_particles=n_particles, gridsize=gridsize)
end

"""
ablation_main:
- mode=:generate  -> generate buffers, save cache, then evaluate from cache
- mode=:load      -> load cache and evaluate only
"""
function ablation_main(bson_path::String;
                       script_dir::String,
                       mode::Symbol = :generate,
                       cache_filename::String = "ablation_cache.bson",
                       rng::AbstractRNG = Random.default_rng(),
                       shared_muenv_spec::MuEnvSpec = MuEnvSpec(),
                       n_particles::Int=50,
                       minN::Int=20,           # still used by other paths if needed
                       iql_gridN::Int=120,
                       gridsize::Int=120)

    cache_path = joinpath(script_dir, cache_filename)

    cache = if mode == :generate
        generate_and_cache_ablation_data(bson_path;
            cache_path=cache_path,
            rng=rng,
            shared_muenv_spec=shared_muenv_spec,
            T=minN
        )
    elseif mode == :load
        load_ablation_cache(cache_path)
    else
        error("Unknown mode=$mode (use :generate or :load)")
    end

    evals = eval_ablation_from_cache(cache;
        n_particles=n_particles,
        iql_gridN=iql_gridN,
        gridsize=gridsize,
        rng=rng
    )

    sums = summarize_ablation(evals)

    out = (
        cache_path = cache_path,
        cache = cache,
        evals = evals,
        summaries = sums,
        meta = Dict(:n_particles => n_particles, :iql_gridN => iql_gridN, :gridsize => gridsize)
    )
    # Save the entire out wholesale
    BSON.@save joinpath(script_dir, "ablation_out_wholesale.bson") out

    return out
end