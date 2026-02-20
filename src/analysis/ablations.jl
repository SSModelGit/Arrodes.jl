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
    field = make_fourier_scalar_field(bank; scaleQ = true)
    obj = make_pomdp_objective_from_field(field)

    menv = Utils.build_shared_menv(muenv_spec)

    agent_params = deepcopy(rec[:agent_params_core])
    agent_params[:menv] = menv
    agent_params[:goals] = Any[]
    # Start state should be consistent and avoid BSON-loaded mdp.menv:
    x0 = agent_params[:start]
    agent_params[:start_state] = KAgentState(x0, [predict_env(menv, x0)], Matrix[])

    mdp = Utils.build_kagent_pomdp(agent_params, obj; name = "abl_$(rec[:id])")
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
function build_ablation_objectives(;
    rng = Random.default_rng(),
    base_cfg::FourierDiscreteCfg = FourierDiscreteCfg(),
    levels::Int = 10,
)

    out = NamedTuple[]

    # Sweep A: number of features K (keep freq/amp “similar”: small ranges)
    # Choose narrow supports by using small Fmax_i and small Amax_i.
    cfgK = FourierDiscreteCfg(;
        Kmax = 10,
        λK = base_cfg.λK,
        Δf = base_cfg.Δf,
        Fmax_i = 3,
        freq_mag_decay = 0.0,
        ΔA = base_cfg.ΔA,
        Amax_i = 1,
        P = base_cfg.P,
    )

    for i in 1:levels
        K = i  # 1..10
        key = sample_fourier_key(cfgK; K_override = K, rng = rng)
        # decode indices -> actual values (fx, fy, A, ϕ)
        ff = decode_fourier_key(key, cfgK)
        field = make_fourier_scalar_field(ff; scaleQ = true)
        obj = make_pomdp_objective_from_field(field)
        push!(
            out,
            (
                id = length(out) + 1,
                sweep = :K,
                level = K,
                cfg = cfgK,
                key = key,
                field = field,
                obj = obj,
            ),
        )
    end

    # Sweep B: frequency range (keep K=2, amplitude range fixed)
    # “Similar freq values” -> small Fmax_i; “very different” -> large Fmax_i.
    cfgF_base = FourierDiscreteCfg(;
        Kmax = 10,
        λK = base_cfg.λK,
        Δf = base_cfg.Δf,
        Fmax_i = 3,
        freq_mag_decay = 0.0,
        ΔA = base_cfg.ΔA,
        Amax_i = base_cfg.Amax_i,  # keep amplitude range fixed
        P = base_cfg.P,
    )

    F_levels = round.(Int, range(2, 30; length = levels))  # monotone increase
    for Fmax in F_levels
        cfgF = FourierDiscreteCfg(;
            Kmax = cfgF_base.Kmax,
            λK = cfgF_base.λK,
            Δf = cfgF_base.Δf,
            Fmax_i = Fmax,
            freq_mag_decay = cfgF_base.freq_mag_decay,
            ΔA = cfgF_base.ΔA,
            Amax_i = cfgF_base.Amax_i,
            P = cfgF_base.P,
        )
        key = sample_fourier_key(cfgF; K_override = 2, rng = rng)
        ff = decode_fourier_key(key, cfgF)
        field = make_fourier_scalar_field(ff; scaleQ = true)
        obj = make_pomdp_objective_from_field(field)
        push!(
            out,
            (
                id = length(out) + 1,
                sweep = :freq_range,
                level = Fmax,
                cfg = cfgF,
                key = key,
                field = field,
                obj = obj,
            ),
        )
    end

    # Sweep C: amplitude range (keep K=2, frequency range fixed)
    cfgA_base = FourierDiscreteCfg(;
        Kmax = 10,
        λK = base_cfg.λK,
        Δf = base_cfg.Δf,
        Fmax_i = base_cfg.Fmax_i,
        freq_mag_decay = base_cfg.freq_mag_decay,
        ΔA = base_cfg.ΔA,
        Amax_i = 3,
        P = base_cfg.P,
    )

    A_levels = round.(Int, range(2, 50; length = levels))
    for Amax in A_levels
        cfgA = FourierDiscreteCfg(;
            Kmax = cfgA_base.Kmax,
            λK = cfgA_base.λK,
            Δf = cfgA_base.Δf,
            Fmax_i = cfgA_base.Fmax_i,
            freq_mag_decay = cfgA_base.freq_mag_decay,
            ΔA = cfgA_base.ΔA,
            Amax_i = Amax,
            P = cfgA_base.P,
        )
        key = sample_fourier_key(cfgA; K_override = 2, rng = rng)
        ff = decode_fourier_key(key, cfgA)
        field = make_fourier_scalar_field(ff; scaleQ = true)
        obj = make_pomdp_objective_from_field(field)
        push!(
            out,
            (
                id = length(out) + 1,
                sweep = :amp_range,
                level = Amax,
                cfg = cfgA,
                key = key,
                field = field,
                obj = obj,
            ),
        )
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
function synthesize_ablation_mdps(
    skeleton_packs::Vector{RunPack},
    objectives::Vector{<:NamedTuple};
    shared_menv = Utils.build_shared_menv(),
    rng = Random.default_rng(),
)

    out = NamedTuple[]
    for objrec in objectives
        sk = rand(rng, skeleton_packs)
        agent_params = Utils.agent_params_from_mdp(sk.mdp)

        agent_params[:menv] = shared_menv
        agent_params[:goals] = Any[]

        mdp_new =
            Utils.build_kagent_pomdp(agent_params, objrec.obj; name = "abl_$(objrec.id)")

        push!(
            out,
            (
                id = objrec.id,
                sweep = objrec.sweep,
                level = objrec.level,
                mdp = mdp_new,
                agent_params = agent_params,
                skeleton_ref = (
                    run_id = sk.run_id,
                    agent = sk.agent,
                    inst = sk.inst,
                    num_obstacles = sk.ann.num_obstacles,
                ),
                objrec = objrec,
            ),
        )
    end
    return out
end

function generate_and_cache_ablation_data(
    meta_or_path::Union{AbstractString, Dict{String, Any}};
    cache_path::String,
    rng::AbstractRNG,
    shared_muenv_spec::MuEnvSpec = MuEnvSpec(),
    nbins::Int = 5,
    per_bin::Int = 5,
    levels::Int = 10,
    T::Int = 20,
)

    # Resolve source metadata to get paths for cache metadata
    source_meta =
        isa(meta_or_path, AbstractString) ? Utils.read_data_metadata(meta_or_path) :
        meta_or_path
    source_data_path = get(source_meta, "data_path", nothing)
    @argcheck !isnothing(source_data_path) "Source metadata is missing 'data_path' field."
    source_data_type = get(source_meta, "data_type", nothing)
    @argcheck !isnothing(source_data_type) "Source metadata is missing 'data_type' field."
    source_meta_path = Utils.infer_metadata_path(source_data_path)

    packs_all, skeletons, bin_info =
        Utils.select_skeleton_mdps(meta_or_path; nbins = nbins, per_bin = per_bin, rng = rng)

    objectives = build_ablation_objectives(; rng = rng, levels = levels)

    # IMPORTANT: do NOT call agent_params_from_mdp in a way that touches BSON-loaded mdp.menv
    mdprecs = synthesize_ablation_mdps(
        skeletons,
        objectives;
        shared_menv = Utils.build_shared_menv(shared_muenv_spec),
        rng = rng,
    )

    records = Vector{Dict}(undef, length(mdprecs))

    for (i, rec) in enumerate(mdprecs)
        mdp = rec.mdp

        # Train SoftQ for generation
        _, π_softq = RL.softq_policy(mdp; N = 2000, epochs = 2, batch_size = 256)

        temperature = get(rec.agent_params, :policy_temperature, 2.0)
        full_buf = RL.rollout_experience_buffer(
            mdp,
            π_softq;
            T = T,
            temperature = temperature,
            rng = rng,
        )

        # Create anon_buf by copying data with Dict{Symbol,Matrix} typing
        full_data = full_buf.data
        anon_data = Dict{Symbol, Matrix}(k => copy(v) for (k, v) in full_data)
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
            :full_data => Dict{Symbol, Matrix}(k => copy(v) for (k, v) in full_data),
            :anon_data => anon_data,
        )
    end

    cache = Dict(
        :meta => Dict(
            :source_bson => source_data_path,
            :nbins => nbins,
            :per_bin => per_bin,
            :n_skeletons => length(skeletons),
            :n_objectives => length(objectives),
            :T => T,
            :bin_info => bin_info,
        ),
        :muenv_spec => shared_muenv_spec,
        :records => records,
    )

    BSON.@save cache_path cache

    # Generate cache metadata file
    cache_meta_path = Utils.write_cache_metadata(
        cache_path;
        n_skeletons = length(skeletons),
        n_objectives = length(objectives),
        T = T,
        nbins = nbins,
        per_bin = per_bin,
        source_data_type = source_data_type,
        source_data_path = source_data_path,
        source_data_meta_path = source_meta_path,
    )

    return cache
end

"""
    load_ablation_cache(cache_meta_path::String)

Load ablation cache data from a cache metadata file.

Given a path to cache metadata (e.g., "ablation_cache_20260219.meta.toml"),
infers the corresponding cache BSON path using the naming convention,
loads the metadata to validate, and returns the cache Dict.

# Arguments:
- `cache_meta_path`: path to the cache metadata file

# Returns:
- Cache Dict with keys :muenv_spec, :records, :meta
"""
function load_ablation_cache(cache_meta_path::String)
    # Infer cache BSON path from metadata path
    @argcheck endswith(cache_meta_path, ".meta.toml") "Cache metadata path must end with .meta.toml; got: $cache_meta_path"
    meta_abs = abspath(cache_meta_path)
    cache_path = meta_abs[1:end-10] * ".bson"  # Strip .meta.toml, add .bson

    # Load and validate metadata
    meta = Utils.read_data_metadata(meta_abs)
    @argcheck meta["data_type"] == "ablation_cache" "Expected data_type=ablation_cache in metadata"

    # Load the cache BSON
    d = BSON.load(cache_path)
    @assert haskey(d, :cache) "Expected BSON to contain key :cache"
    return d[:cache]
end

"""
    eval_ablation_mdp(rec; n_particles=..., iql_gridN=..., minN=20, ...)

rec is one element from synthesize_ablation_mdps output.
Returns NamedTuple with all metrics for modeA and modeB plus identifiers.
"""
function eval_ablation_mdp(
    rec;
    n_particles::Int = 50,
    ess_thresh::Float64 = 0.7,
    refine_every::Int = 5,
    refine_topk::Int = 5,
    iql_gridN::Int = 100,
    minN::Int = 20,
    gridsize::Int = 120,
    rng = Random.default_rng(),
)

    mdp = rec.mdp

    # 1) Train SoftQ for data generation (Mode A “real” dataset)
    _, π_softq = RL.softq_policy(mdp; N = 2000, epochs = 2, batch_size = 256)

    # 2) Generate experience (full + anon identical here)
    temperature = get(rec.agent_params, :policy_temperature, 2.0)
    full_buf = RL.rollout_experience_buffer(
        mdp,
        π_softq;
        T = minN,
        temperature = temperature,
        rng = rng,
    )

    anon_data = Dict{Symbol, Matrix}(k => copy(v) for (k, v) in full_buf.data)
    anon_buf =
        ExperienceBuffer(anon_data, size(anon_data[:s], 2), 1, Array{Int64}[], nothing, 0)
    anonymize_buffer_location!(anon_buf)

    # 3) Train IQL (Mode B surrogate driver)
    π_iql, 𝒟_iql, _ = RL.quick_IQL(mdp, anon_buf)  # uses helper

    # 4) Build π_dist with action mappings
    as = actions(mdp)
    action_list = [as, a -> Flux.onehot(a, as), Flux.onehotbatch(as, as)]
    π_dist = ScoreΠDist(; mdp_params = action_list)

    # 5) Mode A PF
    T = size(full_buf.data[:s], 2)
    data_slices = (T ≤ minN) ? collect(1:T) : [rand(rng, 1:T) for _ in 1:minN]
    state_dataA = full_buf.data[:s][:, data_slices]
    obs_aidxA = Utils.onehot_cols_to_aidx(full_buf.data[:a][:, data_slices])

    pfA = Inference.particle_filter(
        obs_aidxA,
        π_dist,
        rec.agent_params,
        state_dataA,
        n_particles;
        ess_thresh = ess_thresh,
        refine_every = refine_every,
        refine_topk = refine_topk,
    )

    # 6) Mode B PF (IQL grid surrogate)
    iql_state_data, iql_obs_aidx, _ =
        RL.surrogate_dataset_from_iql_grid(π_dist, π_iql, mdp; eval_num = iql_gridN)

    pfB = Inference.particle_filter(
        iql_obs_aidx,
        π_dist,
        rec.agent_params,
        iql_state_data,
        n_particles;
        ess_thresh = ess_thresh,
        refine_every = refine_every,
        refine_topk = refine_topk,
    )

    # 7) Metrics for both modes
    degA = pf_degeneracy(pfA, π_dist; n_particles = n_particles)
    objA = objective_recon_metrics(pfA, π_dist, mdp; gridsize = gridsize)
    polA = policy_match_acc(pfA, π_dist, rec.agent_params, state_dataA, obs_aidxA)

    degB = pf_degeneracy(pfB, π_dist; n_particles = n_particles)
    objB = objective_recon_metrics(pfB, π_dist, mdp; gridsize = gridsize)
    polB = policy_match_acc(pfB, π_dist, rec.agent_params, iql_state_data, iql_obs_aidx)

    return (
        id = rec.id,
        sweep = rec.sweep,
        level = rec.level,
        skeleton_ref = rec.skeleton_ref,
        # Mode A:
        A = (deg = degA, obj = objA, pol = polA),
        # Mode B:
        B = (deg = degB, obj = objB, pol = polB),
    )
end


"""
Run PF + metrics only, using cached buffers.
This reruns quick_IQL (Mode B) from anon_data, but avoids regenerating the trajectories.
"""
function eval_ablation_from_cache(
    cache::Dict;
    n_particles::Int = 50,
    ess_thresh::Float64 = 0.7,
    refine_every::Int = 5,
    refine_topk::Int = 5,
    iql_gridN::Int = 120,
    gridsize::Int = 120,
    ess_min_frac::Float64 = 0.25,   # NEW
    rng::AbstractRNG = Random.default_rng(),
)

    muenv_spec = cache[:muenv_spec]
    records = cache[:records]

    evals = Vector{Any}(undef, length(records))
    ess_min = ess_min_frac * n_particles

    for (i, rec) in enumerate(records)
        mdp, agent_params = reconstruct_mdp_from_cache(rec, muenv_spec)

        full_data = Dict{Symbol, Matrix}(rec[:full_data])
        anon_data = Dict{Symbol, Matrix}(rec[:anon_data])

        full_buf = ExperienceBuffer(
            full_data,
            size(full_data[:s], 2),
            1,
            Array{Int64}[],
            nothing,
            0,
        )
        anon_buf = ExperienceBuffer(
            anon_data,
            size(anon_data[:s], 2),
            1,
            Array{Int64}[],
            nothing,
            0,
        )

        π_iql, 𝒮_iql, _ = RL.quick_IQL(mdp, anon_buf)

        as = actions(mdp)
        action_list = [as, a -> Flux.onehot(a, as), Flux.onehotbatch(as, as)]
        π_dist = ScoreΠDist(; mdp_params = action_list)

        # Mode A PF inputs
        state_dataA = full_buf.data[:s]
        obs_aidxA = Utils.onehot_cols_to_aidx(full_buf.data[:a])
        lobs = Int64(length(obs_aidxA) * 0.1)

        pfA = Inference.particle_filter(
            obs_aidxA[1:lobs],
            π_dist,
            agent_params,
            state_dataA[:, 1:lobs],
            n_particles;
            ess_thresh = ess_thresh,
            refine_every = refine_every,
            refine_topk = refine_topk,
        )

        # Mode B PF inputs: TODO!!!
        # iql_state_data, iql_obs_aidx, _ = RL.surrogate_dataset_from_iql_grid(π_dist, π_iql, mdp; eval_num=iql_gridN)

        pfB = Inference.particle_filter(
            obs_aidxA,
            π_dist,
            agent_params,
            state_dataA,
            n_particles * 3;
            ess_thresh = ess_thresh,
            refine_every = refine_every,
            refine_topk = refine_topk,
        )

        # Degeneracy first
        degA = pf_degeneracy(pfA, π_dist; n_particles = n_particles)
        degB = pf_degeneracy(pfB, π_dist; n_particles = n_particles)

        badA = degA.collapsed || (degA.ess < ess_min)
        badB = degB.collapsed || (degB.ess < ess_min)

        # Only compute other metrics if not degenerate; else NaN them
        objA =
            badA ? (rmse_z = NaN, corr = NaN) :
            objective_recon_metrics(pfA, π_dist, mdp; gridsize = gridsize)
        polA =
            badA ? (acc = NaN,) :
            policy_match_acc(pfA, π_dist, agent_params, state_dataA, obs_aidxA)

        objB =
            badB ? (rmse_z = NaN, corr = NaN) :
            objective_recon_metrics(pfB, π_dist, mdp; gridsize = gridsize)
        polB =
            badB ? (acc = NaN,) :
            policy_match_acc(pfB, π_dist, agent_params, state_dataA, obs_aidxA)

        keyA, probA = badA ? (nothing, NaN) : top_key(pfA, π_dist)
        keyB, probB = badB ? (nothing, NaN) : top_key(pfB, π_dist)


        evals[i] = (
            id = rec[:id],
            sweep = rec[:sweep],
            level = rec[:level],
            skeleton_ref = rec[:skeleton_ref],
            A = (
                deg = degA,
                bad = badA,
                obj = objA,
                pol = polA,
                top_key = keyA,
                top_prob = probA,
            ),
            B = (
                deg = degB,
                bad = badB,
                obj = objB,
                pol = polB,
                top_key = keyB,
                top_prob = probB,
            ),
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
function run_ablation_suite(
    meta_or_path::Union{AbstractString, Dict{String, Any}};
    nbins::Int = 5,
    per_bin::Int = 5,
    rng = Random.default_rng(),
    shared_menv = build_shared_menv(),
    n_particles::Int = 50,
    ess_thresh::Float64 = 0.7,
    refine_every::Int = 5,
    refine_topk::Int = 5,
    iql_gridN::Int = 120,
    minN::Int = 20,
    gridsize::Int = 120,
)

    packs_all, skeletons, bin_info =
        select_skeleton_mdps(meta_or_path; nbins = nbins, per_bin = per_bin, rng = rng)

    objectives = build_ablation_objectives(; rng = rng, levels = 10)
    mdprecs =
        synthesize_ablation_mdps(skeletons, objectives; shared_menv = shared_menv, rng = rng)

    evals = Vector{Any}(undef, length(mdprecs))
    for (i, rec) in enumerate(mdprecs)
        evals[i] = eval_ablation_mdp(
            rec;
            n_particles = n_particles,
            ess_thresh = ess_thresh,
            refine_every = refine_every,
            refine_topk = refine_topk,
            iql_gridN = iql_gridN,
            minN = minN,
            gridsize = gridsize,
            rng = rng,
        )
    end

    return (
        meta = (
            bin_info = bin_info,
            n_skeletons = length(skeletons),
            n_objectives = length(objectives),
            n_mdps = length(mdprecs),
        ),
        evals = evals,
    )
end

function eval_all(packs::Vector{RunPack}; max_tests::Int = 1000, kwargs...)
    packs2 = diversify_packs(packs)
    N = min(length(packs2), max_tests)
    out = Vector{Any}(undef, N)
    for i in 1:N
        out[i] = eval_pack(packs2[i]; kwargs...)
    end
    return out
end

function multi_run_test(
    meta_or_path::Union{AbstractString, Dict{String, Any}};
    max_tests::Int = 1000,
    n_particles::Int = 50,
    ess_thresh::Float64 = 0.7,
    refine_every::Int = 5,
    refine_topk::Int = 5,
    minN::Int = 20,
    iql_gridN::Int = 80,
    gridsize::Int = 120,
)

    packs = Utils.load_runpacks(meta_or_path)
    evals = eval_all(
        packs;
        max_tests = max_tests,
        n_particles = n_particles,
        ess_thresh = ess_thresh,
        refine_every = refine_every,
        refine_topk = refine_topk,
        minN = minN,
        iql_gridN = iql_gridN,
    )

    return summarize_eval(evals; n_particles = n_particles, gridsize = gridsize)
end

"""
    ablation_main(meta_or_path; script_dir, mode, cache_metadata_filename, ...)

Orchestrate ablation generation or loading from cache metadata.

# Arguments:
- `meta_or_path`: training data metadata (path or Dict) for generating cache
- `script_dir`: directory where cache metadata will be stored
- `mode`: `:generate` to create new cache, `:load` to use existing cache
- `cache_metadata_filename`: metadata filename (e.g., "ablation_cache_20260219.meta.toml")
- Other kwargs passed to cache generation/evaluation

# Behavior:
In `:generate` mode:
  - Converts `cache_metadata_filename` → cache BSON path using naming convention
  - Calls `generate_and_cache_ablation_data` which creates both BSON and metadata
In `:load` mode:
  - Uses `cache_metadata_filename` directly to load cache via `load_ablation_cache`
"""
function ablation_main(
    meta_or_path::Union{AbstractString, Dict{String, Any}};
    script_dir::String,
    mode::Symbol = :generate,
    cache_metadata_filename::String = "ablation_cache.meta.toml",
    rng::AbstractRNG = Random.default_rng(),
    shared_muenv_spec::MuEnvSpec = MuEnvSpec(),
    n_particles::Int = 50,
    minN::Int = 20,           # still used by other paths if needed
    iql_gridN::Int = 120,
    gridsize::Int = 120,
)

    @argcheck endswith(cache_metadata_filename, ".meta.toml") "cache_metadata_filename must end with .meta.toml; got: $cache_metadata_filename"
    cache_meta_path = joinpath(script_dir, cache_metadata_filename)

    cache = if mode == :generate
        # Convert cache metadata filename to cache BSON filename
        cache_bson_filename = cache_metadata_filename[1:end-10] * ".bson"  # Replace .meta.toml with .bson
        cache_path = joinpath(script_dir, cache_bson_filename)

        generate_and_cache_ablation_data(
            meta_or_path;
            cache_path = cache_path,
            rng = rng,
            shared_muenv_spec = shared_muenv_spec,
            T = minN,
        )
    elseif mode == :load
        load_ablation_cache(cache_meta_path)
    else
        error("Unknown mode=$mode (use :generate or :load)")
    end

    evals = eval_ablation_from_cache(
        cache;
        n_particles = n_particles,
        iql_gridN = iql_gridN,
        gridsize = gridsize,
        rng = rng,
    )

    sums = summarize_ablation(evals)

    out = (
        cache_meta_path = cache_meta_path,
        cache = cache,
        evals = evals,
        summaries = sums,
        meta = Dict(
            :n_particles => n_particles,
            :iql_gridN => iql_gridN,
            :gridsize => gridsize,
        ),
    )
    # Save the entire out wholesale
    # BSON.@save joinpath(script_dir, "ablation_out_wholesale.bson") out

    return out
end


"""
    write_wholesale_metadata(wholesale_path; cache_meta_path, mode, version="1.0.0")

Write a metadata TOML file for ablation wholesale output.

Converts `wholesale_path` (ending in .bson) to a .meta.toml file with descriptive
information about the wholesale output structure, plotting labels, and data provenance.

# Arguments
- `wholesale_path`: Path to the wholesale BSON file
- `cache_meta_path`: Path to the associated cache metadata file
- `mode`: Generation mode (`:generate` or `:load`)
- `version`: Version string (default: "1.0.0")

# Returns
- Path to the created metadata file
"""
function write_wholesale_metadata(
    wholesale_path::AbstractString;
    cache_meta_path::AbstractString,
    mode::Symbol,
    version::AbstractString = "1.0.0",
)

    # Enforce naming convention: <wholesale>.bson -> <wholesale>.meta.toml
    @argcheck endswith(wholesale_path, ".bson") "Wholesale path must end with .bson; got: $wholesale_path"
    meta_path = wholesale_path[1:end-5] * ".meta.toml"

    created_at = Dates.format(Dates.now(), "yyyy-mm-ddTHH:MM:SSZ")

    # Build metadata structure
    meta_content = """
    schema_version = $(version)
    data_type = "ablation_wholesale"
    data_path = "$(wholesale_path)"
    format = "bson"
    created_at = "$(created_at)"
    created_by = "iq_sips_ablation.jl"
    notes = "Complete ablation study output with cache, evaluations, and summaries"

    [source]
    cache_meta_path = "$(cache_meta_path)"
    generation_mode = "$(String(mode))"

    [structure]
    top_level_key = "out"
    fields = ["cache_meta_path", "cache", "evals", "summaries", "meta"]

    [evals_structure]
    description = "Vector of evaluation records (one per objective) with mode A vs mode B comparisons"
    mode_A = "IQ-SIPS inference using real expert data (SoftQ rollouts)"
    mode_B = "IQ-SIPS inference using IQL-generated surrogate data"
    mode_A_label = "Real Data"
    mode_B_label = "IQL Surrogate"

    [sweeps]
    keys = ["K", "freq_range", "amp_range"]

    [sweeps.K]
    description = "Number of Fourier features"
    xlabel = "Number of Features (K)"

    [sweeps.freq_range]
    description = "Frequency range bandwidth"
    xlabel = "Frequency Range"

    [sweeps.amp_range]
    description = "Amplitude range"
    xlabel = "Amplitude Range"

    [metrics]
    keys = ["ess", "rmse", "acc"]

    [metrics.ess]
    ylabel = "Effective Sample Size"
    field_A = "essA"
    field_B = "essB"

    [metrics.rmse]
    ylabel = "RMSE"
    field_A = "rmseA"
    field_B = "rmseB"

    [metrics.acc]
    ylabel = "Policy Accuracy"
    field_A = "accA"
    field_B = "accB"
    """

    open(meta_path, "w") do io
        write(io, meta_content)
    end

    return meta_path
end