export eval_pack, summarize_eval, summarize_ablation, best_eval_by_accuracy

"""
    eval_pack(pack::RunPack;
                   n_particles::Int=50,
                   ess_thresh::Float64=0.7,
                   refine_every::Int=5,
                   refine_topk::Int=5,
                   iql_gridN::Int=100,
                   minN::Int=20)

Results are returned as NamedTuples for easy downstream processing
* (keeps code compact; no separate Result type required)
"""
function eval_pack(pack::RunPack;
                   n_particles::Int=50,
                   ess_thresh::Float64=0.7,
                   refine_every::Int=5,
                   refine_topk::Int=5,
                   iql_gridN::Int=100,
                   minN::Int=20)

    mdp = pack.mdp

    # 1) train IQL on anon buffer
    # In geninf_on_rff.jl, quick_IQL(kworld, anon_data) trains using mdp=get_agent(kworld,"ag1").
    # Here we use a minimal per-mdp pattern (works if OnlineIQLearn etc already imported):
    π_iql, 𝒟_iql, _, _ = quick_IQL(mdp, pack.anon)

    # 2) build π_dist action mappings from this mdp’s action set
    as = actions(mdp)
    action_list = [as, a->Flux.onehot(a, as), Flux.onehotbatch(as, as)]
    π_dist = ScoreΠDist(; mdp_params=action_list)

    # 3) agent_params from mdp
    agent_params = agent_params_from_mdp(mdp)
    T = size(pack.full.data[:s], 2) # num of cols in state # TODO: should be ...data.elements

    # 4) Decide on T
    data_slices = (T ≤ minN) ? collect(1:T) : [rand(1:T) for _ in 1:minN]

    ########################
    # Mode A: real dataset
    ########################
    # PF uses (state_data[:,1:T], aidx[1:T]) from the full buffer
    state_data = pack.full.data[:s][:, data_slices]
    obs_aidx   = onehot_cols_to_aidx(pack.full.data[:a][:, data_slices])

    pf_real = particle_filter(obs_aidx, π_dist, agent_params, state_data, n_particles;
                              ess_thresh=ess_thresh, refine_every=refine_every, refine_topk=refine_topk)

    ###############################
    # Mode B: IQL grid surrogate PF
    ###############################
    iql_state_data, iql_obs_aidx, _ = surrogate_dataset_from_iql_grid(π_dist, π_iql, mdp; eval_num=iql_gridN)

    pf_iql = particle_filter(iql_obs_aidx, π_dist, agent_params, iql_state_data, n_particles;
                             ess_thresh=ess_thresh, refine_every=refine_every, refine_topk=refine_topk)

    return (
        pack = pack,
        mdp = mdp,
        agent_params = agent_params,
        π_dist = π_dist,
        π_iql = π_iql,
        pf_real = pf_real,
        pf_iql  = pf_iql,
        real = (state_data=state_data, obs_aidx=obs_aidx),
        iql  = (state_data=iql_state_data, obs_aidx=iql_obs_aidx)
    )
end

"""
    summarize_eval(evals; n_particles::Int, gridsize::Int=120)

TBW
"""
function summarize_eval(evals; n_particles::Int, gridsize::Int=120)
    rows_real = NamedTuple[]
    rows_iql  = NamedTuple[]

    for E in evals
        pack = E.pack
        ann  = pack.ann

        # Real-mode metrics
        degR = pf_degeneracy(E.pf_real, E.π_dist; n_particles=n_particles)
        objR = objective_recon_metrics(E.pf_real, E.π_dist, E.mdp; gridsize=gridsize)
        polR = policy_match_acc(E.pf_real, E.π_dist, E.agent_params, E.real.state_data, E.real.obs_aidx)

        push!(rows_real, (
            run_id=pack.run_id, agent=pack.agent, inst=pack.inst,
            num_goals=ann.num_goals, num_obstacles=ann.num_obstacles, max_goal_sep=ann.max_goal_separation,
            obj_rmse_z=objR.rmse_z, obj_corr=objR.corr,
            policy_acc=polR.acc, policy_N=polR.N,
            deg_all_ninf=degR.all_logw_ninf, deg_nunique=degR.nunique, deg_collapsed=degR.collapsed, ess=degR.ess
        ))

        # IQL-surrogate-mode metrics (policy_acc computed against surrogate actions at surrogate states)
        degI = pf_degeneracy(E.pf_iql, E.π_dist; n_particles=n_particles)
        objI = objective_recon_metrics(E.pf_iql, E.π_dist, E.mdp; gridsize=gridsize)
        polI = policy_match_acc(E.pf_iql, E.π_dist, E.agent_params, E.iql.state_data, E.iql.obs_aidx)

        push!(rows_iql, (
            run_id=pack.run_id, agent=pack.agent, inst=pack.inst,
            num_goals=ann.num_goals, num_obstacles=ann.num_obstacles, max_goal_sep=ann.max_goal_separation,
            obj_rmse_z=objI.rmse_z, obj_corr=objI.corr,
            policy_acc=polI.acc, policy_N=polI.N,
            deg_all_ninf=degI.all_logw_ninf, deg_nunique=degI.nunique, deg_collapsed=degI.collapsed, ess=degI.ess
        ))
    end

    return (real=rows_real, iql=rows_iql)
end


"""
    pack_feat(p)

Feature vector used to diversify ordering.
"""
pack_feat(p) = Float64[p.ann.num_goals, p.ann.num_obstacles, p.ann.max_goal_separation]

"""
    diversify_packs(packs::Vector{RunPack})

Greedy farthest-next ordering to maximize diversity between consecutive packs.
"""
function diversify_packs(packs::Vector{RunPack})
    n = length(packs)
    n <= 2 && return packs

    F = [pack_feat(p) for p in packs]

    # Normalize each feature dimension for sane distances
    M = reduce(hcat, F)  # 3×n
    μ = mean(M, dims=2)
    σ = std(M, dims=2)
    σ .= max.(σ, 1e-9)
    Mz = (M .- μ) ./ σ

    # Start at an extreme point (max norm) to reduce dependence on initial ordering
    norms = vec(sum(abs2, Mz; dims=1))
    start = argmax(norms)

    order = Int[start]
    remaining = Set(1:n)
    delete!(remaining, start)

    while !isempty(remaining)
        last = order[end]
        best_i = first(remaining)
        best_d = -Inf
        @inbounds for i in remaining
            d = sum(abs2, Mz[:, i] .- Mz[:, last])  # squared L2
            if d > best_d
                best_d = d
                best_i = i
            end
        end
        push!(order, best_i)
        delete!(remaining, best_i)
    end

    return packs[order]
end

"""
    summarize_ablation(evals)

Summarize the ablation formed by a set of evals.
"""
function summarize_ablation(evals)
    sweeps = unique(e.sweep for e in evals)
    out = Dict{Symbol,Any}()

    for sw in sweeps
        Es = filter(e->e.sweep==sw, evals)
        levels = sort(unique(e.level for e in Es))

        # NaN-safe aggregation over replicates at each level
        function agg(f)
            [begin
                vals = [f(e) for e in Es if e.level==lv]
                vals = filter(x -> !(ismissing(x) || (x isa Real && isnan(x))), vals)
                nanmean(vals)
             end for lv in levels]
        end

        # Degeneracy
        essA = agg(e->e.A.deg.ess)
        essB = agg(e->e.B.deg.ess)
        colA = agg(e->e.A.deg.collapsed ? 1.0 : 0.0)
        colB = agg(e->e.B.deg.collapsed ? 1.0 : 0.0)

        badA = agg(e->e.A.bad ? 1.0 : 0.0)
        badB = agg(e->e.B.bad ? 1.0 : 0.0)

        # Objective recon
        rmseA = agg(e->e.A.obj.rmse_z)
        rmseB = agg(e->e.B.obj.rmse_z)
        corA  = agg(e->e.A.obj.corr)
        corB  = agg(e->e.B.obj.corr)

        # Policy match
        accA  = agg(e->e.A.pol.acc)
        accB  = agg(e->e.B.pol.acc)

        out[sw] = (levels=levels,
                   essA=essA, essB=essB,
                   collapsedA=colA, collapsedB=colB,
                   badA=badA, badB=badB,
                   rmseA=rmseA, rmseB=rmseB,
                   corrA=corA, corrB=corB,
                   accA=accA, accB=accB)
    end

    return out
end

degmask_from_summary(metricA::Vector, metricB::Vector, colA::Vector, colB::Vector) = (
    ((colA .>= 0.5) .| isnan.(Float64.(metricA))),   # degenerate A
    ((colB .>= 0.5) .| isnan.(Float64.(metricB)))    # degenerate B
)

"""
    cache_record_for_eval(cache, e)

Find the cache record corresponding to eval entry e.
Supports either direct index by id (1..N) or lookup by rec[:id].
"""
function cache_record_for_eval(cache::Dict, e)
    records = cache[:records]
    # fast path if ids are 1..N in order
    if 1 ≤ e.id ≤ length(records) && haskey(records[e.id], :id) && records[e.id][:id] == e.id
        return records[e.id]
    end
    # fallback lookup
    for r in records
        if r[:id] == e.id
            return r
        end
    end
    error("No cache record found for eval id=$(e.id)")
end

"""
    best_eval_by_accuracy(evals; requireA=false, requireB=true)

Select eval with highest IQ-SIPS accuracy subject to degeneracy constraints.
Skips NaNs.
"""
function best_eval_by_accuracy(evals; requireA::Bool=false, requireB::Bool=true)
    best = nothing
    best_acc = -Inf

    for e in evals
        if requireB && get(e.B, :bad, false)
            continue
        end
        if requireA && get(e.A, :bad, false)
            continue
        end

        acc = get(e.B.pol, :acc, NaN)
        if isnan(acc)
            continue
        end

        if acc > best_acc
            best_acc = acc
            best = e
        end
    end

    best === nothing && error("No eval matched constraints requireA=$requireA requireB=$requireB")
    return best
end