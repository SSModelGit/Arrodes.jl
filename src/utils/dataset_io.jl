"""
    safe_get_obstacle_count(mdp_or_pack)

Prefer annotations from RunPack when available; else fall back to mdp.obcs length.
"""
function safe_get_obstacle_count(x)
    if hasproperty(x, :ann)
        return getproperty(x.ann, :num_obstacles)
    end
    if hasproperty(x, :obcs)
        return length(getproperty(x, :obcs))
    end
    return missing
end

"""
    _normalize_run_payload(x)

Ensures the resultant run payload has the right shaping across multiple input stream types.

Supports:
  - stored as raw[:data] = (kworld, dataDict)
  - stored as raw[:runs] = [ ... ]  (each a run dict or (kworld,data))
"""
function _normalize_run_payload(x)
    if x isa Tuple && length(x) == 2
        kw, d = x
        d isa Dict || error("Expected (kworld, Dict) in run payload.")
        dd = deepcopy(d)
        dd["kworld"] = kw
        return dd
    end
    x isa Dict || error("Expected Dict run payload.")
    return x
end

function load_runpacks(bson_path::AbstractString)
    raw = BSON.load(bson_path)

    runs =
        haskey(raw, :runs)  ? raw[:runs]  :
        haskey(raw, "runs") ? raw["runs"] :
        haskey(raw, :data)  ? [raw[:data]] :
        haskey(raw, "data") ? [raw["data"]] :
        [raw]

    packs = RunPack[]
    for (rid, r0) in enumerate(runs)
        run = _normalize_run_payload(r0)

        kworld = haskey(run, "kworld") ? run["kworld"] :
                 haskey(run, :kworld)  ? run[:kworld]  :
                 error("No kworld in run $rid")

        ann = kworld_annotations(kworld)

        # agent keys are Strings like "ag1".."ag7"
        agent_keys = sort([k for k in keys(run) if k isa String && startswith(k, "ag")])

        for agent in agent_keys
            expdict = run[agent]  # Dict(:ind_exps=>..., :total_exp=>...)
            insts = expdict[:ind_exps]

            for (k, inst) in enumerate(insts)
                full_buf, anon_buf = inst
                full_buf = data_cleaner(full_buf, [2,2,12,10,1],Bool[1,1,1,0,1])
                anon_buf = data_cleaner(anon_buf, [2,2,12,10,1],Bool[1,1,1,0,1])
                name = agent * "_" * string(k)
                mdp  = kworld.inhabitants[name]  # matches generator naming
                push!(packs, RunPack(rid, agent, k, mdp, full_buf, anon_buf, ann))
            end
        end
    end
    return packs
end

"""
    select_skeleton_mdps(bson_path; nbins=5, per_bin=5, rng=Random.default_rng())

Loads RunPacks, counts them, bins by obstacle count (least→most), and selects `per_bin` packs per bin.
Returns:
- packs_all
- chosen_packs (length nbins*per_bin)
- bin_info (NamedTuple with boundaries and counts)
"""
function select_skeleton_mdps(bson_path::AbstractString;
                              nbins::Int=5,
                              per_bin::Int=5,
                              rng=Random.default_rng())

    packs_all = load_runpacks(bson_path)
    N_total = length(packs_all)

    # Sort by obstacle count ascending
    obs = [p.ann.num_obstacles for p in packs_all]
    order = sortperm(obs)
    packs_sorted = packs_all[order]
    obs_sorted = obs[order]

    # Split into nbins contiguous bins (equal size as possible)
    bins = Vector{Vector{RunPack}}(undef, nbins)
    idxs = collect(1:N_total)
    # chunk boundaries
    for b in 1:nbins
        lo = floor(Int, (b-1)*N_total/nbins) + 1
        hi = floor(Int, b*N_total/nbins)
        bins[b] = packs_sorted[lo:hi]
    end

    chosen = RunPack[]
    boundaries = NamedTuple[]

    for (b, binpacks) in enumerate(bins)
        binN = length(binpacks)
        if binN == 0
            push!(boundaries, (bin=b, min_obstacles=missing, max_obstacles=missing, count=0))
            continue
        end
        mino = minimum(p.ann.num_obstacles for p in binpacks)
        maxo = maximum(p.ann.num_obstacles for p in binpacks)

        push!(boundaries, (bin=b, min_obstacles=mino, max_obstacles=maxo, count=binN))

        k = min(per_bin, binN)
        picks = randperm(rng, binN)[1:k]
        append!(chosen, binpacks[picks])
    end

    return packs_all, chosen, (total=N_total, nbins=nbins, per_bin=per_bin, boundaries=boundaries)
end