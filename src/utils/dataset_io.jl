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
        @argcheck d isa Dict "Expected (kworld, Dict) in run payload."
        dd = deepcopy(d)
        dd["kworld"] = kw
        return dd
    end
    @argcheck x isa Dict "Expected Dict run payload."
    return x
end

"""
    infer_metadata_path(data_path)

Given a data file path following the naming convention, compute the corresponding
metadata file path. For example:
  - "cache.bson" → "cache.meta.toml"
  - "data.jld2" → "data.meta.toml"
  - "/path/to/file.xyz" → "/path/to/file.meta.toml"

The function replaces the extension with `.meta.toml` regardless of whether
the input has an extension.

# Arguments:
- `data_path::AbstractString`: path to the data file (BSON, JLD2, or other format)

# Returns:
- `String`: absolute path to the inferred metadata file

# Example:
```julia
infer_metadata_path("ablation_cache.bson")  # → absolute path ending in "ablation_cache.meta.toml"
```
"""
function infer_metadata_path(data_path::AbstractString)
    data_abs = abspath(data_path)
    # Remove extension and append .meta.toml
    base, _ = splitext(data_abs)
    return base * ".meta.toml"
end

"""
    read_data_metadata(meta_path)

Parse a TOML metadata file and return the parsed Dict{String,Any}.
This is a thin wrapper around `TOML.parsefile` kept here so callers
can uniformly request metadata via a named utility.
"""
function read_data_metadata(meta_path::AbstractString)
    meta_abs = abspath(meta_path)
    return TOML.parsefile(meta_abs)
end

"""
    write_cache_metadata(cache_path; n_skeletons, n_objectives, T, nbins, per_bin,
                        source_data_type, source_data_path, source_data_meta_path)

Generate a cache metadata TOML file for an ablation cache BSON.

The metadata file is written to `<cache_path>.meta.toml` (replacing `.bson` with `.meta.toml`).
This enforces the style convention that cache metadata must be named identically to the cache
file but with the `.meta.toml` extension.

Arguments:
- `cache_path`: path to the cache BSON file (e.g., "data/ablation_cache_20260219.bson")
- `n_skeletons`, `n_objectives`, `T`, `nbins`, `per_bin`: summary statistics from cache generation
- `source_data_type`: data type of the source dataset (e.g., "multi_run")
- `source_data_path`: path to the source dataset BSON
- `source_data_meta_path`: path to the source dataset metadata TOML

Optional keyword arguments:
- `version`: semantic version string (default: "1.0.0")
- `records_key`: BSON key where cache Dict is stored (default: "cache")
"""
function write_cache_metadata(
    cache_path::AbstractString;
    n_skeletons::Int,
    n_objectives::Int,
    T::Int,
    nbins::Int,
    per_bin::Int,
    source_data_type::AbstractString,
    source_data_path::AbstractString,
    source_data_meta_path::AbstractString,
    version::AbstractString = "1.0.0",
    records_key::AbstractString = "cache",
)

    # Enforce naming convention: <cache>.bson -> <cache>.meta.toml
    @argcheck endswith(cache_path, ".bson") "Cache path must end with .bson; got: $cache_path"
    meta_path = cache_path[1:end-5] * ".meta.toml"

    created_at = Dates.format(Dates.now(), "yyyy-mm-ddTHH:MM:SSZ")

    # Build metadata structure
    meta_content = """
    data_type = "ablation_cache"
    cache_path = "$(cache_path)"
    created_at = "$(created_at)"
    version = "$(version)"
    records_key = "$(records_key)"

    [summary]
    n_skeletons = $(n_skeletons)
    n_objectives = $(n_objectives)
    T = $(T)
    nbins = $(nbins)
    per_bin = $(per_bin)

    [source_meta]
    data_type = "$(source_data_type)"
    data_path = "$(source_data_path)"
    data_meta_path = "$(source_data_meta_path)"
    """

    open(meta_path, "w") do io
        write(io, meta_content)
    end

    return meta_path
end

function load_runpacks(meta_or_path::Union{AbstractString, Dict{String, Any}})
    # Resolve metadata (parse TOML once if a path was provided)
    meta =
        isa(meta_or_path, AbstractString) ? read_data_metadata(meta_or_path) : meta_or_path

    # This loader is explicitly for multi-run datasets
    @argcheck haskey(meta, "data_type") && meta["data_type"] == "multi_run" "load_runpacks expects metadata for a multi_run dataset; got: $(get(meta, "data_type", "<missing>"))"

    # Load BSON from the data_path specified in metadata
    bson_path = meta["data_path"]
    raw = BSON.load(bson_path)

    # locate run container according to metadata (fallback to legacy keys)
    rc_key = get(get(meta, "loader", Dict{String, Any}()), "run_container_key", "runs")
    runs =
        haskey(raw, Symbol(rc_key)) ? raw[Symbol(rc_key)] :
        haskey(raw, rc_key) ? raw[rc_key] :
        (haskey(raw, :runs) ? raw[:runs] : (haskey(raw, "runs") ? raw["runs"] : [raw]))

    packs = RunPack[]
    for (rid, r0) in enumerate(runs)
        run = _normalize_run_payload(r0)

        kworld =
            haskey(run, "kworld") ? run["kworld"] :
            haskey(run, :kworld) ? run[:kworld] : (@argcheck false "No kworld in run $rid")

        ann = kworld_annotations(kworld)

        # determine agent keys: prefer explicit list in metadata, else fall back to legacy prefix detector
        agent_keys =
            haskey(meta, "agent_names") ? meta["agent_names"] :
            sort([k for k in keys(run) if k isa String && startswith(k, "ag")])

        # run index key (inside each agent entry) -- default is ind_exps for legacy compat
        run_index_key =
            get(get(meta, "loader", Dict{String, Any}()), "run_index_key", "ind_exps")

        # state-cleaner metadata (required for multi-run path)
        state_meta = get(meta, "state", get(meta, "cleaner", nothing))
        @argcheck state_meta !== nothing "Missing state metadata in multi-run metadata (expected [state] section, or [cleaner] section for legacy compatibility)."
        s_sizes = state_meta["state_field_sizes"]
        s_keep = state_meta["keep_state_fields"]

        for agent in agent_keys
            expdict = run[agent]  # Dict(:ind_exps=>..., :total_exp=>...)
            insts = expdict[run_index_key]

            for (k, inst) in enumerate(insts)
                full_buf, anon_buf = inst
                full_buf = data_cleaner(full_buf, s_sizes, s_keep)
                anon_buf = data_cleaner(anon_buf, s_sizes, s_keep)
                name = agent * "_" * string(k)
                mdp = kworld.inhabitants[name]  # matches generator naming
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
function select_skeleton_mdps(
    meta_or_path::Union{AbstractString, Dict{String, Any}};
    nbins::Int = 5,
    per_bin::Int = 5,
    rng = Random.default_rng(),
)

    # resolve metadata if a path was provided
    meta =
        isa(meta_or_path, AbstractString) ? read_data_metadata(meta_or_path) : meta_or_path
    packs_all = load_runpacks(meta)
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
        lo = floor(Int, (b - 1) * N_total / nbins) + 1
        hi = floor(Int, b * N_total / nbins)
        bins[b] = packs_sorted[lo:hi]
    end

    chosen = RunPack[]
    boundaries = NamedTuple[]

    for (b, binpacks) in enumerate(bins)
        binN = length(binpacks)
        if binN == 0
            push!(
                boundaries,
                (bin = b, min_obstacles = missing, max_obstacles = missing, count = 0),
            )
            continue
        end
        mino = minimum(p.ann.num_obstacles for p in binpacks)
        maxo = maximum(p.ann.num_obstacles for p in binpacks)

        push!(
            boundaries,
            (bin = b, min_obstacles = mino, max_obstacles = maxo, count = binN),
        )

        k = min(per_bin, binN)
        picks = randperm(rng, binN)[1:k]
        append!(chosen, binpacks[picks])
    end

    return packs_all,
    chosen,
    (total = N_total, nbins = nbins, per_bin = per_bin, boundaries = boundaries)
end