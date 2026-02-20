# iq_sips_ablation.jl
# Wrapper script to run the IQ-SIPS ablation pipeline using Arrodes.jl

using Arrodes
using Random
using BSON
using Dates

# Output directory for cache and plots (data subfolder)
# Also used as the default location for loading caches when mode == :load
examples_dir = @__DIR__
DATA_DIR = joinpath(examples_dir, "data")

# Path to a BSON file containing runs (input runpacks). If running in :generate
# mode this must point to your runpack file. When mode == :load this argument is
# ignored and the script will use the most-recent cache in the data folder by default.
const DEFAULT_RUNS_META = joinpath(DATA_DIR, "50_15_10_3_multi_trace_run.meta.toml")

# find most recent cache metadata file in DATA_DIR matching our naming convention
function latest_cache_file(datadir::AbstractString)
    files = filter(f -> occursin(r"^ablation_cache_.*\.meta\.toml$", f), readdir(datadir))
    if isempty(files)
        return nothing
    end
    # sort by modification time
    paths = joinpath.(datadir, files)
    sort!(paths, by = p -> stat(p).mtime; rev = true)
    return first(paths)
end

# timestamp helper
function timestamp_str(t::DateTime = now())
    return Dates.format(t, dateformat"yyyymmddTHHMM")
end

"""
    main(; training_meta_path::AbstractString = DEFAULT_RUNS_META,
                cache_metadata_filename::Union{Nothing,AbstractString} = nothing,
                mode::Symbol = :generate,
                rng = Random.default_rng())

Run the IQ-SIPS ablation pipeline.
By default, this will generate a new cache and wholesale output.

Two crucial inputs:
    * `training_meta_path`: path to a TOML metadata file describing the multi-run dataset.
        The TOML must include `data_path` which points to the BSON of runpacks.
        Required when `mode == :generate`; ignored internally when mode == :load.
    * `cache_metadata_filename` (default=nothing): metadata filename for cache
      (e.g., "ablation_cache_20260219.meta.toml").
        * If in :generate mode, this name is generated with timestamp if not specified.
        * If in :load mode, defaults to loading the most recent metadata in the data folder.
"""
function main(;
    training_meta_path::AbstractString = DEFAULT_RUNS_META,
    cache_metadata_filename::Union{Nothing, AbstractString} = nothing,
    mode::Symbol = :generate,
    rng = Random.default_rng(),
)

    ts = timestamp_str()
    if mode == :generate
        if isnothing(cache_metadata_filename)
            cache_metadata_filename = "ablation_cache_$(ts).meta.toml"
        end
        println(
            "Going to generate cache - metadata will be saved at: ",
            joinpath(DATA_DIR, cache_metadata_basename),
        )
    elseif mode == :load
        if isnothing(cache_metadata_filename)
            cache_metadata_filename = latest_cache_file(DATA_DIR)
            if cache_metadata_filename === nothing
                error(
                    "No cache metadata files found in $(DATA_DIR). Run with mode=:generate first or supply a cache metadata filename.",
                )
            end
        end
        println("Loading cache from metadata: ", cache_metadata_filename)
    else
        error("Unknown mode=$(mode). Use :generate or :load.")
    end

    # Doing weird stuff here because can't be bothered to fix how we're loading caches
    # better make sure your metadata is in the data folder
    cache_metadata_basename = basename(cache_metadata_filename)

    out = ablation_main(
            training_meta_path;
            script_dir = DATA_DIR,
            mode = mode,
            cache_metadata_filename = cache_metadata_basename,
            rng = rng,
    )
    
    wholesale_name = joinpath(DATA_DIR, "ablation_wholesale_$(ts).bson")
    BSON.@save wholesale_name out
    println("Wholesale output saved as: ", wholesale_name)
    
    # Write wholesale metadata
    wholesale_meta_path = write_wholesale_metadata(
        wholesale_name;
        cache_meta_path = joinpath(DATA_DIR, cache_metadata_basename),
        mode = mode,
    )
    println("Wholesale metadata saved as: ", wholesale_meta_path)
    
    return out
end

# main()
