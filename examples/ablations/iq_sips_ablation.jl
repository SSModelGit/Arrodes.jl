# iq_sips_ablation.jl
# Wrapper script to run the IQ-SIPS ablation pipeline using Arrodes.jl

using Pkg
# Optional: activate the local examples project
Pkg.activate(@__DIR__)

using Arrodes
using Random
using BSON
using Dates

# Path to a BSON file containing runs (input runpacks). If running in :generate
# mode this must point to your runpack file. When mode == :load this argument is
# ignored and the script will use the most-recent cache in the data folder by default.
const DEFAULT_RUNS_BSON = joinpath(@__DIR__, "..", "..", "examples", "runs_example.bson")

# Output directory for cache and plots (data subfolder)
examples_dir = @__DIR__
DATA_DIR = joinpath(examples_dir, "data")

# find most recent cache file in DATA_DIR matching our cache naming convention
function latest_cache_file(datadir::AbstractString)
    files = filter(f->occursin(r"^ablation_cache_.*\.bson$", f), readdir(datadir))
    if isempty(files)
        return nothing
    end
    # sort by modification time
    paths = joinpath.(datadir, files)
    sort!(paths, by = p -> stat(p).mtime; rev=true)
    return first(paths)
end

# timestamp helper
function timestamp_str(t::DateTime = now())
    return Dates.format(t, dateformat"yyyymmddTHHMM")
end

"""
    main(; training_bson_path::AbstractString = DEFAULT_RUNS_BSON,
                cache_filename::Union{Nothing,AbstractString} = nothing,
                mode::Symbol = :generate,
                rng = Random.default_rng())

Run the IQ-SIPS ablation pipeline.
By default, this will generate a new cache and wholesale output.

Two crucial inputs:
  * `training_bson_path`: path to a BSON file containing training runs.
    * Required when mode == :generate; ignored intenally when mode == :load.
  * `cache_filename`(default=nothing): filename to store the generated cache in (if mode == :generate)
    * If in :load mode, this is the optional path to a cache file to load.
    * If not specified, defaults to loading the most recent cache in the data folder.
"""
function main(; training_bson_path::AbstractString = DEFAULT_RUNS_BSON,
                cache_filename::Union{Nothing,AbstractString} = nothing,
                mode::Symbol = :generate,
                rng = Random.default_rng())

    ts = timestamp_str()

    if mode == :generate
        if isnothing(cache_filename); cache_filename = "ablation_cache_$(ts).bson"; end
        # call ablation_main; set script_dir to DATA_DIR so cache is written there
        out = ablation_main(training_bson_path; script_dir=DATA_DIR, mode=:generate, cache_filename=cache_filename, rng=rng)
        println("Generated cache saved as: ", joinpath(DATA_DIR, cache_filename))

    elseif mode == :load
        if isnothing(cache_filename)
            cache_filename = latest_cache_file(DATA_DIR)
            if cache_filename === nothing
                error("No cache files found in $(DATA_DIR). Run with mode=:generate first or supply a cache path.")
            end
        end
        println("Loading cache: ", cache_filename)
        # doing weird stuff here because I'm not going to bother fixing how ablation_main handles paths rn
        cache_basename = basename(cache_filename)
        out = ablation_main(training_bson_path; script_dir=DATA_DIR, mode=:load, cache_filename=cache_basename, rng=rng)

    else
        error("Unknown mode=$(mode). Use :generate or :load.")
    end

    wholesale_name = joinpath(DATA_DIR, "ablation_wholesale_$(ts).bson")
    BSON.@save wholesale_name out
    println("Wholesale output saved as: ", wholesale_name)
    return out
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
