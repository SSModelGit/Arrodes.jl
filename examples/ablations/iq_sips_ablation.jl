# iq_sips_ablation.jl
# Wrapper script to run the IQ-SIPS ablation pipeline using Arrodes.jl

using Arrodes
using Random
using BSON
using Dates
using Plots
using ArgCheck

# Output directory for cache and plots (data subfolder)
# Also used as the default location for loading caches when mode == :load
examples_dir = @__DIR__
DATA_DIR = joinpath(examples_dir, "data")

# Path to a BSON file containing runs (input runpacks). If running in :generate
# mode this must point to your runpack file. When mode == :load this argument is
# ignored and the script will use the most-recent cache in the data folder by default.
const DEFAULT_RUNS_META = joinpath(DATA_DIR, "30_15_10_3_multi_trace_run.meta.toml")

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
            joinpath(DATA_DIR, cache_metadata_filename),
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

"""
    make_and_save_wholesale_plots(wholesale_meta_path::AbstractString; save_dir::AbstractString=DATA_DIR)

Load the wholesale ablation data and its metadata from a .meta.toml file, then produce a set of
diagnostic plots using the Visualizations submodule and save them to `save_dir`.

# Arguments
- `wholesale_meta_path`: Path to the `.meta.toml` metadata file for the wholesale output
- `save_dir`: Directory where a subfolder with plots will be created (default: DATA_DIR)

The subfolder is named after the metadata file (without the `.meta.toml` extension) and contains
all generated plots organized by type.

Returns a Dict of produced plots / filenames.
"""
function make_and_save_wholesale_plots(wholesale_meta_path::AbstractString; save_dir::AbstractString=DATA_DIR)
    @argcheck endswith(wholesale_meta_path, ".meta.toml") "wholesale_meta_path must end with .meta.toml; got: $wholesale_meta_path"
    
    # Extract folder name from metadata filename (strip .meta.toml)
    meta_basename = basename(wholesale_meta_path)
    folder_name = meta_basename[1:end-10]  # Remove ".meta.toml"
    plot_dir = joinpath(save_dir, folder_name)
    mkpath(plot_dir)

    # Load wholesale data from metadata file
    out, metadata = load_ablation_wholesale_from_metadata(wholesale_meta_path)

    results = Dict{Symbol,Any}()

    # 1) Final inference figures (two summary plots)
    try
        figs = make_final_inference_figures(out)
        p1 = figs.p1; p2 = figs.p2
        f1 = joinpath(plot_dir, "final_inference_p1.png")
        f2 = joinpath(plot_dir, "final_inference_p2.png")
        savefig(p1, f1)
        savefig(p2, f2)
        results[:final_inference] = (p1=f1, p2=f2, meta=(best_iqsips=figs.best_iqsips, best_both=figs.best_both))
    catch err
        @warn "make_final_inference_figures failed" error=err
    end

    # 2) All objectives pages (paginated); save into subdir
    try
        obj_dir = joinpath(plot_dir, "objectives_pages")
        files = plot_all_objectives_from_cache(out.cache; savepath=obj_dir)
        results[:all_objectives_pages] = files
    catch err
        @warn "plot_all_objectives_from_cache failed" error=err
    end

    # 3) Ablation barplots
    barplots = make_ablation_barplots(wholesale_meta_path)
    ap_dir = joinpath(plot_dir, "ablation_barplots")
    mkpath(ap_dir)
    saved = Dict{Symbol,Dict{Symbol,String}}()
    for (sw, dict) in barplots
        saved[sw] = Dict{Symbol,String}()
        for (metric, p) in dict
            fname = joinpath(ap_dir, "ablation_$(sw)_$(metric).png")
            savefig(p, fname)
            saved[sw][metric] = fname
        end
    end
    results[:ablation_barplots] = saved

    return results
end

# main()
