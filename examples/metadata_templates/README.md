Metadata templates

This directory contains example metadata TOML files for datasets used by the repository.

Purpose
- Each dataset should have a companion metadata file (TOML) that describes where the data lives and how it should be unpacked or interpreted.
- Parse the metadata once and pass the parsed Dict down the pipeline.

Usage

```julia
meta = TOML.parsefile("/path/to/multi_run.meta.toml")
packs = Utils.load_runpacks(meta)
```

Or pass the path directly (it will be parsed inside `load_runpacks`):

```julia
packs = Utils.load_runpacks("/path/to/multi_run.meta.toml")
```

Templates

Multi-run dataset metadata (multi_run.meta.toml)

```toml
schema_version = 1
data_path = "/path/to/multi_run_data.bson"
format = "bson"
data_type = "multi_run"
created_at = "2026-02-18T14:00:00Z"
created_by = "yourname"
notes = "Example metadata for a multi-run dataset"

# top-level multi-run info
n_agents = 3
agent_names = ["ag1","ag2","ag3"]
runs_per_agent = 10
agent_key_pattern = "ag%s"

[loader]
run_container_key = "runs"
run_index_key = "ind_exps"
agent_entry_key = "agents"
full_key = "full_data"
anon_key = "anon_data"
expected_keys = ["s","sp","a","r"]
unpack_strategy = "runs-array"

[state]
state_field_sizes = [2, 2, 12, 10, 1]
state_field_names = ["loc","vel","obcs","goals","time"]
keep_state_fields = [true, true, true, false, true]
anonize_first_rows = 2
auto_clean = true
```

Explanation and usage in Arrodes:
- schema_version: informational for future migrations; not consumed by loaders.
- data_path: used by `Utils.load_runpacks` to locate the BSON file to load.
- format: informational; not consumed by loaders.
- data_type: used by `Utils.load_runpacks` to validate the dataset type.
- created_at, created_by, notes: informational; not consumed by loaders.
- n_agents, runs_per_agent, agent_key_pattern: informational; not consumed by loaders.
- agent_names: optional; if present, `Utils.load_runpacks` uses this instead of auto-detecting agent keys.
- loader.run_container_key: used by `Utils.load_runpacks` to find the run container in the BSON.
- loader.run_index_key: used by `Utils.load_runpacks` to find the per-agent run list (e.g., `ind_exps`).
- loader.agent_entry_key, loader.full_key, loader.anon_key, loader.expected_keys, loader.unpack_strategy: reserved for future loaders; not consumed currently.
- state.state_field_sizes and state.keep_state_fields: used by `data_cleaner` (called in `Utils.load_runpacks`) to slice state vectors.
- state.state_field_names, state.anonize_first_rows, state.auto_clean: informational for now; `data_cleaner` only consumes sizes and keep flags.

Ablation cache metadata (ablation_cache.meta.toml)

```toml
data_type = "ablation_cache"
cache_path = "/path/to/ablation_cache.bson"
created_at = "2026-02-18T12:00:00Z"
version = "1.0.0"
records_key = "cache"

[summary]
n_skeletons = 25
n_objectives = 30
T = 20
nbins = 5
per_bin = 5

[source_meta]
data_type = "multi_run"
data_path = "/path/to/multi_run_data.bson"
data_meta_path = "/path/to/multi_run.meta.toml"
```

Explanation and usage in Arrodes:
- data_type: identifier for cache metadata; not consumed by loaders today.
- cache_path: path to the cache BSON; typically used by scripts before calling `load_ablation_cache`.
- created_at, version: provenance fields; not consumed by loaders today.
- records_key: identifies the top-level key inside the BSON that stores the cache Dict (default is `cache` as expected by `load_ablation_cache`).
- summary.n_skeletons, summary.n_objectives, summary.T, summary.nbins, summary.per_bin: provenance and reporting fields derived from the cache `:meta` (not consumed by loaders today).
- source_meta.data_type: provenance for the source dataset type (usually `multi_run`).
- source_meta.data_path: path to the dataset BSON used to generate the cache.
- source_meta.data_meta_path: path to the dataset metadata that describes the source dataset.

Ablation wholesale metadata (ablation_wholesale.meta.toml)

```toml
schema_version = 1
data_type = "ablation_wholesale"
data_path = "/path/to/ablation_wholesale_TIMESTAMP.bson"
format = "bson"
created_at = "2026-02-19T12:00:00Z"
created_by = "iq_sips_ablation.jl"
notes = "Complete ablation study output with cache, evaluations, and summaries"

[source]
cache_meta_path = "/path/to/ablation_cache_TIMESTAMP.meta.toml"
generation_mode = "generate"  # or "load"

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
```

Explanation and usage in Arrodes:
- schema_version: metadata schema version (for future compatibility).
- data_type: identifies this as ablation wholesale output (complete study results).
- data_path: path to the wholesale BSON file.
- format: file format (bson).
- created_at: timestamp when the wholesale output was generated.
- created_by: script that generated this output (typically `iq_sips_ablation.jl`).
- notes: human-readable description of the dataset.
- source.cache_meta_path: links to the cache metadata file used/generated for this study.
- source.generation_mode: whether cache was generated fresh (`generate`) or loaded from existing (`load`).
- structure.top_level_key: the key in the BSON file that contains the data (`out`).
- structure.fields: list of fields available in the output NamedTuple.
- evals_structure.description: describes what the `evals` vector contains.
- evals_structure.mode_A: full description of what Mode A represents (real expert data).
- evals_structure.mode_B: full description of what Mode B represents (IQL surrogate data).
- evals_structure.mode_A_label: short label for Mode A used in plot legends ("Real Data").
- evals_structure.mode_B_label: short label for Mode B used in plot legends ("IQL Surrogate").
- sweeps.keys: list of sweep parameter types (K, freq_range, amp_range).
- sweeps.K.description: describes what the K sweep varies.
- sweeps.K.xlabel: x-axis label for K sweep plots.
- sweeps.freq_range.description: describes what the freq_range sweep varies.
- sweeps.freq_range.xlabel: x-axis label for freq_range sweep plots.
- sweeps.amp_range.description: describes what the amp_range sweep varies.
- sweeps.amp_range.xlabel: x-axis label for amp_range sweep plots.
- metrics.keys: list of available metrics (ess, rmse, acc).
- metrics.ess.ylabel: y-axis label for ESS metric plots.
- metrics.ess.field_A: field name in summaries Dict for Mode A ESS values (`essA`).
- metrics.ess.field_B: field name in summaries Dict for Mode B ESS values (`essB`).
- metrics.rmse.ylabel: y-axis label for RMSE metric plots.
- metrics.rmse.field_A: field name in summaries Dict for Mode A RMSE values (`rmseA`).
- metrics.rmse.field_B: field name in summaries Dict for Mode B RMSE values (`rmseB`).
- metrics.acc.ylabel: y-axis label for accuracy metric plots.
- metrics.acc.field_A: field name in summaries Dict for Mode A accuracy values (`accA`).
- metrics.acc.field_B: field name in summaries Dict for Mode B accuracy values (`accB`).

Notes
- The loader functions assume the metadata contains the necessary keys. Missing keys will cause the loader to error.
- Keep file paths absolute when possible to avoid ambiguity.

Naming conventions
- **All metadata files follow a strict naming convention**: `<data_file>.meta.toml`
  - If your data BSON is `multi_run_data.bson`, the metadata must be `multi_run_data.meta.toml`
  - If your cache BSON is `ablation_cache_20260219.bson`, the metadata must be `ablation_cache_20260219.meta.toml`
  - This convention is enforced by metadata writers (`write_cache_metadata`, `write_multi_run_metadata` in MuKumari, etc.)
  - Helper function `Utils.infer_metadata_path(data_path)` computes the metadata path from any data file path
  - Do not use custom names for metadata files; always follow the convention
