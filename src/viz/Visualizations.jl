module Visualizations

using Printf
import GeoInterface as GI
using TOML, BSON

using Plots
using StatsPlots
using Measures
using Colors
using MuKumari
using Crux
using Flux

using ..Utils
using ..Priors
using ..RL
using ..Analysis
import ..Arrodes: ScoreΠDist, InferenceConfig

include("objectives.jl")
export plot_top_objective_with_trajectories,
    plot_objective_side_by_side,
    plot_true_objective_vs_iqsips_rollout,
    plot_objective_triptych,
    make_final_inference_figures,
    plot_all_objectives_from_cache,
    compare_iql_vs_true_policy,
    plot_particle_filter_explanation

include("ablation_plots.jl")
export load_ablation_wholesale_from_metadata,
    make_ablation_barplots, grouped_bars_with_degenerate_overlay

end
