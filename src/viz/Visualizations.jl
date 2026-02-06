module Visualizations

using Printf
import GeoInterface as GI

using MuKumari

using ..Utils
using ..Priors
using ..Core
using ..Analysis
import ..Arrodes: ScoreΠDist

include("objectives.jl")
export plot_top_objective_with_trajectories, plot_objective_side_by_side,
       plot_true_objective_vs_iqsips_rollout, plot_objective_triptych, make_final_inference_figures

include("ablation_plots.jl")
export make_ablation_barplots, grouped_bars_with_degenerate_overlay

end