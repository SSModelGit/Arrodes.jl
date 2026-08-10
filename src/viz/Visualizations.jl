module Visualizations

using Colors
using LinearAlgebra
using Plots
using Random
using SCRIBE

using ..Inference
using ..Planning

include("filter_explanations.jl")

export plot_particle_filter_explanation,
    plot_particle_filter_frame,
    plot_particle_heatmaps_frame,
    make_particle_filter_frame_fn,
    make_particle_heatmaps_frame_fn,
    animate_particle_filter_from_frames,
    save_particle_filter_animation,
    quick_heatmap,
    plot_world_filter_frame,
    plot_world_diagnostics,
    plot_world_modes,
    plot_world_ancestry,
    plot_world_deployment,
    plot_world_particle_plans

end
