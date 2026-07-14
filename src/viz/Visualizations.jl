module Visualizations

using Colors
using Plots
using Random

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
    quick_heatmap

end
