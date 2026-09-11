module Visualizations

using Colors
using LinearAlgebra
import MuKumari
using Plots
using Random
import SCRIBE
using Statistics

using ..ObjectiveInference: ObjectiveInferenceProblem, objective_probabilities
using ..ObjectiveInference: hypothesis_index, hypothesis_mdp
using ..ObjectiveInference: objective_observation_count
using ..ObjectiveInference: top_objective_hypotheses
using ..BehaviorModels: prepare_behavior, rollout_behavior
using ..WorldInference: EOFWorldInferenceResult, SOMWorldInferenceResult,
    target_measure_mmd

include("filter_explanations.jl")
include("world_inference.jl")

export plot_particle_filter_explanation,
    plot_particle_filter_frame,
    plot_particle_heatmaps_frame,
    make_particle_filter_frame_fn,
    make_particle_heatmaps_frame_fn,
    animate_particle_filter_from_frames,
    save_particle_filter_animation,
    quick_heatmap,
    plot_world_posterior_comparison,
    save_world_posterior_animation,
    plot_world_particle_distribution,
    plot_world_trial_recovery,
    plot_world_trial_particles,
    save_world_inference_visualizations,
    world_result_comparison_panels,
    plot_world_result_comparison,
    save_world_result_comparison_animation

end
