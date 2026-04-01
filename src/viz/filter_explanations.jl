
"""
    plot_particle_filter_explanation(pf_state, config::InferenceConfig, 
                                     component_fields::Vector, true_objective_fn,
                                     state_data::Matrix, agent_params::Dict,
                                     π_dist::ScoreΠDist, mdp::KAgentPOMDP;
                                     gridsize::Int=180, n_top::Int=10)

Visualize what the particle filter is thinking by showing:
1. Heatmap of the true objective over the MDP domain
2. Simulated trajectories from the top N particles (ranked by posterior weight)
   - Darker dots = higher ranked particle
   - Lighter dots = lower ranked particle
3. Observed trajectory in red

# Arguments
- `pf_state`: Particle filter state from inference
- `config::InferenceConfig`: Configuration used for inference
- `component_fields::Vector`: Component field specifications
- `true_objective_fn`: Ground truth objective function (x, y) -> Real
- `state_data::Matrix`: Observed state trajectory (features × timesteps)
- `agent_params::Dict`: Agent configuration dict
- `π_dist::ScoreΠDist`: Cached policy distribution
- `mdp::KAgentPOMDP`: Original MDP
- `gridsize::Int`: Resolution of heatmap grid
- `n_top::Int`: Number of top particles to visualize

Returns a Plots.jl plot object.
"""
function plot_particle_filter_explanation(pf_state, config::InferenceConfig, 
                                         component_fields::Vector, true_objective_fn,
                                         state_data::Matrix, agent_params::Dict,
                                         π_dist::ScoreΠDist, mdp::KAgentPOMDP;
                                         gridsize::Int=180, n_top::Int=10)
    
    # Extract observed trajectory
    obs_x, obs_y = Utils.xy_path_from_state_matrix(state_data)
    
    # Create objective heatmap grid
    xs, ys = Utils._grid_from_mdp(mdp; gridsize=gridsize)
    Z_true = [true_objective_fn(x, y) for y in ys, x in xs]
    
    # Initialize plot with true objective heatmap
    p = heatmap(xs, ys, Z_true;
        aspect_ratio=1,
        dpi=220,
        size=(1000, 900),
        title="Particle Filter Explanation: Top $(min(n_top, length(pf_state.traces))) Particles",
        xlabel="x (world units)",
        ylabel="y (world units)",
        colorbar_title="True Objective",
        legend=:outertopleft,
        legendfontsize=8,
        margin=5Plots.mm)
    
    # Get traces and weights, sort by weight descending
    traces = pf_state.traces
    log_weights = pf_state.log_weights
    
    n_particles = length(traces)
    top_n = min(n_top, n_particles)
    
    # Get indices of top particles by weight
    _, top_indices = findmax(log_weights), nothing
    sorted_indices = sortperm(log_weights; rev=true)
    top_indices = sorted_indices[1:top_n]
    
    # Starting position for all rollouts
    start_state = blindstart_KAgentState(mdp, agent_params[:start])
    n_timesteps = size(state_data, 2)
    
    # Plot each top particle's trajectory
    for (rank, particle_idx) in enumerate(top_indices)
        trace = traces[particle_idx]
        log_weight = log_weights[particle_idx]
        
        # Extract component info to compute config_key
        component_idxs = Vector{Int}(undef, config.k_components)
        component_params = Vector{Dict}(undef, config.k_components)
        
        for k in 1:config.k_components
            idx, params = trace[k => :component]
            component_idxs[k] = idx
            component_params[k] = params
        end
        
        # Reconstruct the config_key (needed to retrieve policy from π_dist)
        particle_key = hash((component_idxs, [Dict(collect(p)) for p in component_params]))
        
        # Get the learned policy from π_dist (already trained)
        policy = RL.get_π_proposal(π_dist, particle_key, mdp, config)

        # Run simulation with learned policy
        sim_trace = stepthrough_sim(mdp, policy, n_timesteps)
        
        # Extract x,y coordinates from simulation trace
        sim_xs = [stateaction[1].x[1,1] for stateaction in sim_trace]
        sim_ys = [stateaction[1].x[1,2] for stateaction in sim_trace]
        
        # Compute color: darker for higher rank (darker = closer to black)
        # rank 1 -> very dark gray, rank n -> lighter gray
        # Interpolate grayscale value from 0.1 (very dark) to 0.7 (light gray)
        gray_value = 0.1 + (rank - 1.0) / max(1.0, top_n - 1.0) * 0.6
        
        # Use grayscale color from Plots (0 = black, 1 = white)
        marker_color = Gray(gray_value)
        
        # Plot line connecting the trajectory points (dotted, no legend)
        plot!(p, sim_xs, sim_ys;
                linewidth=1.5,
                linestyle=:dot,
                alpha=0.6,
                color=marker_color,
                label="")
        
        # Plot particle trajectory as scatter with darkness based on rank
        scatter!(p, sim_xs, sim_ys;
                label="$rank: $(@sprintf("%.2f", log_weight))",
                markersize=4,
                alpha=0.8,
                color=marker_color)
    end
    
    # Plot observed trajectory in red
    scatter!(p, obs_x, obs_y;
            label="observed",
            markersize=5,
            color=:red,
            markerstrokewidth=0)
    
    return p
end

"""
    plot_particle_filter_frame(state_data::Matrix{Float64}, t::Int, pf_state, config::InferenceConfig, 
                                component_fields::Vector, true_objective_fn::Function, mdp::KAgentPOMDP, 
                                agent_params::Dict, π_dist::ScoreΠDist; gridsize::Int=120, n_top::Int=10,
                                trace_from_current::Bool=true)

Generate a single frame visualization of the particle filter state at timestep t.

Shows:
- True objective function as heatmap
- Observed trajectory up to timestep t (bright to dull dots with connecting line)
- Top 10 particle predictions for remaining timesteps (gray lines, darkness by rank)

# Arguments
- `trace_from_current::Bool`: If true, trajectory traces start from current observed state.
  If false, traces start from the initial agent state. (default: true)

Returns a Plots.jl plot object.
"""
function plot_particle_filter_frame(state_data::Matrix{Float64}, t::Int, pf_state, 
                                     config::InferenceConfig, component_fields::Vector, 
                                     true_objective_fn::Function, mdp::KAgentPOMDP, 
                                     agent_params::Dict, π_dist::ScoreΠDist; gridsize::Int=120, n_top::Int=10,
                                     trace_from_current::Bool=true)
    
    # Create grid and plot true objective using utility function
    xs, ys = Utils._grid_from_mdp(mdp; gridsize=gridsize)
    Z = [true_objective_fn(x, y) for y in ys, x in xs]
    
    p = heatmap(xs, ys, Z; aspect_ratio=1, 
                title="Particle Filter State at Timestep $t",
                xlabel="x", ylabel="y", legend=false)
    
    # Plot observed trajectory up to timestep t
    obs_x, obs_y = Utils.xy_path_from_state_matrix(state_data[:, 1:t])
    
    # Dull dots for history
    if t > 1
        scatter!(p, obs_x[1:end-1], obs_y[1:end-1]; 
                label="history", markersize=3, color=:red, alpha=0.3,
                markerstrokewidth=0)
    end
    
    # Bright dot for current position
    scatter!(p, [obs_x[end]], [obs_y[end]]; 
            label="current obs", markersize=6, color=:red,
            markerstrokewidth=0)
    
    # Line connecting observations
    plot!(p, obs_x, obs_y; label="path", color=:red, alpha=0.5, linewidth=1)
    
    # Extract top particles
    traces = pf_state.traces
    log_weights = pf_state.log_weights
    
    # Get indices sorted by log_weights (descending)
    top_indices = sortperm(log_weights; rev=true)[1:min(n_top, length(log_weights))]
    
    n_timesteps = size(state_data, 2)
    n_remaining = n_timesteps - t
    
    # Determine starting point for trajectory traces based on keyword argument
    if trace_from_current
        # Trajectory traces start from current observed state
        current_state_obs = state_data[1:2, t]  # Extract x, y coordinates
        trace_starting_state = blindstart_KAgentState(mdp, reshape(current_state_obs, 1, 2))
    else
        # Trajectory traces start from initial agent state
        trace_starting_state = blindstart_KAgentState(mdp, agent_params[:start])
    end
    
    for (rank, particle_idx) in enumerate(top_indices)
        try
            # Extract component info directly from trace (following plot_particle_filter_explanation pattern)
            trace = traces[particle_idx]
            
            component_idxs = Vector{Int}(undef, config.k_components)
            component_params = Vector{Dict}(undef, config.k_components)
            
            for k in 1:config.k_components
                idx, params = trace[k => :component]
                component_idxs[k] = idx
                component_params[k] = params
            end
            
            # Reconstruct the particle key
            particle_key = hash((component_idxs, [Dict(collect(p)) for p in component_params]))
            
            # Get the learned policy from π_dist (already trained)
            policy = RL.get_π_proposal(π_dist, particle_key, mdp, config)

            # Simulate forward from chosen starting state for remaining timesteps
            if n_remaining > 0
                sim_trace = stepthrough_sim(mdp, policy, n_remaining; start_state=trace_starting_state)
            else
                sim_trace = []
            end
            
            # Extract x,y coordinates from simulation trace
            if length(sim_trace) > 0
                sim_xs = [stateaction[1].x[1,1] for stateaction in sim_trace]
                sim_ys = [stateaction[1].x[1,2] for stateaction in sim_trace]
                
                # Compute color: darker for higher rank
                gray_value = 0.1 + (rank - 1.0) / max(1.0, n_top - 1.0) * 0.6
                marker_color = Gray(gray_value)
                
                # Plot prediction as dashed line with darkness based on rank
                plot!(p, sim_xs, sim_ys;
                        linewidth=1.5,
                        linestyle=:dash,
                        alpha=0.6,
                        color=marker_color,
                        label="")
                
                scatter!(p, sim_xs, sim_ys;
                        label="",
                        markersize=3,
                        alpha=0.7,
                        color=marker_color,
                        markerstrokewidth=0)
            end
            
        catch e
            # Skip particles that fail (e.g., solver issues)
            @warn "Failed to process particle $particle_idx: $e"
        end
    end
    
    return p
end

"""
    make_particle_filter_frame_fn(true_objective_fn::Function, mdp::KAgentPOMDP, 
                                   agent_params::Dict, π_dist::ScoreΠDist;
                                   gridsize::Int=120, n_top::Int=10, predict_from_current::Bool=true)

Create a frame-generation function for use with `particle_filter(...; frame_fn=...)`.

The returned function has signature `(state_data, t, state, config) -> plot_object`.

# Arguments
- `true_objective_fn::Function`: Ground truth objective function (x, y) -> Real
- `mdp::KAgentPOMDP`: MDP instance
- `agent_params::Dict`: Agent configuration dictionary
- `π_dist::ScoreΠDist`: Cached policy distribution
- `gridsize::Int`: Resolution of heatmap grid (default: 120)
- `n_top::Int`: Number of top particles to visualize (default: 10)
- `trace_from_current::Bool`: If true, traces start from current observed state; 
  if false, from initial state (default: true)

# Returns
A closure function ready to pass to `particle_filter(...; frame_fn=...)`
"""
function make_particle_filter_frame_fn(true_objective_fn::Function, mdp::KAgentPOMDP, 
                                       agent_params::Dict, π_dist::ScoreΠDist;
                                       gridsize::Int=120, n_top::Int=10, trace_from_current::Bool=true)
    
    component_fields = nothing  # Will be bound from config when called
    
    return function frame_fn(state_data::Matrix{Float64}, t::Int, pf_state, config::InferenceConfig)
        if isnothing(component_fields)
            component_fields = [tuple[1] for tuple in config.component_tuples]
        end
        
        return plot_particle_filter_frame(state_data, t, pf_state, config, component_fields,
                                         true_objective_fn, mdp, agent_params, π_dist;
                                         gridsize=gridsize, n_top=n_top,
                                         trace_from_current=trace_from_current)
    end
end

"""
    animate_particle_filter_from_frames(frames::Vector; fps::Int=2)

Convert a sequence of plot frames into an animated GIF.

# Arguments
- `frames::Vector`: Vector of Plots.jl plot objects (from particle_filter(...; frame_fn=...)[2])
- `fps::Int`: Frames per second for animation (default: 2)

# Returns
A Plots.jl Animation object, saveable via `gif(anim, "filename.gif"; fps=fps)`
"""
function animate_particle_filter_from_frames(frames::Vector; fps::Int=2)
    anim = @animate for p in frames
        plot(p)
    end
    return (anim, fps)
end