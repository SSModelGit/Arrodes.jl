using Gen
using Random
using GenParticleFilters
using POMDPs
using MuKumari
using Flux

import GeoInterface as GI

"""
    setup_observable_agent(; rng=Random.default_rng())

Create a mock observable agent (MDP) with ground-truth objective.
This is the "true" system whose objective we're trying to infer.

Returns: (mdp, agent_params, true_objective_fn)
"""
function setup_observable_agent(; rng=Random.default_rng())
    spec = MuEnvSpec()
    menv = build_shared_menv(spec)

    agent_params = Dict(
        :start => [1.0 1.0],
        :dimensions => (0.0, 10.0),
        :menv => menv,
        :obcs => [
            GI.Polygon([[(2.0, 2.0), (2.0, 3.0), (3.0, 3.0), (3.0, 2.0), (2.0, 2.0)]]),
            GI.Polygon([[(5.0, 5.0), (5.0, 6.0), (6.0, 6.0), (6.0, 5.0), (5.0, 5.0)]])
        ]
    )

    # Define ground-truth objective: mixture of Fourier and RBF
    # Fourier component: 0.6 * cos(0.5*x + 0.5*y + π/4)
    # RBF component: 0.4 * exp(-((x-5)^2 + (y-5)^2) / 0.5^2)
    true_objective_fn = (x, y) -> (
        0.6 * cos(0.5*x + 0.5*y + π/4) +
        0.4 * exp(-((x-5.0)^2 + (y-5.0)^2) / (2*0.5^2))
    )

    # Wrap objective in POMDP format: (reward::Float64, done::Bool)
    mdp_objective = x -> (true_objective_fn(x.x[1,1], x.x[1,2]), false)

    mdp = Utils.build_kagent_pomdp(agent_params, mdp_objective)

    return mdp, agent_params, true_objective_fn
end

"""
    generate_observations(mdp::KAgentPOMDP, n_timesteps::Int; rng=Random.default_rng())

Generate synthetic trajectory data from the observable agent using random actions.

Returns: (observations::Vector{Int}, state_data::Matrix{Float64}, alist::Vector)
"""
function generate_observations(mdp::KAgentPOMDP, n_timesteps::Int; rng=Random.default_rng())
    alist = collect(POMDPs.actions(mdp))

    # Initialize state trajectory container
    obs0 = shape_state_as_obs(mdp, blindstart_KAgentState(mdp, mdp.start))
    obs_dim = length(obs0)
    state_data = zeros(Float64, obs_dim, n_timesteps)

    # Rollout: random actions, record observations
    actions_used = Vector{Symbol}(undef, n_timesteps)
    s = blindstart_KAgentState(mdp, mdp.start)

    for t in 1:n_timesteps
        a = alist[rand(rng, 1:length(alist))]
        actions_used[t] = a
        res = POMDPs.gen(mdp, s, a, rng)
        s = res.sp
        o = shape_state_as_obs(mdp, s)
        state_data[:, t] .= Float64.(o)
    end

    # Convert symbolic actions to integer indices
    A = Float64.(Flux.onehotbatch(actions_used, alist))
    observations = Utils.onehot_cols_to_aidx(A)

    return observations, state_data, alist
end

"""
    setup_inference_config(component_tuples::Vector{Tuple}, agent_params::Dict)

Construct InferenceConfig with component switch and sampler from provided component tuples.

Returns: InferenceConfig
"""
function setup_inference_config(component_tuples::Vector, agent_params::Dict)
    # Build component parameter switch and sampler
    param_switch, component_fields = Priors.build_component_param_switch(component_tuples)
    component_type_sampler = Priors.component_type_sampler(component_fields)

    # Configure RL training parameters
    rl_config = RLConfig(
        n_iterations=200,
        epochs=2,
        batch_size=256,
        temperature=1.5
    )

    # Bundle all inference parameters
    config = InferenceConfig(
        component_tuples=component_tuples,
        component_params_switch=param_switch,
        component_type_sampler=component_type_sampler,
        k_components=1,  # Infer single component
        rl_config=rl_config,
        agent_params=agent_params
    )

    return config
end

"""
    run_inference_pipeline(observations, state_data, config, alist; n_particles=20, rng=Random.default_rng())

Execute the full SMC³ particle filter for objective inference.

Returns: pf_state (particle filter state with posterior over component configurations)
"""
function run_inference_pipeline(observations, state_data, config, alist; 
                                n_particles=20, rng=Random.default_rng())
    # Initialize network caching infrastructure
    π_dist = ScoreΠDist(
        mdp_params=[
            alist,
            a -> Float64.(Flux.onehot(a, alist)),
            Float64.(Flux.onehotbatch(alist, alist))
        ]
    )

    # Run particle filter
    pf_state = Inference.particle_filter(
        observations,
        config,
        π_dist,
        state_data,
        n_particles
    )

    return pf_state
end

@testset "Default Inference Pipeline - Fourier & RBF" begin

    @testset "Setup: Observable Agent Creation" begin
        mdp, agent_params, true_obj = setup_observable_agent(rng=Random.MersenneTwister(42))
        
        @test isa(mdp, KAgentPOMDP)
        @test haskey(agent_params, :start)
        @test haskey(agent_params, :dimensions)
        @test haskey(agent_params, :menv)
        @test isa(true_obj, Function)
        
        # Test objective evaluation at a point
        val = true_obj(5.0, 5.0)
        @test isa(val, Float64)
        @test isfinite(val)
    end

    @testset "Setup: Observation Generation" begin
        mdp, agent_params, _ = setup_observable_agent(rng=Random.MersenneTwister(42))
        rng = Random.MersenneTwister(123)
        n_timesteps = 10
        
        observations, state_data, alist = generate_observations(mdp, n_timesteps; rng=rng)
        
        @test isa(observations, Vector{Int})
        @test length(observations) == n_timesteps
        @test all(1 .<= observations .<= length(alist))
        
        @test isa(state_data, Matrix{Float64})
        @test size(state_data, 2) == n_timesteps
        @test all(isfinite.(state_data))
    end

    @testset "Setup: Component Field Definitions" begin
        # Create equally-weighted Fourier and RBF prior set
        fourier_field = RandomFourierField(amplitude_max=10.0, freq_max=π)
        rbf_field = RadialBasisField(
            x_min=0.0, x_max=10.0, 
            y_min=0.0, y_max=10.0,
            amp_min=0.1, amp_max=10.0, 
            σ=0.5
        )

        component_tuples = [
            (fourier_field, Priors.fourier_params_sampler(fourier_field)),
            (rbf_field, Priors.rbf_params_sampler(rbf_field))
        ]

        @test length(component_tuples) == 2
        @test isa(component_tuples[1][1], RandomFourierField)
        @test isa(component_tuples[2][1], RadialBasisField)
        @test component_tuples[1][2] !== nothing
        @test component_tuples[2][2] !== nothing
    end

    @testset "Setup: InferenceConfig Creation" begin
        mdp, agent_params, _ = setup_observable_agent(rng=Random.MersenneTwister(42))
        
        fourier_field = RandomFourierField(amplitude_max=10.0, freq_max=π)
        rbf_field = RadialBasisField(
            x_min=0.0, x_max=10.0, 
            y_min=0.0, y_max=10.0,
            amp_min=0.1, amp_max=10.0, 
            σ=0.5
        )

        component_tuples = [
            (fourier_field, Priors.fourier_params_sampler(fourier_field)),
            (rbf_field, Priors.rbf_params_sampler(rbf_field))
        ]

        config = setup_inference_config(component_tuples, agent_params)

        @test isa(config, InferenceConfig)
        @test config.k_components == 1
        @test length(config.component_tuples) == 2
        @test isa(config.component_params_switch, Gen.Switch)
        @test config.component_type_sampler !== nothing
        @test isa(config.rl_config, RLConfig)
    end

    @testset "Generative Model: inference_model_continuous Execution" begin
        mdp, agent_params, _ = setup_observable_agent(rng=Random.MersenneTwister(42))
        
        fourier_field = RandomFourierField(amplitude_max=10.0, freq_max=π)
        rbf_field = RadialBasisField(
            x_min=0.0, x_max=10.0, 
            y_min=0.0, y_max=10.0,
            amp_min=0.1, amp_max=10.0, 
            σ=0.5
        )

        component_tuples = [
            (fourier_field, Priors.fourier_params_sampler(fourier_field)),
            (rbf_field, Priors.rbf_params_sampler(rbf_field))
        ]

        config = setup_inference_config(component_tuples, agent_params)

        # Generate mock observations
        observations, state_data, alist = generate_observations(mdp, 5; rng=Random.MersenneTwister(123))

        # Create π_dist for caching
        π_dist = ScoreΠDist(
            mdp_params=[
                alist,
                a -> Float64.(Flux.onehot(a, alist)),
                Float64.(Flux.onehotbatch(alist, alist))
            ]
        )

        # Trace the generative model
        trace = Gen.simulate(
            Inference.inference_model_continuous,
            (config, observations, state_data, π_dist)
        )

        @test isa(trace, Gen.Trace)
        
        # Return value should be component indices
        component_indices = Gen.get_retval(trace)
        @test isa(component_indices, Vector{Int})
        @test length(component_indices) == config.k_components
        @test all(1 .<= component_indices .<= length(component_tuples))
    end

    @testset "Inference Pipeline: Full End-to-End" begin
        mdp, agent_params, _ = setup_observable_agent(rng=Random.MersenneTwister(42))
        
        fourier_field = RandomFourierField(amplitude_max=10.0, freq_max=π)
        rbf_field = RadialBasisField(
            x_min=0.0, x_max=10.0, 
            y_min=0.0, y_max=10.0,
            amp_min=0.1, amp_max=10.0, 
            σ=0.5
        )

        component_tuples = [
            (fourier_field, Priors.fourier_params_sampler(fourier_field)),
            (rbf_field, Priors.rbf_params_sampler(rbf_field))
        ]

        config = setup_inference_config(component_tuples, agent_params)

        # Generate observations
        observations, state_data, alist = generate_observations(mdp, 8; rng=Random.MersenneTwister(123))

        # Run inference
        n_particles = 6
        pf_state = run_inference_pipeline(
            observations, state_data, config, alist; 
            n_particles=n_particles,
            rng=Random.MersenneTwister(42)
        )

        # Validate particle filter state
        traces = get_traces(pf_state)
        @test isa(traces, Vector)
        @test length(traces) == n_particles
        @test all(tr -> isa(tr, Gen.Trace), traces)

        logw = get_log_weights(pf_state)
        @test isa(logw, AbstractVector)
        @test length(logw) == n_particles
    end

    @testset "Posterior Analysis: Best Particle Extraction" begin
        mdp, agent_params, _ = setup_observable_agent(rng=Random.MersenneTwister(42))
        
        fourier_field = RandomFourierField(amplitude_max=10.0, freq_max=π)
        rbf_field = RadialBasisField(
            x_min=0.0, x_max=10.0, 
            y_min=0.0, y_max=10.0,
            amp_min=0.1, amp_max=10.0, 
            σ=0.5
        )

        component_tuples = [
            (fourier_field, Priors.fourier_params_sampler(fourier_field)),
            (rbf_field, Priors.rbf_params_sampler(rbf_field))
        ]

        config = setup_inference_config(component_tuples, agent_params)
        component_fields = [t[1] for t in component_tuples]

        # Generate observations
        observations, state_data, alist = generate_observations(mdp, 8; rng=Random.MersenneTwister(123))

        # Run inference
        n_particles = 6
        pf_state = run_inference_pipeline(
            observations, state_data, config, alist; 
            n_particles=n_particles,
            rng=Random.MersenneTwister(42)
        )

        # Extract best particle
        best_idx, best_weight, comp_idxs, comp_params, obj_fn = 
            Inference.best_particle(pf_state, config, component_fields)

        @test isa(best_idx, Int)
        @test 1 <= best_idx <= n_particles
        @test isa(best_weight, Real)
        @test isfinite(best_weight)

        @test isa(comp_idxs, Vector{Int})
        @test length(comp_idxs) == config.k_components
        @test all(1 .<= comp_idxs .<= length(component_tuples))

        @test isa(comp_params, Vector{Dict})
        @test length(comp_params) == config.k_components

        @test isa(obj_fn, Function)
        val = obj_fn(5.0, 5.0)
        @test isa(val, Float64)
        @test isfinite(val)
    end

    @testset "Trace Analysis: Component Info Extraction" begin
        mdp, agent_params, _ = setup_observable_agent(rng=Random.MersenneTwister(42))
        
        fourier_field = RandomFourierField(amplitude_max=10.0, freq_max=π)
        rbf_field = RadialBasisField(
            x_min=0.0, x_max=10.0, 
            y_min=0.0, y_max=10.0,
            amp_min=0.1, amp_max=10.0, 
            σ=0.5
        )

        component_tuples = [
            (fourier_field, Priors.fourier_params_sampler(fourier_field)),
            (rbf_field, Priors.rbf_params_sampler(rbf_field))
        ]

        config = setup_inference_config(component_tuples, agent_params)

        # Generate observations
        observations, state_data, alist = generate_observations(mdp, 8; rng=Random.MersenneTwister(123))

        # Create π_dist
        π_dist = ScoreΠDist(
            mdp_params=[
                alist,
                a -> Float64.(Flux.onehot(a, alist)),
                Float64.(Flux.onehotbatch(alist, alist))
            ]
        )

        # Simulate generative model
        trace = Gen.simulate(
            Inference.inference_model_continuous,
            (config, observations, state_data, π_dist)
        )

        # Extract component info from trace
        info = Inference.extract_component_info(trace, config)

        @test isa(info, Dict)
        @test haskey(info, :component_indices)
        @test haskey(info, :component_params)

        @test isa(info[:component_indices], Vector{Int})
        @test isa(info[:component_params], Vector{Dict})
        @test length(info[:component_indices]) == config.k_components
        @test length(info[:component_params]) == config.k_components
    end

    @testset "Objective Reconstruction from Trace" begin
        mdp, agent_params, _ = setup_observable_agent(rng=Random.MersenneTwister(42))
        
        fourier_field = RandomFourierField(amplitude_max=10.0, freq_max=π)
        rbf_field = RadialBasisField(
            x_min=0.0, x_max=10.0, 
            y_min=0.0, y_max=10.0,
            amp_min=0.1, amp_max=10.0, 
            σ=0.5
        )

        component_tuples = [
            (fourier_field, Priors.fourier_params_sampler(fourier_field)),
            (rbf_field, Priors.rbf_params_sampler(rbf_field))
        ]

        config = setup_inference_config(component_tuples, agent_params)

        # Generate observations
        observations, state_data, alist = generate_observations(mdp, 8; rng=Random.MersenneTwister(123))

        # Create π_dist
        π_dist = ScoreΠDist(
            mdp_params=[
                alist,
                a -> Float64.(Flux.onehot(a, alist)),
                Float64.(Flux.onehotbatch(alist, alist))
            ]
        )

        # Simulate generative model
        trace = Gen.simulate(
            Inference.inference_model_continuous,
            (config, observations, state_data, π_dist)
        )

        # Reconstruct objective
        reconstructed_obj = Inference.reconstruct_objective_from_trace(trace, config)

        @test isa(reconstructed_obj, Function)
        
        # Test evaluation at multiple points
        test_points = [(1.0, 1.0), (5.0, 5.0), (9.0, 9.0)]
        for (x, y) in test_points
            val = reconstructed_obj(x, y)
            @test isa(val, Float64)
            @test isfinite(val)
        end
    end

end
