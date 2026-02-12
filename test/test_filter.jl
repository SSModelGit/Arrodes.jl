using Gen
using Random
using GenParticleFilters

@testset "Gen-based Generative Model" begin
    @testset "Gen model functions exist" begin
        @test :gen_K in names(Arrodes)
        @test :gen_mode_indices in names(Arrodes)
        @test :gen_fourier_bank_fixed in names(Arrodes)
        @test :inference_model in names(Arrodes)
    end

    @testset "Particle filter functions" begin
        @test :particle_filter in names(Arrodes)
    end

    @testset "gen_K model" begin
        # Test that gen_K is a valid Gen generative function
        @test isa(gen_K, Gen.GenerativeFunction)
        
        # Test tracing gen_K
        cfg = FourierDiscreteCfg(Kmax=5)
        trace = Gen.simulate(gen_K, (cfg,))
        @test isa(trace, Gen.Trace)
        
        # Extract the returned value
        K = Gen.get_retval(trace)
        @test isa(K, Int)
        @test 1 <= K <= cfg.Kmax
    end

    @testset "gen_mode_indices model" begin
        @test isa(gen_mode_indices, Gen.GenerativeFunction)
        
        cfg = FourierDiscreteCfg(Kmax=5, P=16)
        K = 3
        trace = Gen.simulate(gen_mode_indices, (K, cfg))
        @test isa(trace, Gen.Trace)
        
        indices = Gen.get_retval(trace)
        @test isa(indices, Vector)
    end

    @testset "gen_fourier_bank_fixed model" begin
        @test isa(gen_fourier_bank_fixed, Gen.GenerativeFunction)
        
        cfg = FourierDiscreteCfg(Kmax=5, P=16)
        K = 3
        
        # Simulate the model
        trace = Gen.simulate(gen_fourier_bank_fixed, (K, cfg))
        @test isa(trace, Gen.Trace)
        
        key = Gen.get_retval(trace)
        @test isa(key, Vector)
        @test length(key) == K
    end

    @testset "inference_model" begin
        @test isa(inference_model, Gen.GenerativeFunction)
    end

    @testset "Particle filter execution" begin
        cfg = FourierDiscreteCfg(Kmax=3, P=16)
        
        # Create synthetic observations (simple test data)
        observations = [0.5, 0.3, 0.7]  # Mock observation data
        
        # Test that particle_filter is callable
        @test isa(particle_filter, Function)
    end

    @testset "Generative function traceable" begin
        cfg = FourierDiscreteCfg(Kmax=5, P=16)
        
        # Test that we can trace and extract choices
        trace = Gen.simulate(gen_K, (cfg,))
        choices = Gen.get_choices(trace)
        @test isa(choices, Gen.ChoiceMap)
    end

    @testset "Fourier key decoding consistency" begin
        cfg = FourierDiscreteCfg(Kmax=5, P=16)
        rng = Random.default_rng()
        
        # Generate key and decode it
        key = sample_fourier_key(cfg; K_override=2, rng=rng)
        ff = decode_fourier_key(key, cfg)
        
        # Verify it returns a NamedTuple with numeric fields
        @test isa(ff, NamedTuple)
        numeric_vals = vcat(ff.fx, ff.fy, ff.A, ff.ϕ)
        @test all(isfinite.(numeric_vals))
    end

    @testset "inference_model functional" begin
        spec = MuEnvSpec()
        menv = build_shared_menv(spec)
        π_dist = ScoreΠDist()
        agent_params = Dict(:start => [1.0 1.0], :dimensions => (0.0, 10.0), :menv => menv, :obcs => Any[])
        # TODO: Bad
        # Use the function `shape_state_as_obs` from `MuKumari` to do the observation vector shaping.
        state_data = [1.0 2.0 3.0; 1.0 2.0 3.0]

        trace = Gen.simulate(inference_model, (1, π_dist, agent_params, state_data))
        @test isa(trace, Gen.Trace)

        key = Gen.get_retval(trace)
        @test isa(key, Tuple)
        @test key in π_dist.prop_names
    end

    @testset "particle_filter functional" begin
        spec = MuEnvSpec()
        menv = build_shared_menv(spec)
        π_dist = ScoreΠDist()
        agent_params = Dict(:start => [1.0 1.0], :dimensions => (0.0, 10.0), :menv => menv, :obcs => Any[])
        state_data = [1.0 2.0 3.0; 1.0 2.0 3.0]
        observations = [1, 1, 1]

        n_particles = 6
        state = particle_filter(observations, π_dist, agent_params, state_data, n_particles)
        @test isa(state, Any)

        traces = get_traces(state)
        @test isa(traces, Vector)
        @test length(traces) == n_particles

        ess = effective_sample_size(state)
        @test isfinite(ess) && ess >= 0
    end
end