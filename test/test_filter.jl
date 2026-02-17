using Gen
using Random
using GenParticleFilters
using POMDPs
using MuKumari
using Flux

import GeoInterface as GI

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
        
        # Simulate the cfg-only model (gen_fourier_bank_fixed samples K internally)
        trace = Gen.simulate(gen_fourier_bank_fixed, (cfg,))
        @test isa(trace, Gen.Trace)
        
        bank = Gen.get_retval(trace)
        @test isa(bank, NamedTuple)
        @test isa(bank.K, Int)
        @test 1 <= bank.K <= cfg.Kmax
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
        rng = Random.MersenneTwister(1234)
        
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
        # Add two simple square obstacles so nearest_obstacles can return k=2
        agent_params = Dict(
            :start => [1.0 1.0],
            :dimensions => (0.0, 10.0),
            :menv => menv,
            :obcs => [
                GI.Polygon([[(2.0, 2.0), (2.0, 3.0), (3.0, 3.0), (3.0, 2.0), (2.0, 2.0)]]),
                GI.Polygon([[(5.0, 5.0), (5.0, 6.0), (6.0, 6.0), (6.0, 5.0), (5.0, 5.0)]])
            ]
        )

        # Build a minimal KAgentPOMDP for shaping observations
        mdp = build_kagent_pomdp(agent_params, x->0.0)

        # Prepare action mapping utilities to pass as mdp_params into ScoreΠDist
        alist = collect(POMDPs.actions(mdp))
        a_1hot = sym -> Float64.(Flux.onehot(sym, alist))
        a_1hotall = Float64.(Flux.onehotbatch(alist, alist))

        # Instantiate ScoreΠDist with mdp_params so downstream code can use action mappings
        π_dist = ScoreΠDist(mdp_params = [alist, a_1hot, a_1hotall], fourier_cfg = FourierDiscreteCfg())

        cols = 3
        obs0 = shape_state_as_obs(mdp, blindstart_KAgentState(mdp, mdp.start))
        rows = length(obs0)
        state_data = zeros(Float64, rows, cols)

        # simulate movements using POMDPs.gen (random actions) and record actions
        actions_used = Vector{Symbol}(undef, cols)
        rng = Random.MersenneTwister(1234)
        s = blindstart_KAgentState(mdp, mdp.start)
        for t in 1:cols
            a = alist[rand(rng, 1:length(alist))]
            actions_used[t] = a
            res = POMDPs.gen(mdp, s, a, rng)
            s = res.sp
            o = shape_state_as_obs(mdp, s)
            state_data[:, t] .= Float64.(o)
        end

        trace = Gen.simulate(inference_model, (1, π_dist, agent_params, state_data))
        @test isa(trace, Gen.Trace)

        key = Gen.get_retval(trace)
        @test isa(key, Tuple)
        @test key in π_dist.prop_names
        # Validate that the key decodes to valid Fourier parameters
        ff_key = decode_fourier_key(key, π_dist.fourier_cfg)
        @test isa(ff_key, NamedTuple)
        numeric_vals_k = vcat(ff_key.fx, ff_key.fy, ff_key.A, ff_key.ϕ)
        @test all(isfinite.(numeric_vals_k))
    end

    @testset "particle_filter functional" begin
        spec = MuEnvSpec()
        menv = build_shared_menv(spec)
        # Add two simple square obstacles so nearest_obstacles can return k=2
        agent_params = Dict(
            :start => [1.0 1.0],
            :dimensions => (0.0, 10.0),
            :menv => menv,
            :obcs => [
                GI.Polygon([[(2.0, 2.0), (2.0, 3.0), (3.0, 3.0), (3.0, 2.0), (2.0, 2.0)]]),
                GI.Polygon([[(5.0, 5.0), (5.0, 6.0), (6.0, 6.0), (6.0, 5.0), (5.0, 5.0)]])
            ]
        )

        mdp = build_kagent_pomdp(agent_params, x->0.0)

        # Prepare action mapping utilities and ScoreΠDist as above
        alist = collect(POMDPs.actions(mdp))
        a_1hot = sym -> Float64.(Flux.onehot(sym, alist))
        a_1hotall = Float64.(Flux.onehotbatch(alist, alist))
        π_dist = ScoreΠDist(mdp_params = [alist, a_1hot, a_1hotall], fourier_cfg = FourierDiscreteCfg())

        cols = 3
        obs0 = shape_state_as_obs(mdp, blindstart_KAgentState(mdp, mdp.start))
        rows = length(obs0)
        state_data = zeros(Float64, rows, cols)

        actions_used = Vector{Symbol}(undef, cols)
        rng = Random.MersenneTwister(1234)
        s = blindstart_KAgentState(mdp, mdp.start)
        for t in 1:cols
            a = alist[rand(rng, 1:length(alist))]
            actions_used[t] = a
            res = POMDPs.gen(mdp, s, a, rng)
            s = res.sp
            o = shape_state_as_obs(mdp, s)
            state_data[:, t] .= Float64.(o)
        end

        # Convert actions_used to one-hot columns then to indices using Flux
        A = Float64.(Flux.onehotbatch(actions_used, alist))
        observations = onehot_cols_to_aidx(A)

        n_particles = 6
        state = particle_filter(observations, π_dist, agent_params, state_data, n_particles)
        @test isa(state, Any)

        traces = get_traces(state)
        @test isa(traces, Vector)
        @test length(traces) == n_particles
        # Each trace should be a Gen.Trace and have a valid choices map
        @test all(t -> isa(t, Gen.Trace), traces)
        @test all(tr -> isa(Gen.get_choices(tr), Gen.ChoiceMap), traces)

        # Check log-weights are finite and normalize to proper weights
        logw = get_log_weights(state)
        @test isa(logw, AbstractVector)
        finite_mask = isfinite.(logw)
        @test any(finite_mask)  # at least one finite weight
        lw = logw[finite_mask]
        m = maximum(lw)
        w = exp.(lw .- m)
        Z = sum(w)
        weights = w ./ Z
        @test all(isfinite.(weights)) && all(weights .>= 0.0)
        @test abs(sum(weights) - 1.0) < 1e-8

        ess = effective_sample_size(state)
        @test isfinite(ess) && ess >= 0
        # Assert a minimum ESS to avoid fully degenerate particle sets
        min_ess = max(1, n_particles * 0.1)
        @test ess >= min_ess
    end
end