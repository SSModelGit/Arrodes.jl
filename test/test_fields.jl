using Random

@testset "Field Generation" begin
    @testset "Field functions availability" begin
        # Verify that field-related functions are exported
        @test :make_pomdp_objective_from_field in names(Arrodes)
        @test :objective_grid_from_field in names(Arrodes)
        @test :objective_grid_from_mdp in names(Arrodes)
    end

    @testset "Fourier key sampling and decoding" begin
        cfg = FourierDiscreteCfg(Kmax=5, P=16)
        rng = Random.default_rng()
        
        # Test sampling a fourier key (returns tuple: (K, fx_idx, fy_idx, A_idx, ϕ_idx))
        key = sample_fourier_key(cfg; rng=rng)
        @test isa(key, Tuple)
        K = key[1]
        @test isa(K, Int)
        @test 1 <= K <= cfg.Kmax

        # ensure discrete index vectors have length K
        @test length(key[2]) == K
        @test length(key[3]) == K
        @test length(key[4]) == K
        @test length(key[5]) == K

        # Test decoding fourier key (returns NamedTuple with fields K, fx, fy, A, ϕ, ...)
        ff = decode_fourier_key(key, cfg)
        @test isa(ff, NamedTuple)
        @test ff.K == K
        @test length(ff.fx) == K
    end

    @testset "Fourier basis functions" begin
        # Test K_probs
        cfg = FourierDiscreteCfg(Kmax=5, λK=0.35)
        probs = K_probs(cfg)
        @test length(probs) == cfg.Kmax
        @test isapprox(sum(probs), 1.0, atol=1e-6)
        @test all(probs .>= 0)
    end

    @testset "Frequency grid" begin
        cfg = FourierDiscreteCfg(Δf=0.1, Fmax_i=5)
        freqs, freq_probs = freq_bin_support_and_probs(cfg)
        @test isa(freqs, Vector)
        @test isa(freq_probs, Vector)
        @test length(freqs) == length(freq_probs)
        @test isapprox(sum(freq_probs), 1.0, atol=1e-6)
    end

    @testset "Amplitude grid" begin
        cfg = FourierDiscreteCfg(ΔA=0.1, Amax_i=2)
        amps, amp_probs = amp_bin_support_and_probs(cfg)
        @test isa(amps, Vector)
        @test isa(amp_probs, Vector)
        @test length(amps) == length(amp_probs)
        @test isapprox(sum(amp_probs), 1.0, atol=1e-6)
    end

    @testset "Phase grid" begin
        cfg = FourierDiscreteCfg(P=16)
        phases, phase_probs = phase_bin_support_and_probs(cfg)
        @test isa(phases, Vector)
        @test isa(phase_probs, Vector)
        @test length(phases) == cfg.P
        @test isapprox(sum(phase_probs), 1.0, atol=1e-6)
    end

    @testset "Index to value conversions" begin
        cfg = FourierDiscreteCfg(Δf=0.1, Fmax_i=5)
        
        f_val = f_from_i(0, cfg)
        @test isa(f_val, Float64)
        
        cfg_amp = FourierDiscreteCfg(ΔA=0.1, Amax_i=2)
        A_val = A_from_i(0, cfg_amp)
        @test isa(A_val, Float64)
        @test A_val >= 0
        
        cfg_phase = FourierDiscreteCfg(P=16)
        φ_val = ϕ_from_i(0, cfg_phase)
        @test isa(φ_val, Float64)
    end

    @testset "Scalar field generation" begin
        cfg = FourierDiscreteCfg(Kmax=3, P=16)
        rng = Random.default_rng()
        
        # Generate a random fourier key
        key = sample_fourier_key(cfg; K_override=2, rng=rng)
        bank = decode_fourier_key(key, cfg)
        
        # Create scalar field
        field = make_fourier_scalar_field(bank)
        @test isa(field, Function)
        
        # Test field evaluation
        test_point = [5.0, 5.0]
        val = field(test_point...)
        @test isa(val, Float64)
        @test isfinite(val)
    end

    @testset "POMDP field conversion" begin
        cfg = FourierDiscreteCfg(Kmax=2, P=16)
        key = sample_fourier_key(cfg; K_override=2, rng=Random.default_rng())
        bank = decode_fourier_key(key, cfg)
        field = make_fourier_scalar_field(bank)

        pomdp_obj = make_pomdp_objective_from_field(field)

        menv = build_shared_menv()
        agent_params = Dict(:start => [1.0 1.0], :dimensions => (0.0, 10.0), :menv => menv, :obcs => Any[])
        mdp = build_kagent_pomdp(agent_params, pomdp_obj; name="field_mdp")

        xs = range(0.0, 10.0; length=5)
        ys = range(0.0, 10.0; length=7)

        Zf = objective_grid_from_field(field, xs, ys)
        Zm = objective_grid_from_mdp(mdp, xs, ys)

        @test size(Zf) == (length(ys), length(xs))
        @test size(Zm) == size(Zf)
        @test isapprox(Zf, Zm; atol=1e-6)
    end

    @testset "Hamming distance between fourier keys" begin
        cfg = FourierDiscreteCfg(Kmax=5, P=16)
        rng = Random.default_rng()
        
        key1 = sample_fourier_key(cfg; rng=rng)
        key2 = sample_fourier_key(cfg; rng=rng)
        
        dist = hamming_fourier_key(key1, key2)
        @test isa(dist, Int)
        @test dist >= 0
    end
end