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
        
        # Test sampling a fourier key
        key = sample_fourier_key(cfg; rng=rng)
        @test isa(key, Vector)
        @test length(key) > 0
        
        # Test decoding fourier key
        ff = decode_fourier_key(key, cfg)
        @test isa(ff, Vector)
        @test length(ff) == length(key)
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
        
        # Create scalar field
        field = make_fourier_scalar_field(key, cfg)
        @test isa(field, Function)
        
        # Test field evaluation
        test_point = [5.0, 5.0]
        val = field(test_point)
        @test isa(val, Float64)
        @test isfinite(val)
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