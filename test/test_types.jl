@testset "Custom Types" begin
    @testset "FourierDiscreteCfg" begin
        cfg = FourierDiscreteCfg()
        @test cfg.Kmax == 10
        @test cfg.λK == 0.35
        @test cfg.Δf == 0.1
        @test cfg.Fmax_i == 10
        @test cfg.ΔA == 0.1
        @test cfg.Amax_i == 1
        @test cfg.P == 32
        @test cfg.freq_mag_decay == 0.0
    end

    @testset "FourierDiscreteCfg with custom parameters" begin
        cfg = FourierDiscreteCfg(Kmax=5, λK=0.5, Δf=0.2, P=16)
        @test cfg.Kmax == 5
        @test cfg.λK == 0.5
        @test cfg.Δf == 0.2
        @test cfg.P == 16
    end

    @testset "ScoreΠDist" begin
        dist = ScoreΠDist()
        @test isa(dist, ScoreΠDist)
        @test isa(dist.prop_names, Vector)
    end

    @testset "ScoreΠDist with parameters" begin
        dist = ScoreΠDist(prop_names=[:sin, :cos, :exp])
        @test dist.prop_names == [:sin, :cos, :exp]
    end

    @testset "MuEnvSpec" begin
        spec = MuEnvSpec()
        @test isa(spec, MuEnvSpec)
        @test spec.M == 3
        @test spec.variant == :default_shared
        @test spec.μ_order == [:sin, :exp, :lin]
    end

    @testset "ActionDirac Distribution" begin
        # Test that ActionDirac is a valid Gen distribution
        @test actiondirac isa Gen.Distribution
        
        # Test random sampling
        test_vec = [0.1, 0.8, 0.1]
        sample = Gen.random(actiondirac, test_vec)
        @test sample == test_vec
        
        # Test logpdf with matching argmax
        logp = Gen.logpdf(actiondirac, [0.1, 0.9, 0.0], [0.0, 0.95, 0.05])
        @test logp == 0.0
        
        # Test logpdf with non-matching argmax
        logp = Gen.logpdf(actiondirac, [0.8, 0.1, 0.1], [0.1, 0.8, 0.1])
        @test logp == -Inf
    end

    @testset "METHOD_LABELS" begin
        @test length(METHOD_LABELS) == 2
        @test "Open-Ended SIPS" in METHOD_LABELS
        @test "IQ-SIPS" in METHOD_LABELS
    end
end