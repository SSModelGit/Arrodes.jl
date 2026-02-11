@testset "Metric Functions" begin
    @testset "Metric functions availability" begin
        @test :pf_degeneracy in names(Arrodes)
        @test :objective_recon_metrics in names(Arrodes)
        @test :policy_match_acc in names(Arrodes)
    end

    @testset "Degeneracy metrics" begin
        # Create mock weights for particle filter
        log_weights = [-1.0, -2.0, -3.0, -10.0, -10.0]
        
        # Test degeneracy calculation
        deg = pf_degeneracy(log_weights)
        @test isa(deg, Real)
        @test deg >= 0
        @test deg <= 1
    end

    @testset "Reconstruction metrics" begin
        # Test that the function exists and is callable
        @test isa(objective_recon_metrics, Function)
    end

    @testset "Policy matching accuracy" begin
        # Test that the function exists and is callable
        @test isa(policy_match_acc, Function)
    end

    @testset "Particle filter effective sample size" begin
        # Create mock log weights
        log_weights = [-0.5, -0.6, -0.7, -0.8, -0.9]
        
        # GenParticleFilters has ESS computation
        @test :effective_sample_size in names(GenParticleFilters)
    end

    @testset "Metric numerical stability" begin
        # Test with extreme values
        log_weights_extreme = [-1000.0, -1001.0, -1002.0]
        
        deg = pf_degeneracy(log_weights_extreme)
        @test isfinite(deg)
        @test !isnan(deg)
    end
end