@testset "Deprecated generic objective fields remain isolated" begin
    rff = RandomFourierField(amplitude_max=2.0)
    rbf = RadialBasisField(σ=0.5)
    @test isfinite(make_component(rff, fourier_params_sampler(rff)(MersenneTwister(1)))(1.0, 2.0))
    @test isfinite(make_component(rbf, rbf_params_sampler(rbf)(MersenneTwister(1)))(1.0, 2.0))
end
