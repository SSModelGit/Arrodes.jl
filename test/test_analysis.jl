using Gen
using Random
using GenParticleFilters
using POMDPs
using MuKumari
using Flux

import GeoInterface as GI

function consistent_pf_setup()
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

    mdp = build_kagent_pomdp(agent_params, x->(0.0, false))

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
        sp = POMDPs.@gen(:sp)(mdp, s, a, rng)
        o = shape_state_as_obs(mdp, sp)
        state_data[:, t] .= Float64.(o)
        s = sp
    end

    A = Float64.(Flux.onehotbatch(actions_used, alist))
    obs_aidx = onehot_cols_to_aidx(A)

    n_particles = 6
    pf_state = particle_filter(obs_aidx, π_dist, agent_params, state_data, n_particles)

    return mdp, π_dist, pf_state, obs_aidx, state_data, n_particles
end

@testset "Metric Functions" begin
    @testset "Metric functions availability" begin
        @test :pf_degeneracy in names(Arrodes)
        @test :objective_recon_metrics in names(Arrodes)
        @test :policy_match_acc in names(Arrodes)
    end

    @testset "Degeneracy metrics" begin
        mdp, π_dist, pf_state, obs_aidx, state_data, n_particles = consistent_pf_setup()

        deg = pf_degeneracy(pf_state, π_dist; n_particles=n_particles)
        @test isa(deg, NamedTuple)
        @test haskey(deg, :ess)
        @test isfinite(deg.ess)
        @test isa(deg.nunique, Int)
        @test isa(deg.collapsed, Bool)
    end

    @testset "Reconstruction metrics" begin
        mdp, π_dist, pf_state, obs_aidx, state_data, n_particles = consistent_pf_setup()

        recon = objective_recon_metrics(pf_state, π_dist, mdp; gridsize=20)
        @test isa(recon, NamedTuple)
        @test haskey(recon, :rmse_z)
        @test haskey(recon, :corr)
        @test isfinite(recon.rmse_z) || isnan(recon.rmse_z)
        @test isfinite(recon.corr) || isnan(recon.corr)
    end

    @testset "Policy matching accuracy" begin
        mdp, π_dist, pf_state, obs_aidx, state_data, n_particles = consistent_pf_setup()

        pm = policy_match_acc(pf_state, π_dist, Dict(:start=>[1.0 1.0]), state_data, obs_aidx)
        @test isa(pm, NamedTuple)
        @test haskey(pm, :acc)
        @test haskey(pm, :N)
        @test pm.N == length(obs_aidx)
        @test (isfinite(pm.acc) || isnan(pm.acc))
    end
end
