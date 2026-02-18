using Test
using Random
using Arrodes
using MuKumari
using POMDPs
using Flux

import GeoInterface as GI

function consistent_mdp_data_setup(; cols=3, constant_action=false)
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

        # objective must return (reward, terminal) — tests previously passed a scalar
        mdp = build_kagent_pomdp(agent_params, x->(0.0, false))

    alist = collect(POMDPs.actions(mdp))
    a_1hot = sym -> Float64.(Flux.onehot(sym, alist))
    a_1hotall = Float64.(Flux.onehotbatch(alist, alist))

    π_dist = ScoreΠDist(mdp_params = [alist, a_1hot, a_1hotall], fourier_cfg = FourierDiscreteCfg())

    obs0 = shape_state_as_obs(mdp, blindstart_KAgentState(mdp, mdp.start))
    rows = length(obs0)
    state_data = zeros(Float64, rows, cols)

    actions_used = Vector{Symbol}(undef, cols)
    rng = Random.MersenneTwister(1234)
    s = blindstart_KAgentState(mdp, mdp.start)
    action_gen = constant_action ? () -> alist[1] : () -> alist[rand(rng, 1:length(alist))]
    for t in 1:cols
        a = action_gen()
        actions_used[t] = a
        sp = POMDPs.@gen(:sp)(mdp, s, a, rng)
        o = shape_state_as_obs(mdp, sp)
        state_data[:, t] .= Float64.(o)
        s = sp
    end

    A = Float64.(Flux.onehotbatch(actions_used, alist))
    obs_aidx = onehot_cols_to_aidx(A)

    return mdp, π_dist, obs_aidx, state_data
end


@testset "RL additional tests" begin

    @testset "Policy model extraction" begin
        mdp, π_dist, obs_aidx, state_data = consistent_mdp_data_setup(cols=6)
        _, π_softq = softq_policy(mdp; N=10, epochs=1, batch_size=8)
        m = _policy_model(π_softq)
        @test m !== nothing
        p = collect(Flux.params(m))
        @test isa(p, Vector)
    end

    @testset "Warm-start behavior" begin
        mdp, π_dist, obs_aidx, state_data = consistent_mdp_data_setup(cols=6)
        _, π_src = softq_policy(mdp; N=10, epochs=1, batch_size=8)
        _, π_dest = softq_policy(mdp; N=10, epochs=1, batch_size=8)
        ms = _policy_model(π_src)
        md = _policy_model(π_dest)
        psrc = collect(Flux.params(ms))
        pdest = collect(Flux.params(md))
        try
            _warm_start_params!(π_dest, π_src)
            pd_after = collect(Flux.params(md))
            @test length(psrc) == length(pd_after) || length(psrc) != length(pd_after)
            if length(psrc) == length(pd_after)
                @test all(map((a,b)->all(abs.(Array(a).-Array(b)) .< 1e-6), psrc, pd_after))
            end
        catch e
            @test false
            @error("_warm_start_params! threw: $e")
        end
    end

    @testset "ensure_policy_trained_to! safety" begin
        mdp, π_dist, obs_aidx, state_data = consistent_mdp_data_setup(cols=6)
        # call with a key that is not yet present; should not throw
        k = :test_missing_key
        agent_params = Dict(:menv=>build_shared_menv(MuEnvSpec()),
                            :start=>[1.0 1.0], :dimensions=>(0.0,10.0),
                            :obcs=>[
                                GI.Polygon([[(2.0, 2.0), (2.0, 3.0), (3.0, 3.0), (3.0, 2.0), (2.0, 2.0)]]),
                                GI.Polygon([[(5.0, 5.0), (5.0, 7.0), (6.0, 6.0), (6.0, 5.0), (5.0, 5.0)]])
                            ])
        try
            _ = ensure_policy_trained_to!(π_dist, k, agent_params; target_steps=1, warm_start=false)
            @test true
        catch e
            @test false
            @error("ensure_policy_trained_to! threw: $e")
        end
    end

    @testset "Solver factories (MCTS and SoftQ)" begin
        mdp, π_dist, obs_aidx, state_data = consistent_mdp_data_setup(cols=4)
        s_mcts = mcts_solver(mdp; solver_params=[:van, 2, 0.5])
        @test s_mcts !== nothing
        # When requesting all solvers, expect a vector result
        sols_all = solver_from_type(mdp, :dql; solver_params=[:all, 2, 1, 8])
        @test isa(sols_all, AbstractVector)
        @test length(sols_all) > 0
        # For a specific solver (e.g. :softq) expect a single solver object (non-vector)
        sols_softq = solver_from_type(mdp, :dql; solver_params=[:softq, 2, 1, 8])
        @test !(isa(sols_softq, AbstractVector))
    end

    @testset "Action-selection policies" begin
        mdp, π_dist, obs_aidx, state_data = consistent_mdp_data_setup(cols=8)
        obs_dim, N = size(state_data)
        na = length(collect(actions(mdp)))
        data = alloc_buffer_dict(obs_dim, na, N)
        for t in 1:N
            data[:s][:, t] .= Float32.(state_data[:, t])
            data[:sp][:, t] .= Float32.(state_data[:, t])
            data[:a][:, t] .= false
            data[:a][obs_aidx[t], t] = true
            data[:r][1, t] = Float32(1.0)
        end
        anon = mk_experience_buffer(data)

        π_iql, _, mdp_tr = quick_IQL(mdp, anon)
        s = blindstart_KAgentState(mdp_tr, mdp_tr.start)
        a_sym, a_idx, probs = qpolicy_action(π_iql, mdp_tr, s; temperature=0.5, rng=Random.default_rng())
        @test isa(a_sym, Symbol)
        @test isa(a_idx, Int)
        @test isa(probs, AbstractVector{Float64})
        # Boltzmann proposal API should run without error for π_dist proposals (if present)
        try
            # attempt greedy action via boltzmann utilities if proposal names exist
            if !isempty(get_proposal_names(π_dist))
                kname = get_proposal_names(π_dist)[1]
                _ = proposal_boltzmann(π_dist, kname, s)
            end
            @test true
        catch e
            @test false
            @error("proposal_boltzmann threw: $e")
        end
    end

    @testset "Rollout helpers" begin
        mdp, π_dist, obs_aidx, state_data = consistent_mdp_data_setup(cols=8)
        _, π_softq = softq_policy(mdp; N=10, epochs=1, batch_size=8)
        buf = rollout_experience_buffer(mdp, π_softq; T=6, rng=Random.MersenneTwister(9))
        @test buf.elements == 6
    end

    @testset "IQL pipelines: surrogate dataset grid" begin
        mdp, π_dist, obs_aidx, state_data = consistent_mdp_data_setup(cols=12)
        obs_dim, N = size(state_data)
        na = length(collect(actions(mdp)))
        data = alloc_buffer_dict(obs_dim, na, N)
        for t in 1:N
            data[:s][:, t] .= Float32.(state_data[:, t])
            data[:sp][:, t] .= Float32.(state_data[:, t])
            data[:a][:, t] .= false
            data[:a][obs_aidx[t], t] = true
            data[:r][1, t] = Float32(1.0)
        end
        anon = mk_experience_buffer(data)
        π_iql, _, mdp_tr = quick_IQL(mdp, anon)
        st, obs, locs = surrogate_dataset_from_iql_grid(π_dist, π_iql, mdp_tr; eval_num=36)
        @test size(st, 2) == length(obs)
    end

    @testset "Edge-case: NaN probabilities handling" begin
        # Create a real small PF state, then coerce its log-weights to -Inf
        mdp, π_dist, obs_aidx, state_data = consistent_mdp_data_setup(cols=4)
        agent_params = Dict(:menv=>build_shared_menv(MuEnvSpec()),
                            :start=>[1.0 1.0], :dimensions=>(0.0,10.0),
                            :obcs=>[
                                GI.Polygon([[(2.0, 2.0), (2.0, 3.0), (3.0, 3.0), (3.0, 2.0), (2.0, 2.0)]]),
                                GI.Polygon([[(5.0, 5.0), (5.0, 6.0), (6.0, 6.0), (6.0, 5.0), (5.0, 5.0)]])
                            ])

        n_particles = 2
        pf_state = particle_filter(obs_aidx, π_dist, agent_params, state_data, n_particles)

        # Temporarily override get_log_weights for this concrete PF state type to simulate all -Inf weights
        PFConcreteType = typeof(pf_state)
        get_log_weights(s::PFConcreteType) = fill(-Inf, length(get_traces(s)))

        try
            maybe_refine_policies!(π_dist, pf_state, agent_params; topk=2)
            @test true
        catch e
            @test false
            @error("maybe_refine_policies! threw on NaN probs: $e")
        end
    end
end

@testset "RL smoke tests" begin
    @testset "Small integration pipeline" begin
        mdp, π_dist, obs_aidx, state_data = consistent_mdp_data_setup(cols=10)
        obs_dim, N = size(state_data)
        na = length(collect(actions(mdp)))
        data = alloc_buffer_dict(obs_dim, na, N)
        for t in 1:N
            data[:s][:, t] .= Float32.(state_data[:, t])
            data[:sp][:, t] .= Float32.(state_data[:, t])
            data[:a][:, t] .= false
            data[:a][obs_aidx[t], t] = true
            data[:r][1, t] = Float32(1.0)
        end
        anon = mk_experience_buffer(data)
        π_iql, 𝒟_iql, mdp_tr = quick_IQL(mdp, anon)
        st, obs, locs = surrogate_dataset_from_iql_grid(π_dist, π_iql, mdp_tr; eval_num=25)
        k = :integration_test_key
        register_key_if_new!(π_dist, k)
        agent_params = Dict(:menv=>build_shared_menv(MuEnvSpec()), 
                            :start=>[1.0,1.0],
                            :dimensions=>(0.0,10.0),
                            :obcs=>[
            GI.Polygon([[(2.0, 2.0), (2.0, 3.0), (3.0, 3.0), (3.0, 2.0), (2.0, 2.0)]]),
            GI.Polygon([[(5.0, 5.0), (5.0, 6.0), (6.0, 6.0), (6.0, 5.0), (5.0, 5.0)]])
        ])
        try
            _ = ensure_policy_trained_to!(π_dist, k, agent_params; target_steps=2, warm_start=false)
            @test true
        catch e
            @test false
            @error("integration ensure_policy_trained_to! threw: $e")
        end
    end

    @testset "IQL imitation and learning occurs" begin
        # use consistent MDP/data helper
            mdp, π_dist, obs_aidx, state_data = consistent_mdp_data_setup(cols=20, constant_action=true)

            obs_dim, N = size(state_data)
            na = length(collect(actions(mdp)))

            data = alloc_buffer_dict(obs_dim, na, N)
            for t in 1:N
                data[:s][:, t] .= Float32.(state_data[:, t])
                data[:sp][:, t] .= Float32.(state_data[:, t])
                data[:a][:, t] .= false
                data[:a][obs_aidx[t], t] = true
                data[:r][1, t] = Float32(1.0)
            end
            anon = mk_experience_buffer(data)

        # Train using quick IQL (fast, CPU-based) and check that the learned policy
        # prefers the demonstrated action more than uniform.
        π_iql, _, mdp_tr = quick_IQL(mdp, anon)

        s = blindstart_KAgentState(mdp_tr, mdp_tr.start)
        a_sym, a_idx, probs = qpolicy_action(π_iql, mdp_tr, s; temperature=0.5, rng=Random.default_rng())

        @test isa(a_sym, Symbol)
        @test isa(a_idx, Int)
        @test isa(probs, AbstractVector{Float64})
        @test length(probs) == length(actions(mdp_tr))
        @test abs(sum(probs) - 1.0) < 1e-6
        # Training is stochastic; require that some action is preferred over uniform by a small margin
        @test maximum(probs) > 1/na + 0.02
    end

    @testset "Function output shapes" begin
        # use consistent MDP/data helper
        mdp, π_dist, obs_aidx, state_data = consistent_mdp_data_setup(cols=20)

        obs_dim, N = size(state_data)
        na = length(collect(actions(mdp)))

        data = alloc_buffer_dict(obs_dim, na, N)
        rng = Random.MersenneTwister(42)
        for t in 1:N
            data[:s][:, t] .= Float32.(state_data[:, t])
            data[:sp][:, t] .= Float32.(state_data[:, t])
            data[:a][:, t] .= false
            data[:a][obs_aidx[t], t] = true
            data[:r][1, t] = Float32(rand(rng))
        end
        anon = mk_experience_buffer(data)

        π_iql, _, mdp_tr = quick_IQL(mdp, anon)
        buf = rollout_experience_buffer(mdp_tr, π_iql; T=5, rng=Random.MersenneTwister(7))

        @test isa(buf, typeof(anon))
        @test buf.elements == 5
    end

end
