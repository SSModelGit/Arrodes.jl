using POMDPs, POMDPTools, MCTS
using BSON

using MuKumari

@testset "Utilities Usage" begin

    @testset "OneHot conversion" begin
        A = [1.0 0.0 0.0;
             0.0 1.0 0.0;
             0.0 0.0 1.0]
        aidx = onehot_cols_to_aidx(A)
        @test aidx == [1, 2, 3]
    end

    @testset "MDP builder utilities" begin
        menv3 = build_shared_menv()
        @test isa(menv3, MuEnv)

        agent_params = Dict(:start => [1.0 1.0], :dimensions => (0.0, 10.0), :menv => menv3, :obcs => Any[])
        mdp = build_kagent_pomdp(agent_params, x->0.0; name="test_mdp")
        @test isa(mdp, KAgentPOMDP)
    end

    @testset "MuKumari KWorld -> load_runpacks integration" begin
        # Build a minimal MuEnv
        μfs = [(:sin, x->sin(x[1]) + cos(x[2])),
               (:exp, x->exp(-norm(x.-[8 8.])^2)),
               (:lin, x->x[1]^2 + x[2])]
        μs = Symbol[μf[1] for μf in μfs]
        menv2 = MuEnv(length(μs), μs, Dict(μfs))

        # Global objective landscape with one aerial goal (feature :aer)
        goals = [(:aer, Dict(:target=>[5.0 5.0], :strength=>100., :influence=>5.0, :size=>0.5))]
        obcs = []
        globj = GlobalObjectiveLandscape(; goals=goals, obstacles=obcs, horizons=[])

        # lightweight solver for tests
        solver = MCTSSolver(n_iterations=2, depth=2, exploration_constant=0.5)
        dims = (0.0, 10.0)
        kworld = create_kworld(; solver=solver, dims=dims, gobj=globj, menv=menv2)

        # Add a single agent named "ag1_1" so dataset loader can find mdp by agent_inst name
        ag_params = Dict(:name => "ag1_1",
                         :start => [3.0 3.0],
                         :flist => [:aer],
                         :elist => μs)
        add_agent_to_world(kworld, ag_params)

        # Create simple buffers matching expected format
        data = alloc_buffer_dict(27, 4, 3)
        full_buf = mk_experience_buffer(data)
        anon_buf = mk_experience_buffer(deepcopy(data))

        # Assemble a run payload matching schema used in `load_runpacks`
        run = Dict("kworld" => kworld, "ag1" => Dict(:ind_exps => [(full_buf, anon_buf)]))

        tmpb = tempname() * ".bson"
        BSON.@save tmpb runs=[run]

        meta = Dict(
            "data_type" => "multi_run",
            "data_path" => tmpb,
            "loader" => Dict("run_container_key" => "runs", "run_index_key" => :ind_exps),
            "state" => Dict(
                "state_field_sizes" => [2, 2, 12, 10, 1],
                "keep_state_fields" => Bool[1, 1, 1, 0, 1]
            )
        )

        packs = load_runpacks(meta)
        @test length(packs) == 1
        p = packs[1]
        @test p.run_id == 1
        @test p.agent == "ag1"
        @test p.inst == 1
        @test isa(p.mdp, KAgentPOMDP)
        @test isa(p.full, typeof(full_buf))
    end

end