# Continuous Component API: MDP creation from objective function
ensure_mdp!(π_dist::ScoreΠDist, key, obj::Function, agent_config::InferenceConfig) =
    get!(π_dist.n_propmdp_list, key) do
        build_kagent_pomdp(agent_config.agent_params, obj; name = "component_" * string(key))
    end

function get_𝒮_proposal(π_dist::ScoreΠDist, key, mdp::Any, config::InferenceConfig)
    return get!(π_dist.n_𝒮_proposals, key) do
        deep_q_solver(
            mdp;
            solver_params = [
                :softq,
                config.rl_config.n_iterations,
                config.rl_config.epochs,
                config.rl_config.batch_size,
            ],
        )
    end
end

# Continuous Component API: returns (solver, policy) tuple
get_π_proposal(π_dist::ScoreΠDist, key, mdp::Any, config::InferenceConfig) =
    get!(π_dist.n_π_proposals, key) do
        solver = get_𝒮_proposal(π_dist, key, mdp, config)
        solve(solver, mdp)
    end
