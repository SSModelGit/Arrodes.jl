function objective_loglikelihood(
    problem::ObjectiveInferenceProblem,
    cache::Dict{Symbol,Any},
    objective_index::Int,
    timestep::Int,
)
    scores = get!(cache, :action_loglikelihoods) do
        Dict{Tuple{Int,Int},Float64}()
    end
    get!(scores, (objective_index, timestep)) do
        hypothesis = problem.hypotheses[objective_index]
        mdp = hypothesis_mdp(problem, cache, objective_index)
        states = [
            problem.state_adapter(mdp, state, time)
            for (time, state) in enumerate(problem.states[1:timestep])
        ]
        context = Dict{Symbol,Any}(
            :hypothesis_id => hypothesis.id,
            :timestep => timestep,
            :states => states,
            :actions => problem.actions[1:max(timestep - 1, 0)],
            :horizon => length(problem.actions),
            :rng => cache[:rng],
            :metadata => hypothesis.metadata,
        )
        artifacts = get!(cache, :artifacts) do
            Dict{Symbol,Any}()
        end
        artifact = get!(artifacts, hypothesis.id) do
            prepare_behavior(hypothesis.behavior.solver, mdp, context)
        end
        action_list = collect(POMDPs.actions(mdp))
        observation_loglikelihood(
            hypothesis.behavior.likelihood,
            hypothesis.behavior.solver,
            artifact,
            mdp,
            states[end],
            problem.actions[timestep],
            action_list,
            context,
        )
    end
end
