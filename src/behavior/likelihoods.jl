@with_kw struct BoltzmannScoreLikelihood
    temperature::Float64 = 1.0
end

@with_kw struct EpsilonGreedyLikelihood
    epsilon::Float64 = 0.05
end

@with_kw struct MovementNoiseLikelihood
    n_transition_samples::Int = 64
    bandwidth::Float64 = 0.25
    action_epsilon::Float64 = 0.02
end

function action_distribution(
    likelihood::BoltzmannScoreLikelihood,
    solver,
    artifact,
    mdp,
    state,
    actions,
    context,
)
    scores = action_scores(solver, artifact, mdp, state, context)
    logits = (scores .- maximum(scores)) ./ likelihood.temperature
    probabilities = exp.(logits)
    probabilities ./ sum(probabilities)
end

function action_distribution(
    likelihood::EpsilonGreedyLikelihood,
    solver,
    artifact,
    mdp,
    state,
    actions,
    context,
)
    selected = planned_action(solver, artifact, mdp, state, context)
    index = findfirst(isequal(selected), actions)
    probabilities = fill(likelihood.epsilon / length(actions), length(actions))
    probabilities[index] += 1 - likelihood.epsilon
    probabilities
end

function observation_loglikelihood(
    likelihood,
    solver,
    artifact,
    mdp,
    state,
    observed_action,
    actions,
    context,
)
    probabilities = action_distribution(
        likelihood,
        solver,
        artifact,
        mdp,
        state,
        actions,
        context,
    )
    index = findfirst(isequal(observed_action), actions)
    isnothing(index) ? -Inf : log(probabilities[index])
end

state_vector(state::MuKumari.KAgentState) = Float64.(vec(state.x))
state_vector(state::AbstractVector) = Float64.(state)

function observation_loglikelihood(
    likelihood::MovementNoiseLikelihood,
    solver,
    artifact,
    mdp,
    state,
    observed_action,
    actions,
    context,
)
    action_term = observation_loglikelihood(
        EpsilonGreedyLikelihood(epsilon=likelihood.action_epsilon),
        solver,
        artifact,
        mdp,
        state,
        observed_action,
        actions,
        context,
    )
    length(context[:states]) < 2 && return action_term
    previous_state = context[:states][end - 1]
    previous_context = copy(context)
    previous_context[:timestep] = max(context[:timestep] - 1, 1)
    previous_context[:states] = context[:states][1:end-1]
    nominal_action = planned_action(
        solver,
        artifact,
        mdp,
        previous_state,
        previous_context,
    )
    kernel_sum = 0.0
    for _ in 1:likelihood.n_transition_samples
        transition = POMDPs.gen(mdp, previous_state, nominal_action, context[:rng])
        distance = norm(state_vector(transition.sp) - state_vector(state))
        kernel_sum += exp(-0.5 * (distance / likelihood.bandwidth)^2)
    end
    action_term + log(max(
        kernel_sum / likelihood.n_transition_samples,
        eps(Float64),
    ))
end
