export particle_filter

function particle_filter(observations::Vector{Int}, π_dist::ScoreΠDist, agent_params::Dict, state_data::Matrix,
                         n_particles::Int = 50; ess_thresh::Float64 = 0.5,
                         rejuv_modes::Int = 8, rejuv_recent_actions::Int = 3,
                         resample_alg::Symbol = :residual,
                         refine_every::Int = 5,
                         refine_topk::Int = 5)

    N = length(observations)
    obs_choices = [choicemap((n => :aidx, observations[n])) for n in 1:N]

    state = pf_initialize(inference_model, (1, π_dist, agent_params, state_data), obs_choices[1], n_particles)

    for n in 2:N
        if effective_sample_size(state) < ess_thresh * n_particles
            pf_resample!(state, resample_alg)

            # rejuvenation selection
            sels = Any[:fourier => :K]
            M = min(rejuv_modes, π_dist.fourier_cfg.Kmax)

            for m in 1:M
                push!(sels, (:fourier, :mode, m) => :fx_idx)
                push!(sels, (:fourier, :mode, m) => :fy_idx)
                push!(sels, (:fourier, :mode, m) => :A_idx)
                push!(sels, (:fourier, :mode, m) => :ϕ_idx)
            end

            a_lo = max(1, n - rejuv_recent_actions)
            for τ in a_lo:(n-1)
                push!(sels, (τ => :aidx))
            end

            pf_rejuvenate!(state, mh, (select(sels...),))

            # after a resample/rejuv event is a great time to refine top policies
            maybe_refine_policies!(π_dist, state, agent_params; topk=refine_topk)
        end

        # Update with new observation
        pf_update!(state,
                   (n, π_dist, agent_params, state_data),
                   (UnknownChange(), NoChange(), NoChange(), NoChange()),
                   obs_choices[n])

        # periodic refinement (lightweight)
        if (n % refine_every) == 0
            maybe_refine_policies!(π_dist, state, agent_params; topk=refine_topk)
        end
    end

    return state
end