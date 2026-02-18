################
### Plotting ###
################

"""
    plot_top_objective_with_trajectories(pf_state, π_dist, agent_params;
                                         observed_state_matrix,
                                         gridsize=140,
                                         xy_rows=(1,2),
                                         show_predicted=true,
                                         title_prefix="Top objective")

Heatmap of the inferred top objective function, overlaying:
- observed agent trajectory (from data)
- predicted rollout under the inferred objective's MDP+policy (greedy)

Returns a Plots.jl plot object.
"""
function plot_top_objective_with_trajectories(pf_state, π_dist::ScoreΠDist, agent_params::Dict;
                                             observed_state_matrix::AbstractMatrix,
                                             gridsize::Int=140,
                                             xy_rows::Tuple{Int,Int}=(1,2),
                                             show_predicted::Bool=true,
                                             title_prefix::String="Top objective")

    key, prob = top_key(pf_state, π_dist)

    # Build inferred scalar field from decoded params
    ff = decode_fourier_key(key, π_dist.fourier_cfg)
    field = make_fourier_scalar_field(ff; scaleQ=true)

    # Need an mdp for plotting bounds (use cached/ensured proposal mdp)
    mdp_hat = ensure_mdp!(π_dist, key)
    xs_grid, ys_grid = _grid_from_mdp(mdp_hat; gridsize=gridsize)

    Z = objective_grid_from_field(field, xs_grid, ys_grid)

    # Observed trajectory
    obs_x, obs_y = xy_path_from_state_matrix(observed_state_matrix; xy_rows=xy_rows)
    T = length(obs_x)

    p = heatmap(xs_grid, ys_grid, Z;
               aspect_ratio=1,
               title="$(title_prefix) (posterior ≈ $(prob))",
               xlabel="x", ylabel="y",
               colorbar_title="objective")

    plot!(p, obs_x, obs_y; label="observed", linewidth=3)

    if show_predicted
        start_state = agent_params[:start_state]
        pred_x, pred_y, _ = rollout_greedy_policy(π_dist, key; start_state=start_state, T=T)
        plot!(p, pred_x, pred_y; label="predicted (greedy)", linewidth=3, linestyle=:dash)
    end

    # Mark start/end for quick visual sanity
    scatter!(p, [obs_x[1]], [obs_y[1]]; label="obs start", markersize=6)
    scatter!(p, [obs_x[end]], [obs_y[end]]; label="obs end", markersize=6)

    return p
end

"""
    plot_objective_side_by_side(pf_state, π_dist;
                                observed_mdp,
                                gridsize=140,
                                title_left="Inferred top objective",
                                title_right="Observed MDP objective")

Side-by-side heatmaps:
- inferred top objective field (from Fourier features)
- observed MDP objective (mdp.obj(s)[1])

Returns a Plots.jl plot object with layout (1,2).
"""
function plot_objective_side_by_side(pf_state, π_dist::ScoreΠDist;
                                    observed_mdp::KAgentPOMDP,
                                    gridsize::Int=140,
                                    title_left::String="Inferred top objective",
                                    title_right::String="Observed MDP objective")

    key, prob = top_key(pf_state, π_dist)

    ff = decode_fourier_key(key, π_dist.fourier_cfg)
    field = make_fourier_scalar_field(ff; scaleQ=true)

    # Use observed mdp bounds for both to make comparison apples-to-apples
    xs_grid, ys_grid = _grid_from_mdp(observed_mdp; gridsize=gridsize)

    Z_inf = objective_grid_from_field(field, xs_grid, ys_grid)
    Z_obs = objective_grid_from_mdp(observed_mdp, xs_grid, ys_grid)

    p1 = heatmap(xs_grid, ys_grid, Z_inf;
                 aspect_ratio=1,
                 title="$(title_left)\n(posterior ≈ $(prob))",
                 xlabel="x", ylabel="y",
                 colorbar_title="objective")

    p2 = heatmap(xs_grid, ys_grid, Z_obs;
                 aspect_ratio=1,
                 title=title_right,
                 xlabel="x", ylabel="y",
                 colorbar_title="objective")

    return plot(p1, p2; layout=(1,2))
end

##################################
# Make Inference Debugging Figures
##################################

"""
Plot 1:
- Heatmap of true objective
- Overlay: observed trajectory from cache full_data[:s]
- Overlay: rollout under IQ-SIPS top inferred objective (greedy) from same start, same horizon

Returns a Plots.jl plot.
"""
function plot_true_objective_vs_iqsips_rollout(cache::Dict, e;
                                               gridsize::Int=180,
                                               xy_rows::Tuple{Int,Int}=(1,2))
    rec = cache_record_for_eval(cache, e)
    muenv_spec = cache[:muenv_spec]

    mdp, agent_params = reconstruct_mdp_from_cache(rec, muenv_spec)

    # Grid + true objective
    xs, ys = _grid_from_mdp(mdp; gridsize=gridsize)
    Z_true = objective_grid_from_mdp(mdp, xs, ys)

    # Observed trajectory from cached data
    Sobs = rec[:full_data][:s]
    obs_x, obs_y = xy_path_from_state_matrix(Sobs; xy_rows=xy_rows)
    T = length(obs_x)

    # IQ-SIPS inferred rollout (requires top_key)
    keyB = e.B.top_key
    probB = get(e.B, :top_prob, NaN)

    # Build π_dist for rollout helper
    as = actions(mdp)
    action_list = [as, a->Flux.onehot(a, as), Flux.onehotbatch(as, as)]
    π_dist = ScoreΠDist(; mdp_params=action_list)

    # -------------------- FIX: ensure MDP exists for this key --------------------
    cfgB = rec[:cfg]  # FourierDiscreteCfg used during ablation
    ffB  = decode_fourier_key(keyB, cfgB)
    ensure_mdp!(π_dist, keyB, ffB, agent_params)   # populates n_propmdp_list[keyB]
    # ---------------------------------------------------------------------------

    pred_x, pred_y, _ = rollout_greedy_policy(π_dist, keyB; start_state=agent_params[:start_state], T=10)

    p = heatmap(xs, ys, Z_true;
        aspect_ratio=1,
        dpi=220,
        title="True Objective vs IQ-SIPS Inferred Behavior (posterior ≈ $(isnan(probB) ? "?" : string(round(probB, digits=3))))",
        xlabel="x (world units)",
        ylabel="y (world units)",
        colorbar_title="Objective value")

    plot!(p, obs_x, obs_y; label="Observed trajectory", linewidth=3)
    plot!(p, pred_x, pred_y; label="IQ-SIPS rollout (greedy, top key)", linewidth=3, linestyle=:dash)

    scatter!(p, [obs_x[1]], [obs_y[1]]; label="Start", markersize=6)
    scatter!(p, [obs_x[end]], [obs_y[end]]; label="End", markersize=6)

    return p
end

"""
Plot 2:
- Heatmap true objective
- Heatmap inferred objective (Open-Ended SIPS top key)
- Heatmap inferred objective (IQ-SIPS top key)

Returns a 1x3 Plots.jl layout plot.
"""
function plot_objective_triptych(cache::Dict, e;
                                 gridsize::Int=180)
    rec = cache_record_for_eval(cache, e)
    muenv_spec = cache[:muenv_spec]

    mdp, _ = reconstruct_mdp_from_cache(rec, muenv_spec)

    xs, ys = _grid_from_mdp(mdp; gridsize=gridsize)

    # True objective from reconstructed mdp
    Z_true = objective_grid_from_mdp(mdp, xs, ys)

    # Inferred objectives from stored keys
    cfg = rec[:cfg]   # FourierDiscreteCfg used to decode keys
    keyA = e.A.top_key
    keyB = e.B.top_key

    probA = get(e.A, :top_prob, NaN)
    probB = get(e.B, :top_prob, NaN)

    Z_A = objective_grid_from_key(keyA, cfg, xs, ys)
    Z_B = objective_grid_from_key(keyB, cfg, xs, ys)

    p_true = heatmap(xs, ys, Z_true;
        aspect_ratio=1, dpi=220,
        title="True Objective",
        xlabel="x (world units)", ylabel="y (world units)",
        colorbar_title="Objective")

    p_A = heatmap(xs, ys, Z_A;
        aspect_ratio=1, dpi=220,
        title="Open-Ended SIPS (posterior ≈ $(isnan(probA) ? "?" : string(round(probA, digits=3))))",
        xlabel="x (world units)", ylabel="y (world units)",
        colorbar_title="Objective")

    p_B = heatmap(xs, ys, Z_B;
        aspect_ratio=1, dpi=220,
        title="IQ-SIPS (posterior ≈ $(isnan(probB) ? "?" : string(round(probB, digits=3))))",
        xlabel="x (world units)", ylabel="y (world units)",
        colorbar_title="Objective")

    return plot(p_true, p_A, p_B; layout=(1,3), size=(1500, 480))
end

"""
    make_final_inference_figures(out; ...)

Produces:
1) Plot 1: best accuracy run with IQ-SIPS non-degenerate
2) Plot 2: best accuracy run with both methods non-degenerate

Returns a NamedTuple with plots and selected evals.
"""
function make_final_inference_figures(out;
                                      gridsize::Int=180,
                                      xy_rows::Tuple{Int,Int}=(1,2))
    cache = out.cache
    evals = out.evals

    # (1) best accuracy with IQ-SIPS non-degenerate
    e1 = best_eval_by_accuracy(evals; requireA=false, requireB=true)
    p1 = plot_true_objective_vs_iqsips_rollout(cache, e1; gridsize=gridsize, xy_rows=xy_rows)

    # (2) best accuracy with both non-degenerate
    e2 = best_eval_by_accuracy(evals; requireA=true, requireB=true)
    p2 = plot_objective_triptych(cache, e2; gridsize=gridsize)

    return (p1=p1, p2=p2, best_iqsips=e1, best_both=e2)
end

"""
    plot_all_objectives_from_cache(cache; gridsize=140, max_per_page=9, cmap=:viridis, savepath=nothing)

Paginated multi-panel heatmaps of every objective field stored in `cache[:records]`.
If `savepath` is provided (directory), PNG files are saved as `objectives_page_<p>.png` and a vector of filenames is returned.
Otherwise a Plots.jl object for the single page (if one) or last page is returned.
"""
function plot_all_objectives_from_cache(cache::Dict; gridsize::Int=140, max_per_page::Int=9, cmap=:viridis, savepath=nothing)
    records = cache[:records]
    n = length(records)
    if n == 0
        @warn "No records found in cache"
        return nothing
    end

    pages = ceil(Int, n / max_per_page)
    saved_files = String[]
    p_last = nothing

    for page in 1:pages
        idx0 = (page-1)*max_per_page + 1
        idx1 = min(page*max_per_page, n)
        ids = idx0:idx1
        m = length(ids)

        # choose grid layout
        ncol = ceil(Int, sqrt(m))
        nrow = ceil(Int, m / ncol)

        p = plot(layout=(nrow, ncol), size=(ncol*380, nrow*320))

        for (ii, rec_i) in enumerate(ids)
            rec = records[rec_i]
            # reconstruct mdp for bounds
            mdp, _ = reconstruct_mdp_from_cache(rec, cache[:muenv_spec])
            xs, ys = _grid_from_mdp(mdp; gridsize=gridsize)

            # compute objective grid using stored key+cfg helper
            Z = objective_grid_from_key(rec[:key], rec[:cfg], xs, ys)

            tit = "id=$(rec[:id]) $(rec[:sweep])=$(rec[:level])"
            heatmap!(p, xs, ys, Z; subplot=ii, aspect_ratio=1, title=tit, xlabel="x", ylabel="y", colorbar_title="obj", c=cmap)
        end

        if savepath !== nothing
            isdir(savepath) || mkpath(savepath)
            fname = joinpath(savepath, "objectives_page_$(page).png")
            savefig(p, fname)
            push!(saved_files, fname)
        end
        p_last = p
    end

    if savepath !== nothing
        return saved_files
    end
    return p_last
end


"""
    compare_iql_vs_true_policy(cache, e; gridsize=80, solver_type=:mcts, solver_params=[:dpw, 1000, 10.0], show_quiver=true, savepath=nothing)

For the cache record referenced by eval `e`, trains a quick IQL policy on the anonymized buffer stored in the cache and computes, across a grid of states, the action
action chosen by:
 - a planning solver (MCTS/DPW) that we treat as the "true"/expert policy
 - the quick_IQL learned policy

Produces a 1x3 figure: (true action quiver, IQL action quiver, agreement heatmap). If `savepath` is provided the PNG is written and the filename returned; otherwise the Plots.jl object is returned.
"""
function compare_iql_vs_true_policy(cache::Dict, e; gridsize::Int=80, solver_type::Symbol=:mcts, solver_params=[:dpw, 1000, 10.0], show_quiver::Bool=true, savepath=nothing)
    rec = cache_record_for_eval(cache, e)
    muenv_spec = cache[:muenv_spec]

    mdp, agent_params = reconstruct_mdp_from_cache(rec, muenv_spec)

    # Recreate anon buffer and train quick IQL
    anon_data = Dict{Symbol, Matrix}(rec[:anon_data])
    anon_buf  = ExperienceBuffer(anon_data, size(anon_data[:s],2), 1, Array{Int64}[], nothing, 0)

    # Train quick IQL (may be fast) on this mdp + anon buffer
    π_iql, _, _ = quick_IQL(mdp, anon_buf)

    # Build planner (true policy)
    𝒮 = solver_from_type(mdp, solver_type; solver_params=solver_params)
    # solver_from_type may return an array for :all; pick first if so
    if isa(𝒮, AbstractArray)
        𝒮 = 𝒮[1]
    end
    π_true = solve(𝒮, mdp)

    # helpers
    as = actions(mdp)
    na = length(as)

    # function mapping action symbol -> unit vector for quiver; fallback to angular mapping
    function action_to_unit_vector(asymb)
        s = Symbol(asymb)
        if s in (:up, :north)
            return (0.0, 1.0)
        elseif s in (:down, :south)
            return (0.0, -1.0)
        elseif s in (:left, :west)
            return (-1.0, 0.0)
        elseif s in (:right, :east)
            return (1.0, 0.0)
        elseif s in (:up_right, :ne, :northeast)
            v = 1/sqrt(2); return (v, v)
        elseif s in (:up_left, :nw, :northwest)
            v = 1/sqrt(2); return (-v, v)
        elseif s in (:down_right, :se, :southeast)
            v = 1/sqrt(2); return (v, -v)
        elseif s in (:down_left, :sw, :southwest)
            v = 1/sqrt(2); return (-v, -v)
        else
            # fallback: map index -> angle
            idx = findfirst(x->x==asymb, as)
            if idx === nothing
                return (0.0, 0.0)
            end
            ang = 2π*(idx-1)/max(na,1)
            return (cos(ang), sin(ang))
        end
    end

    # Grid
    xs, ys = _grid_from_mdp(mdp; gridsize=gridsize)
    nx = length(xs); ny = length(ys)

    U_true = zeros(Float64, ny, nx)
    V_true = zeros(Float64, ny, nx)
    U_iql  = zeros(Float64, ny, nx)
    V_iql  = zeros(Float64, ny, nx)
    Z_agree = zeros(Float64, ny, nx)

    @inbounds for (j, y) in enumerate(ys)
        for (i, x) in enumerate(xs)
            s = _state_at_xy(mdp, x, y)
            # true action (planner)
            try
                a_true = action(π_true, s)[1]
            catch err
                # fallback to evaluating Q-values
                as_list = actions(mdp)
                all_a_onehot = Flux.onehotbatch(as_list, as_list)
                qv = vec(Crux.value(π_true, MuKumari.shape_state_as_obs(mdp, s), all_a_onehot))
                aidx = argmax(qv)
                a_true = as_list[aidx]
            end

            # iql action
            try
                a_iql = action(π_iql, s)[1]
            catch err
                as_list = actions(mdp)
                all_a_onehot = Flux.onehotbatch(as_list, as_list)
                qv = vec(Crux.value(π_iql, MuKumari.shape_state_as_obs(mdp, s), all_a_onehot))
                aidx = argmax(qv)
                a_iql = as_list[aidx]
            end

            ux_t, vy_t = action_to_unit_vector(a_true)
            ux_i, vy_i = action_to_unit_vector(a_iql)

            U_true[j,i] = ux_t
            V_true[j,i] = vy_t
            U_iql[j,i]  = ux_i
            V_iql[j,i]  = vy_i

            Z_agree[j,i] = (a_true == a_iql) ? 1.0 : 0.0
        end
    end

    # Build plots
    p_true = show_quiver ? quiver(xs, ys, quiver=(U_true, V_true); aspect_ratio=1, title="Planner (true) policy", xlabel="x", ylabel="y") : heatmap(xs, ys, Z_agree; aspect_ratio=1, title="Planner actions (overlay suppressed)")
    p_iql  = show_quiver ? quiver(xs, ys, quiver=(U_iql, V_iql); aspect_ratio=1, title="IQL learned policy", xlabel="x", ylabel="y") : heatmap(xs, ys, Z_agree; aspect_ratio=1, title="IQL actions (overlay suppressed)")
    p_agree = heatmap(xs, ys, Z_agree; aspect_ratio=1, title="Action agreement (1=agree)", colorbar_title="agree")

    p_all = plot(p_true, p_iql, p_agree; layout=(1,3), size=(1400,480))

    if savepath !== nothing
        isdir(savepath) || mkpath(savepath)
        fname = joinpath(savepath, "iql_vs_true_id_$(rec[:id])_sweep_$(rec[:sweep])_level_$(rec[:level]).png")
        savefig(p_all, fname)
        return fname
    end

    return p_all
end