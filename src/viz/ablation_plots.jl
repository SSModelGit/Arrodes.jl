function default_deg_height(yA_plot::Vector{<:Real}, yB_plot::Vector{<:Real}; ylims=nothing)
    # Prefer ylims if provided (best for ACC/ESS)
    if ylims !== nothing
        ymin, ymax = ylims
        yr = max(ymax - ymin, eps(Float64))
        return 0.03 * yr
    end

    # Otherwise infer from data scale (RMSE often)
    ys = vcat(yA_plot, yB_plot)
    ymax = maximum(ys)
    if !isfinite(ymax) || ymax ≤ 0
        return 0.05
    end
    return max(0.03 * ymax, 1e-6)
end

function hatch_rect!(p, x_left::Real, x_right::Real, y0::Real, y1::Real;
                     spacing_frac::Real=0.18, linecolor=:black, linewidth::Real=1.5, direction::Symbol=:/)
    w = x_right - x_left
    h = y1 - y0
    if w ≤ 0 || h ≤ 0
        return p
    end

    spacing = spacing_frac * w
    # We draw a family of parallel lines that intersect the rectangle.
    # direction = :/ means rising left->right, :\ means falling left->right.
    if direction == :/
        # Lines: y = (h/w)*(x - c) + y0; sweep c
        cmin = x_left - h * (w/h)  # safe over-sweep
        cmax = x_right
        cs = collect(cmin:spacing:cmax)
        for c in cs
            # segment endpoints clipped to rectangle
            # compute intersection with bottom/top edges
            x0 = c
            y_at_xleft  = y0 + (h/w) * (x_left - c)
            y_at_xright = y0 + (h/w) * (x_right - c)

            # candidate points on left/right edges
            pts = Tuple{Float64,Float64}[]
            if y0 ≤ y_at_xleft ≤ y1
                push!(pts, (x_left, y_at_xleft))
            end
            if y0 ≤ y_at_xright ≤ y1
                push!(pts, (x_right, y_at_xright))
            end
            # intersections with bottom/top edges
            x_at_y0 = c
            x_at_y1 = c + (w/h)*h  # c + w
            # Actually for this parameterization, easier: solve for x given y:
            # y = y0 + (h/w)(x - c) => x = c + (w/h)(y - y0)
            x_bot = c + (w/h)*(0.0)
            x_top = c + (w/h)*(h)
            if x_left ≤ x_bot ≤ x_right
                push!(pts, (x_bot, y0))
            end
            if x_left ≤ x_top ≤ x_right
                push!(pts, (x_top, y1))
            end

            if length(pts) ≥ 2
                # pick two farthest points (simple: first two after unique)
                (xA,yA),(xB,yB) = pts[1], pts[2]
                plot!(p, [xA,xB], [yA,yB]; color=linecolor, linewidth=linewidth, label=nothing)
            end
        end
    else
        # direction == :\ : mirror by swapping left/right in the slope sign
        # Use same approach but slope negative.
        spacing = spacing_frac * w
        cs = collect((x_left):spacing:(x_right + h*(w/h)))
        for c in cs
            # line: y = y0 + (h/w)*(c - x)
            y_at_xleft  = y0 + (h/w) * (c - x_left)
            y_at_xright = y0 + (h/w) * (c - x_right)

            pts = Tuple{Float64,Float64}[]
            if y0 ≤ y_at_xleft ≤ y1
                push!(pts, (x_left, y_at_xleft))
            end
            if y0 ≤ y_at_xright ≤ y1
                push!(pts, (x_right, y_at_xright))
            end

            # Solve for x on bottom/top: y = y0 + (h/w)*(c - x) => x = c - (w/h)(y - y0)
            x_bot = c - (w/h)*(0.0)
            x_top = c - (w/h)*(h)
            if x_left ≤ x_bot ≤ x_right
                push!(pts, (x_bot, y0))
            end
            if x_left ≤ x_top ≤ x_right
                push!(pts, (x_top, y1))
            end

            if length(pts) ≥ 2
                (xA,yA),(xB,yB) = pts[1], pts[2]
                plot!(p, [xA,xB], [yA,yB]; color=linecolor, linewidth=linewidth, label=nothing)
            end
        end
    end

    return p
end

# Convert sweep levels (stored as bin max indices) to interpretable labels in physical units.
# Uses the cfg stored in cache per-record (best, because it reflects the actual sweep).
function sweep_tick_labels_from_cache(cache::Dict, sw::Symbol, levels::Vector{Int})
    recs = cache[:records]

    # Helper: find cfg for a (sweep, level)
    function cfg_for(sw, lv)
        for r in recs
            if r[:sweep] == sw && r[:level] == lv
                return r[:cfg]
            end
        end
        error("No cache record found for sweep=$(sw), level=$(lv)")
    end

    labels = String[]
    for lv in levels
        cfg = cfg_for(sw, lv)
        if sw == :K
            push!(labels, string(lv))  # K itself is meaningful
        elseif sw == :freq_range
            # lv is Fmax_i; Δf is physical step
            halfspan = lv * cfg.Δf
            width = 2 * halfspan
            push!(labels, @sprintf("%.2f", width))  # show width, not index
            # alternatively: push!(labels, "±$(round(halfspan,digits=2))")
        elseif sw == :amp_range
            # lv is Amax_i; ΔA is physical step
            amax = lv * cfg.ΔA
            push!(labels, @sprintf("%.2f", amax))
        else
            push!(labels, string(lv))
        end
    end
    return labels
end

function pretty_xlabel(sw::Symbol)
    sw == :K && return "K (number of Fourier features)"
    sw == :freq_range && return "Frequency range width, 2Fₘₐₓ (units)"
    sw == :amp_range && return "Amplitude maximum, Aₘₐₓ (units)"
    return "Sweep level"
end

function pretty_title(sw::Symbol, metric::Symbol)
    sweep_name = sw == :K ? "K Sweep" :
                 sw == :freq_range ? "Frequency Range Sweep" :
                 sw == :amp_range ? "Amplitude Range Sweep" : string(sw)
    metric_name = metric == :ess ? "Effective Sample Size (ESS)" :
                  metric == :rmse ? "Objective Reconstruction Error (RMSE)" :
                  metric == :acc ? "Policy Match Accuracy" : string(metric)
    return "$(sweep_name): $(metric_name)"
end

function pretty_ylabel(metric::Symbol)
    metric == :ess && return "ESS (particles)"
    metric == :rmse && return "RMSE (objective value)"
    metric == :acc && return "Accuracy (fraction)"
    return string(metric)
end

function grouped_bars(level_labels::Vector{String}, yA::Vector, yB::Vector;
                      title::String, xlabel::String, ylabel::String,
                      ylims=nothing)

    n = length(level_labels)
    @assert length(yA) == n && length(yB) == n

    # IMPORTANT: numeric x; labels supplied via xticks
    x = 1:n

    # Y must be n×2 where each column is a method (A, B)
    Y = hcat(yA, yB)

    p = groupedbar(
        x, Y;
        bar_position = :dodge,      # side-by-side
        label = METHOD_LABELS,
        xticks = (x, level_labels),
        xrotation = 25,
        title = title,
        xlabel = xlabel,
        ylabel = ylabel,
        size = (950, 560),
        dpi = 220,
        framestyle = :box,
        gridalpha = 0.15,
        legend = :topright,
        left_margin = 12mm, right_margin = 6mm,
        top_margin = 10mm, bottom_margin = 12mm
    )

    if ylims !== nothing
        ylims!(p, ylims)
    end

    return p
end

function grouped_bars_with_degenerate_overlay(
    level_labels::Vector{String},
    yA::Vector, yB::Vector,
    degA::AbstractVector{Bool}, degB::AbstractVector{Bool};
    title::String, xlabel::String, ylabel::String,
    ylims=nothing,
    deg_height::Union{Nothing,Float64}=nothing
)
    n = length(level_labels)
    @assert length(yA)==n && length(yB)==n
    @assert length(degA)==n && length(degB)==n

    x = 1:n
    yA_plot = replace_nan_with_zero(yA)
    yB_plot = replace_nan_with_zero(yB)
    Y = hcat(yA_plot, yB_plot)

    p = groupedbar(
        x, Y;
        bar_position=:dodge,
        label=METHOD_LABELS,
        xticks=(x, level_labels),
        xrotation=25,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        size=(950, 560),
        dpi=220,
        framestyle=:box,
        gridalpha=0.15,
        legend=:topright,
        left_margin=12mm, right_margin=6mm,
        top_margin=10mm, bottom_margin=12mm
    )

    if ylims !== nothing
        ylims!(p, ylims)
    end

    # Choose a small visible marker height
    h = deg_height === nothing ? default_deg_height(yA_plot, yB_plot; ylims=ylims) : deg_height

    # Approximate dodge geometry for 2-series groupedbar:
    dx = 0.18
    bw = 0.32

    # Draw small bars + hatch lines
    for i in 1:n
        if degA[i]
            xc = x[i] - dx
            # draw outline bar
            bar!(p, [xc], [h]; bar_width=bw, fillalpha=0.0, linecolor=:black, linewidth=2, label=nothing)
            # hatch over the rectangle
            hatch_rect!(p, xc - bw/2, xc + bw/2, 0.0, h; direction=:/, linewidth=1.2)
        end
        if degB[i]
            xc = x[i] + dx
            bar!(p, [xc], [h]; bar_width=bw, fillalpha=0.0, linecolor=:black, linewidth=2, label=nothing)
            hatch_rect!(p, xc - bw/2, xc + bw/2, 0.0, h; direction=:\, linewidth=1.2)
        end
    end

    return p
end

function make_ablation_barplots(out)
    sumdict = out.summaries
    cache   = out.cache
    plots   = Dict{Symbol,Dict{Symbol,Any}}()

    for (sw, S) in sumdict
        levels = S.levels
        tick_labels = sweep_tick_labels_from_cache(cache, sw, levels)

        # Degeneracy masks per metric: use collapsed flags + NaNs in that metric
        degA_ess,  degB_ess  = degmask_from_summary(S.essA,  S.essB,  S.collapsedA, S.collapsedB)
        degA_rmse, degB_rmse = degmask_from_summary(S.rmseA, S.rmseB, S.collapsedA, S.collapsedB)
        degA_acc,  degB_acc  = degmask_from_summary(S.accA,  S.accB,  S.collapsedA, S.collapsedB)

        p_ess = grouped_bars_with_degenerate_overlay(
            tick_labels, S.essA, S.essB, degA_ess, degB_ess;
            title=pretty_title(sw, :ess),
            xlabel=pretty_xlabel(sw),
            ylabel=pretty_ylabel(:ess),
            ylims=(0, out.meta[:n_particles])
        )

        p_rmse = grouped_bars_with_degenerate_overlay(
            tick_labels, S.rmseA, S.rmseB, degA_rmse, degB_rmse;
            title=pretty_title(sw, :rmse),
            xlabel=pretty_xlabel(sw),
            ylabel=pretty_ylabel(:rmse)
        )

        p_acc = grouped_bars_with_degenerate_overlay(
            tick_labels, S.accA, S.accB, degA_acc, degB_acc;
            title=pretty_title(sw, :acc),
            xlabel=pretty_xlabel(sw),
            ylabel=pretty_ylabel(:acc),
            ylims=(0, 1)
        )

        plots[sw] = Dict(:ess=>p_ess, :rmse=>p_rmse, :acc=>p_acc)
    end

    return plots
end