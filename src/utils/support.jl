nanmean(v) = isempty(v) ? NaN : mean(v)

"""
    randcat(rng, p)

Sample an index in 1:length(p) with probabilities p (assumed nonnegative, not necessarily normalized).
"""
function randcat(rng::AbstractRNG, p::AbstractVector{<:Real})
    s = 0.0
    @inbounds for i in eachindex(p)
        s += float(p[i])
    end
    u = rand(rng) * s
    c = 0.0
    @inbounds for i in eachindex(p)
        c += float(p[i])
        if u <= c
            return Int(i)
        end
    end
    return Int(lastindex(p))  # numerical fallback
end

replace_nan_with_zero(v::Vector) = [isnan(Float64(x)) ? 0.0 : Float64(x) for x in v]

#####################
### Objective Helpers
#####################

"""
    _dims_to_bounds(dimensions) -> (lo, hi)

Standardizes boxworld construction from 2-tuple `dimensions` with corners (d1,d1) and (d2,d2).
"""
@inline function _dims_to_bounds(dimensions)
    lo, hi = dimensions[1], dimensions[2]
    lo <= hi || error("dimensions must satisfy dimensions[1] <= dimensions[2]; got $(dimensions)")
    return lo, hi
end

"""
    _grid_from_mdp(mdp; gridsize=100) -> (xs, ys)

Grid over (x,y) spanning mdp.dimensions.
"""
function _grid_from_mdp(mdp::KAgentPOMDP; gridsize::Int=100)
    lo, hi = _dims_to_bounds(mdp.dimensions)
    xs = range(lo, hi; length=gridsize)
    ys = range(lo, hi; length=gridsize)
    return xs, ys
end

"""
    xy_path_from_state_matrix(S; xy_rows=(1,2)) -> (xs, ys)

Extracts the trajectory from `data::ExperienceBufer`

- S is (n_features × T)
- returns vectors length T
"""
function xy_path_from_state_matrix(S::AbstractMatrix; xy_rows::Tuple{Int,Int}=(1,2))
    rx, ry = xy_rows
    T = size(S, 2)
    xs = Vector{Float64}(undef, T)
    ys = Vector{Float64}(undef, T)
    @inbounds for t in 1:T
        xs[t] = Float64(S[rx, t])
        ys[t] = Float64(S[ry, t])
    end
    return xs, ys
end

##########################
# Minimal annotations API
##########################

# robust target extraction: goals look like (:aer, Dict(:target=>[x,y], ...))
function _goal_targets(goals)
    ts = Vector{Vector{Float64}}()
    for g in goals
        d = g[2]
        if d isa Dict && haskey(d, :target)
            push!(ts, vec(Float64.(d[:target])))
        end
    end
    return ts
end

function _max_pairwise_dist(X::Vector{Vector{Float64}})
    n = length(X)
    n ≤ 1 && return 0.0
    best = 0.0
    @inbounds for i in 1:n, j in (i+1):n
        best = max(best, norm(X[i] .- X[j]))
    end
    return best
end

function kworld_annotations(kworld)
    gl = getproperty(kworld, :glob_landscape)
    goals = getproperty(gl, :goals)
    obcs  = getproperty(gl, :obstacles)
    return (
        num_goals = length(goals),
        num_obstacles = length(obcs),
        max_goal_separation = _max_pairwise_dist(_goal_targets(goals))
    )
end
