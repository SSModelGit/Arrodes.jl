"""
    make_pomdp_objective_from_field(field; done_mode=:never, done_threshold=Inf)

Converts a scalar field into MuKumari's objective signature:
  obj(s)::Any = Any[reward::Real, done::Bool]

Note that it defaults to just saying `false`, as there is no clear `done` in the open-ended case.
"""
function make_pomdp_objective_from_field(field::Function)
    return (s) -> Any[field(s.x[1,1], s.x[1,2]), false]
end

"""
    objective_grid_from_field(field, xs, ys) -> Matrix

Returns Z where Z[j,i] = field(xs[i], ys[j]) (i = x index, j = y index),
which matches Plots.heatmap(x, y, Z) conventions.
"""
function objective_grid_from_field(field::Function, xs, ys)
    Z = Matrix{Float64}(undef, length(ys), length(xs))
    @inbounds for (j, y) in enumerate(ys)
        for (i, x) in enumerate(xs)
            Z[j,i] = field(x, y)
        end
    end
    return Z
end

"""
    _state_at_xy(mdp, x, y) -> KAgentState

Constructs a state located at (x,y) using MuKumari's blindstart helper.
"""
@inline function _state_at_xy(mdp::KAgentPOMDP, x::Real, y::Real)
    return blindstart_KAgentState(mdp, reshape([Float64(x), Float64(y)], (1,2)))
end

"""
    objective_grid_from_mdp(mdp, xs, ys) -> Matrix

Uses mdp.obj(s)[1] as the scalar objective/reward at (x,y).
"""
function objective_grid_from_mdp(mdp::KAgentPOMDP, xs, ys)
    Z = Matrix{Float64}(undef, length(ys), length(xs))
    @inbounds for (j, y) in enumerate(ys)
        for (i, x) in enumerate(xs)
            s = _state_at_xy(mdp, x, y)
            r = mdp.obj(s)[1]
            Z[j,i] = Float64(r)
        end
    end
    return Z
end