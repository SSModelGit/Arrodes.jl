"""
    build_kagent_pomdp(agent_params::Dict, obj::Function; name="fourier_obj")

Required keys in agent_params:
- :start::Matrix
- :dimensions::Tuple   # (d1, d2), same semantics as MuKumari
- :menv::MuEnv

Optional keys (with defaults aligned to init_standard_KAgentPOMDP):
- :digits::Int
- :agent_width::Float64
- :agent_speed::Float64
- :ag_mvt_noise::Float64
- :obs_noise::Float64
- :mdp_discount::Float64
- :obcs::Vector   # optional obstacles geometry (default empty)
- :goals::Vector  # optional goals geometry (default empty)
"""
function build_kagent_pomdp(agent_params::Dict, obj::Function; name::String="fourier_obj")
    @assert haskey(agent_params, :start)
    @assert haskey(agent_params, :dimensions)
    @assert haskey(agent_params, :menv)
    @assert haskey(agent_params, :obcs)

    start      = agent_params[:start]
    d          = agent_params[:dimensions]
    menv       = agent_params[:menv]
    obcs       = agent_params[:obcs]

    digits     = get(agent_params, :digits, 3)
    width      = get(agent_params, :agent_width, 0.1)
    speed      = get(agent_params, :agent_speed, 1.0)
    ag_noise   = get(agent_params, :ag_mvt_noise, 0.05)
    obs_noise  = get(agent_params, :obs_noise, 0.05)
    γ          = get(agent_params, :mdp_discount, 0.95)

    goals      = get(agent_params, :goals, Any[])

    # Minimal agent landscape placeholder (not used to define obj)
    objl = AgentObjectiveLandscape(objectives=Any[], f_types=Any[])

    # Mirror init_standard_KAgentPOMDP world construction
    boxworld = GI.Polygon([[(d[1], d[1]), (d[1], d[2]), (d[2], d[2]), (d[2], d[1]), (d[1], d[1])]])
    # Note: if obcs are empty, world is just the exterior ring.
    world = isempty(obcs) ? boxworld : GI.Polygon([GI.getexterior(boxworld), map(o -> GI.getexterior(o), obcs)...])

    return KAgentPOMDP(name=name, start=start,
                      dimensions=d, boxworld=boxworld,
                      objl=objl, obcs=obcs, goals=goals,
                      obj=obj,
                      world=world,
                      width=width, s=speed, w=ag_noise,
                      menv=menv, v=obs_noise, γ=γ,
                      digits=digits)
end

"""
    build_shared_menv(; M::Int=3)

Builds a standard MuEnv from a specified number of dimensions.

No change to the internal environment to avoid world-age issues.
"""
function build_shared_menv(; M::Int=3)
    μfs = [
        (:sin, x->sin(x[1]) + cos(x[2])),
        (:exp, x->100*exp(-norm(x.-[8 8.])^2 / 1.0)),
        (:lin, x->x[1]^2 + x[2])
    ]
    μs = Symbol[μfs[i][1] for i in 1:M]
    return MuEnv(M, μs, Dict(μfs))
end

"""
    build_shared_menv(spec::MuEnvSpec)

Builds a standard MuEnv from a specified number of dimensions.

Constant internal environment across function calls to avoid world-age issues.
"""
function build_shared_menv(spec::MuEnvSpec)
    μfs = [
        (:sin, x->sin(x[1]) + cos(x[2])),
        (:exp, x->100*exp(-norm(x.-[8 8.])^2 / 1.0)),
        (:lin, x->x[1]^2 + x[2])
    ]
    μdict = Dict(μfs)
    return MuEnv(spec.M, spec.μ_order, μdict)
end

"""
    agent_params_from_mdp(mdp::KAgentPOMDP) -> Dict{Symbol,Any}

Extracts all non-objective agent and environment parameters from an existing
`KAgentPOMDP`, so that new POMDPs can be constructed with identical dynamics,
geometry, noise, discounting, etc., but a different objective function.

The returned dictionary is compatible with `build_kagent_pomdp(agent_params, obj)`.
"""
function agent_params_from_mdp(mdp::KAgentPOMDP)
    return Dict(
        # --- required ---
        :start        => mdp.start,
        :start_state  => KAgentState(mdp.start, [predict_env(menv, mdp.start)], Matrix[]),
        :dimensions   => mdp.dimensions,
        :menv         => menv,

        # --- dynamics / noise ---
        :agent_width  => mdp.width,
        :agent_speed  => mdp.s,
        :ag_mvt_noise => mdp.w,
        :obs_noise    => mdp.v,
        :mdp_discount => mdp.γ,

        # --- geometry ---
        :obcs         => mdp.obcs,
        :goals        => Any[],

        # --- misc ---
        :digits       => mdp.digits,
        :policy_temperature => 2.0
    )
end