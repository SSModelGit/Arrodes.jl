@with_kw_noshow struct KAgentMDPConfig
    start::Matrix{Float64}
    dimensions::Tuple{Float64,Float64}
    menv::MuKumari.MuEnv
    obstacles::Vector = Any[]
    goals::Vector = Any[]
    digits::Int = 3
    agent_width::Float64 = 0.1
    agent_speed::Float64 = 1.0
    movement_noise::Float64 = 0.05
    observation_noise::Float64 = 0.05
    discount::Float64 = 0.95
end

"""Construct a native MuKumari POMDP with one explicitly supplied objective."""
function build_kagent_pomdp(config::KAgentMDPConfig, objective::Function;
                            name="objective_hypothesis")
    lo, hi = config.dimensions
    exterior = GI.Polygon([[
        (lo, lo), (lo, hi), (hi, hi), (hi, lo), (lo, lo),
    ]])
    world = isempty(config.obstacles) ? exterior : GI.Polygon([
        GI.getexterior(exterior),
        (GI.getexterior(obstacle) for obstacle in config.obstacles)...,
    ])
    landscape = MuKumari.AgentObjectiveLandscape(objectives=Any[], f_types=Any[])
    MuKumari.KAgentPOMDP(
        name=name,
        start=config.start,
        dimensions=config.dimensions,
        boxworld=exterior,
        objl=landscape,
        obcs=config.obstacles,
        goals=config.goals,
        obj=objective,
        world=world,
        width=config.agent_width,
        s=config.agent_speed,
        w=config.movement_noise,
        menv=config.menv,
        v=config.observation_noise,
        γ=config.discount,
        digits=config.digits,
    )
end

function build_shared_menv(spec::MuEnvSpec=MuEnvSpec())
    fields = [
        (:sin, x -> sin(x[1]) + cos(x[2])),
        (:exp, x -> 100exp(-norm(x .- [8 8])^2)),
        (:lin, x -> x[1]^2 + x[2]),
    ]
    MuKumari.MuEnv(spec.M, spec.μ_order, Dict(fields))
end

"""Copy the dynamics and geometry of a MuKumari agent without its objective."""
function agent_config_from_mdp(mdp; environment=mdp.menv)
    KAgentMDPConfig(
        start=Matrix{Float64}(mdp.start),
        dimensions=Tuple{Float64,Float64}(mdp.dimensions),
        menv=environment,
        obstacles=copy(mdp.obcs),
        goals=Any[],
        digits=mdp.digits,
        agent_width=mdp.width,
        agent_speed=mdp.s,
        movement_noise=mdp.w,
        observation_noise=mdp.v,
        discount=mdp.γ,
    )
end
