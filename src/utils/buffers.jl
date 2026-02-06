"""
    mk_experience_buffer(data::Dict)

Construct exactly: ExperienceBuffer(data, max_steps, 1, Array{Int64}[], nothing, 0)
where max_steps is the number of columns in data[:s].
"""
function mk_experience_buffer(data::Dict{Symbol, Matrix})
    max_steps = size(data[:s], 2)
    return ExperienceBuffer(data, max_steps, 1, Array{Int64}[], nothing, 0)
end

function alloc_buffer_dict(obs_dims::Int, a_dims::Int, max_steps::Int)
    a_list = Matrix{Bool}(undef, a_dims, max_steps)

    s_list  = zeros(Float64, obs_dims, max_steps)
    sp_list = zeros(Float64, obs_dims, max_steps)

    expert_val_list = ones(Float32, 1, max_steps)
    r_list   = Matrix{Float64}(undef, 1, max_steps)
    t_list   = Matrix{Int64}(undef, 1, max_steps)
    done_list = Matrix{Bool}(undef, 1, max_steps)

    # initialize the ones that must be deterministic
    t_list[1, :] .= collect(Int64, 1:max_steps)
    done_list[1, :] .= false

    return Dict(
        :a => a_list,
        :s => s_list,
        :sp => sp_list,
        :r => r_list,
        :t => t_list,
        :expert_val => expert_val_list,
        :done => done_list,
    )
end

"""
    wrap_like(template_buf, data)

Create a new ExperienceBuffer by cloning `template_buf` and replacing `.data`
(and step counters) so Crux/IQL code accepts it.
"""
function wrap_like(template_buf, data::Dict{Symbol,Any})
    buf = deepcopy(template_buf)
    buf.data = data
    if hasproperty(buf, :elements)
        buf.elements = size(data[:s], 2)
    end
    if hasproperty(buf, :max_steps)
        buf.max_steps = size(data[:s], 2)
    end
    return buf
end

"""
    anonymize_buffer_location!(buf)

Zeroes out the first two rows (location dims) of :s and :sp.
Works in-place on ExperienceBuffer (buf.data is a Dict).
"""
function anonymize_buffer_location!(buf)
    @assert hasproperty(buf, :data) "Expected an ExperienceBuffer-like object with `.data`"
    D = buf.data
    @assert haskey(D, :s) && haskey(D, :sp) "Buffer data missing :s or :sp"

    @assert size(D[:s], 1) ≥ 2 && size(D[:sp], 1) ≥ 2 "State obs dim < 2; cannot anonymize first two rows"

    D[:s][1:2, :] .= 0.0
    D[:sp][1:2, :] .= 0.0
    return buf
end

"""
    onehot_cols_to_aidx(A::AbstractMatrix) -> Vector{Int}

Convert action matrix A (nactions × T) where each column is one-hot
(or nearly one-hot) into indices aidx[t] ∈ 1:nactions.

Uses argmax per column. Returns vector of integer indices.
"""
function onehot_cols_to_aidx(A::AbstractMatrix; tol::Real=1e-8)
    na, T = size(A)
    aidx = Vector{Int}(undef, T)
    @inbounds for t in 1:T
        col = view(A, :, t)
        # index of maximum entry (should map as idx within actions(mdp))
        aidx[t] = argmax(col)
    end
    return aidx
end

"""
    data_cleaner(data::ExperienceBuffer, state_field_sizes::Vector{Int64}=[2, 2, 12, 10, 1], keep_state_fields::Vector{Bool}=Bool[1,1,1,0,1])

Helper for cleaning up older data buffer in ways necessary to keep code running.
"""
function data_cleaner(data::ExperienceBuffer, state_field_sizes::Vector{Int64}=[2, 2, 12, 10, 1], keep_state_fields::Vector{Bool}=Bool[1,1,1,0,1])
    # verify elements keeps right size
    actual_size = size(data.data[:s])[2]
    if data.elements ≠ actual_size
        data.elements = actual_size
    end

    # clean state and next-state vectors
    idx = 1
    keep_idxs = []
    for (i, field_size) in enumerate(state_field_sizes)
        if keep_state_fields[i]
            append!(keep_idxs, idx:(idx+field_size-1))
        end
        idx += field_size
    end
    data.data[:s] = data.data[:s][keep_idxs, :]
    data.data[:sp] = data.data[:sp][keep_idxs, :]

    return data
end