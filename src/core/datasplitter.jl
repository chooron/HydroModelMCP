module DataSplitter

export split_data

"""
    split_data(obs, forcing_nt; method, ratio, warmup) -> Dict

将观测数据和驱动数据划分为校准集和验证集。

方法 (Strategy 4):
- "recent_first": 最新数据用于校准 (适合运行预测部署)
- "split_sample": 经典前后分割 (适合过程理解/Klemes 检验)
- "use_all": 全部数据用于校准，无验证集 (适合最终部署前的最大化利用)

参数:
- obs: 观测值向量
- forcing_nt: 驱动数据 NamedTuple
- method: 划分方法
- ratio: 校准集占比 (0-1)
- warmup: 预热期长度 (时间步数)
"""
function split_data(obs::AbstractVector, forcing_nt::NamedTuple;
                    method::String="split_sample",
                    ratio::Float64=0.7,
                    warmup::Int=365)
    n = length(obs)
    warmup = min(warmup, div(n, 4))  # 安全上限

    cal_range, val_range = if method == "recent_first"
        # 最新数据优先用于校准
        split_idx = n - round(Int, n * ratio)
        (max(split_idx, warmup + 1):n, (warmup + 1):(split_idx - 1))
    elseif method == "split_sample"
        # 经典：前段校准，后段验证
        split_idx = round(Int, n * ratio)
        ((warmup + 1):split_idx, (split_idx + 1):n)
    elseif method == "use_all"
        ((warmup + 1):n, 1:0)  # 空 range
    else
        throw(ArgumentError("未知划分方法: $(method)。可选: recent_first, split_sample, use_all"))
    end

    # 构建分割后的 forcing NamedTuple
    cal_forcing = NamedTuple{keys(forcing_nt)}(
        Tuple(v[cal_range] for v in values(forcing_nt))
    )

    val_forcing = if !isempty(val_range)
        NamedTuple{keys(forcing_nt)}(
            Tuple(v[val_range] for v in values(forcing_nt))
        )
    else
        nothing
    end

    return Dict{String,Any}(
        "cal_obs"      => collect(obs[cal_range]),
        "val_obs"      => isempty(val_range) ? Float64[] : collect(obs[val_range]),
        "cal_forcing"  => cal_forcing,
        "val_forcing"  => val_forcing,
        "cal_indices"  => [first(cal_range), last(cal_range)],
        "val_indices"  => isempty(val_range) ? Int[] : [first(val_range), last(val_range)],
        "method"       => method,
        "warmup"       => warmup,
        "total_length" => n,
        "cal_length"   => length(cal_range),
        "val_length"   => length(val_range)
    )
end

end # module
