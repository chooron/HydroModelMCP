module DataSplitter

using Dates

export split_data

"""
    split_data(obs, forcing_nt; kwargs...) -> Dict

将观测数据和驱动数据划分为校准集和验证集（train/test）。

自动划分方法:
- "recent_first": 最新数据用于校准 (适合运行预测部署)
- "split_sample": 经典前后分割 (适合过程理解/Klemes 检验)
- "use_all": 全部数据用于校准，无验证集 (适合最终部署前的最大化利用)

也支持显式时间段划分：
- calibration_period + validation_period (start/end 或 start_index/end_index)
- 若只给了 start/end 且 forcing 无日期列，将基于请求时间边界构造 synthetic timeline

参数:
- obs: 观测值向量
- forcing_nt: 驱动数据 NamedTuple
- method: 自动划分方法
- ratio: 校准集占比 (0-1)
- warmup: 自动划分预热期长度 (时间步数)
- calibration_period / validation_period: 显式时间段（可选）
- dates: 与 forcing 对齐的日期序列（可选）
"""
const DATE_PARSE_FORMATS = (
    dateformat"yyyy-mm-dd",
    dateformat"yyyy/mm/dd",
    dateformat"yyyymmdd",
    dateformat"yyyy-mm-dd HH:MM:SS",
    dateformat"yyyy/mm/dd HH:MM:SS",
    dateformat"yyyy-mm-ddTHH:MM:SS",
    dateformat"yyyy-mm-ddTHH:MM:SS.s",
    dateformat"yyyy-mm-ddTHH:MM:SS.sss",
)

_as_string_dict(value::AbstractDict) = Dict{String,Any}(string(k) => v for (k, v) in pairs(value))

function _slice_forcing(forcing_nt::NamedTuple, indices)
    names = keys(forcing_nt)
    values = Tuple([forcing_nt[name][indices] for name in names])
    return NamedTuple{names}(values)
end

function _trim_to_common_length(obs::AbstractVector, forcing_nt::NamedTuple, dates)
    forcing_lengths = Int[length(v) for v in values(forcing_nt)]
    forcing_min = isempty(forcing_lengths) ? length(obs) : minimum(forcing_lengths)
    n = minimum((length(obs), forcing_min))
    n >= 1 || throw(ArgumentError("obs and forcing series must contain at least one sample"))

    obs_trimmed = Float64.(collect(obs[1:n]))
    forcing_trimmed = _slice_forcing(forcing_nt, 1:n)
    dates_trimmed = isnothing(dates) ? nothing : Date.(dates[1:n])
    return obs_trimmed, forcing_trimmed, dates_trimmed, n
end

function _parse_date_like(value, field::String)
    if value isa Date
        return value
    elseif value isa DateTime
        return Date(value)
    end

    text = strip(string(value))
    isempty(text) && throw(ArgumentError("$field cannot be empty"))

    for fmt in DATE_PARSE_FORMATS
        try
            return Date(text, fmt)
        catch
        end

        try
            return Date(DateTime(text, fmt))
        catch
        end
    end

    try
        return Date(text)
    catch
    end

    try
        return Date(DateTime(text))
    catch
    end

    throw(ArgumentError("$field must be parseable as date, for example 2020-01-01"))
end

function _build_synthetic_dates(calibration_period::AbstractDict, validation_period::AbstractDict, length_hint::Int)
    cal_start = _parse_date_like(calibration_period["start"], "calibration.start")
    cal_end = _parse_date_like(calibration_period["end"], "calibration.end")
    val_start = _parse_date_like(validation_period["start"], "validation.start")
    val_end = _parse_date_like(validation_period["end"], "validation.end")

    cal_start <= cal_end || throw(ArgumentError("calibration.start must be <= calibration.end"))
    val_start <= val_end || throw(ArgumentError("validation.start must be <= validation.end"))

    global_start = minimum((cal_start, val_start))
    global_end = maximum((cal_end, val_end))
    horizon_days = max(Dates.value(global_end - global_start), 1)

    length_hint >= 1 || throw(ArgumentError("length_hint must be >= 1"))
    length_hint == 1 && return [global_start]

    return [
        global_start + Day(round(Int, (idx - 1) * horizon_days / (length_hint - 1)))
        for idx in 1:length_hint
    ]
end

function _period_indices(period::AbstractDict, dates::Union{Nothing,Vector{Date}}, length_hint::Int, role::String)
    if haskey(period, "start_index") && haskey(period, "end_index")
        start_idx = Int(period["start_index"])
        end_idx = Int(period["end_index"])
        1 <= start_idx <= end_idx <= length_hint ||
            throw(ArgumentError("$role period indices out of bounds: [$start_idx, $end_idx], length=$length_hint"))
        return start_idx:end_idx
    end

    haskey(period, "start") && haskey(period, "end") ||
        throw(ArgumentError("$role period must include either start/end or start_index/end_index"))

    if isnothing(dates)
        start_idx = try
            Int(period["start"])
        catch
            nothing
        end
        end_idx = try
            Int(period["end"])
        catch
            nothing
        end

        if !(start_idx === nothing || end_idx === nothing)
            1 <= start_idx <= end_idx <= length_hint ||
                throw(ArgumentError("$role period indices out of bounds: [$start_idx, $end_idx], length=$length_hint"))
            return start_idx:end_idx
        end

        throw(ArgumentError(
            "$role period uses start/end, but dates are unavailable. Provide start_index/end_index, or pass dates.",
        ))
    end

    start_date = _parse_date_like(period["start"], "$role.start")
    end_date = _parse_date_like(period["end"], "$role.end")
    start_date <= end_date || throw(ArgumentError("$role.start must be <= $role.end"))

    idx = findall(d -> start_date <= d <= end_date, dates)
    isempty(idx) && throw(ArgumentError("$role period selected no samples"))
    return idx
end

function _split_auto_ranges(n::Int, method::String, ratio::Float64, warmup::Int)
    method in ("recent_first", "split_sample", "use_all") || throw(ArgumentError(
        "未知划分方法: $(method)。可选: recent_first, split_sample, use_all",
    ))

    if method != "use_all"
        0.0 < ratio < 1.0 || throw(ArgumentError("ratio must be in (0, 1) for method '$method'"))
    end

    if method == "recent_first"
        split_idx = n - round(Int, n * ratio)
        cal_start = max(split_idx, warmup + 1)
        cal_range = cal_start:n
        val_end = cal_start - 1
        val_range = val_end >= (warmup + 1) ? ((warmup + 1):val_end) : (1:0)
        return cal_range, val_range
    elseif method == "split_sample"
        split_idx = clamp(round(Int, n * ratio), warmup + 1, n)
        cal_range = (warmup + 1):split_idx
        val_range = split_idx < n ? ((split_idx + 1):n) : (1:0)
        return cal_range, val_range
    end

    return (warmup + 1):n, 1:0
end

function split_data(obs::AbstractVector, forcing_nt::NamedTuple;
                    method::String="split_sample",
                    ratio::Float64=0.7,
                    warmup::Int=365,
                    calibration_period=nothing,
                    validation_period=nothing,
                    dates=nothing)
    has_period_cal = !(calibration_period === nothing)
    has_period_val = !(validation_period === nothing)
    has_period_cal == has_period_val || throw(ArgumentError(
        "calibration_period and validation_period must be provided together",
    ))

    dates_vec = if isnothing(dates)
        nothing
    elseif dates isa AbstractVector
        Date[_parse_date_like(d, "dates") for d in dates]
    else
        throw(ArgumentError("dates must be an array when provided"))
    end

    obs_vec, forcing_vec, dates_trimmed, n = _trim_to_common_length(obs, forcing_nt, dates_vec)
    warmup_cap = min(max(Int(warmup), 0), div(n, 4))

    cal_range = nothing
    val_range = nothing
    split_mode = "auto"
    method_used = method
    warmup_used = warmup_cap
    used_synthetic_dates = false
    cal_period_out = nothing
    val_period_out = nothing

    if has_period_cal && has_period_val
        split_mode = "period"
        method_used = "period_split"
        warmup_used = 0

        cal_period = _as_string_dict(calibration_period)
        val_period = _as_string_dict(validation_period)
        cal_period_out = cal_period
        val_period_out = val_period

        if isnothing(dates_trimmed) &&
           haskey(cal_period, "start") && haskey(cal_period, "end") &&
           haskey(val_period, "start") && haskey(val_period, "end")
            dates_trimmed = _build_synthetic_dates(cal_period, val_period, n)
            used_synthetic_dates = true
        end

        cal_range = _period_indices(cal_period, dates_trimmed, n, "calibration")
        val_range = _period_indices(val_period, dates_trimmed, n, "validation")
    else
        cal_range, val_range = _split_auto_ranges(n, method, ratio, warmup_cap)
    end

    isempty(cal_range) && throw(ArgumentError("calibration/train split is empty"))

    cal_forcing = _slice_forcing(forcing_vec, cal_range)
    val_forcing = isempty(val_range) ? nothing : _slice_forcing(forcing_vec, val_range)

    cal_obs = Float64.(collect(obs_vec[cal_range]))
    val_obs = isempty(val_range) ? Float64[] : Float64.(collect(obs_vec[val_range]))

    return Dict{String,Any}(
        "cal_obs" => cal_obs,
        "val_obs" => val_obs,
        "cal_forcing" => cal_forcing,
        "val_forcing" => val_forcing,
        "cal_indices" => [first(cal_range), last(cal_range)],
        "val_indices" => isempty(val_range) ? Int[] : [first(val_range), last(val_range)],
        "method" => method_used,
        "split_mode" => split_mode,
        "warmup" => warmup_used,
        "total_length" => n,
        "cal_length" => length(cal_range),
        "val_length" => length(val_range),
        "train_obs" => cal_obs,
        "test_obs" => val_obs,
        "train_forcing" => cal_forcing,
        "test_forcing" => val_forcing,
        "train_indices" => [first(cal_range), last(cal_range)],
        "test_indices" => isempty(val_range) ? Int[] : [first(val_range), last(val_range)],
        "calibration_period" => cal_period_out,
        "validation_period" => val_period_out,
        "dates_available" => !isnothing(dates_trimmed),
        "used_synthetic_dates" => used_synthetic_dates,
    )
end

end # module
