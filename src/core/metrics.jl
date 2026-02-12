module Metrics

using ..Statistics

export compute_metrics, nse, kge, kge_components, pbias, r_squared, rmse,
       weighted_metric, transform_data, inverse_transform_data

# ==============================================================================
# 辅助函数
# ==============================================================================

"""安全对数变换，floor 防止 log(0)"""
_safe_log(x::AbstractVector; floor_val=1e-6) = log.(max.(x, floor_val))

function _maybe_transform(sim::AbstractVector, obs::AbstractVector, log_transform::Bool)
    if log_transform
        return _safe_log(sim), _safe_log(obs)
    end
    return sim, obs
end

# ==============================================================================
# 各评价指标实现
# ==============================================================================

"""
    nse(sim, obs; log_transform=false) -> Float64

Nash-Sutcliffe Efficiency。范围 (-Inf, 1]，最优值 = 1。
"""
function nse(sim::AbstractVector, obs::AbstractVector; log_transform::Bool=false)
    s, o = _maybe_transform(sim, obs, log_transform)
    numerator = sum((o .- s) .^ 2)
    denominator = sum((o .- mean(o)) .^ 2)
    denominator == 0.0 && return -Inf
    return 1.0 - numerator / denominator
end

"""
    kge(sim, obs; log_transform=false) -> Float64

Kling-Gupta Efficiency。范围 (-Inf, 1]，最优值 = 1。
返回 KGE 值，同时可通过 kge_components 获取 r, alpha, beta 分量。
"""
function kge(sim::AbstractVector, obs::AbstractVector; log_transform::Bool=false)
    s, o = _maybe_transform(sim, obs, log_transform)
    r_val = cor(s, o)
    alpha = std(s) / std(o)
    beta = mean(s) / mean(o)
    return 1.0 - sqrt((r_val - 1.0)^2 + (alpha - 1.0)^2 + (beta - 1.0)^2)
end

"""KGE 分量详情"""
function kge_components(sim::AbstractVector, obs::AbstractVector; log_transform::Bool=false)
    s, o = _maybe_transform(sim, obs, log_transform)
    r_val = cor(s, o)
    alpha = std(s) / std(o)
    beta = mean(s) / mean(o)
    kge_val = 1.0 - sqrt((r_val - 1.0)^2 + (alpha - 1.0)^2 + (beta - 1.0)^2)
    return Dict("KGE" => kge_val, "r" => r_val, "alpha" => alpha, "beta" => beta)
end

"""
    pbias(sim, obs) -> Float64

Percent Bias (%)。最优值 = 0。正值表示模拟偏高，负值表示模拟偏低。
"""
function pbias(sim::AbstractVector, obs::AbstractVector; kwargs...)
    s_obs = sum(obs)
    s_obs == 0.0 && return Inf
    return 100.0 * sum(sim .- obs) / s_obs
end

"""
    r_squared(sim, obs; log_transform=false) -> Float64

决定系数 r²。范围 [0, 1]，最优值 = 1。
"""
function r_squared(sim::AbstractVector, obs::AbstractVector; log_transform::Bool=false)
    s, o = _maybe_transform(sim, obs, log_transform)
    return cor(s, o)^2
end

"""
    rmse(sim, obs; log_transform=false) -> Float64

均方根误差。最优值 = 0。
"""
function rmse(sim::AbstractVector, obs::AbstractVector; log_transform::Bool=false)
    s, o = _maybe_transform(sim, obs, log_transform)
    return sqrt(mean((s .- o) .^ 2))
end

# ==============================================================================
# 指标分发表
# ==============================================================================

const METRIC_FUNCTIONS = Dict{String,Function}(
    "NSE"    => nse,
    "KGE"    => kge,
    "LogNSE" => (s, o; kw...) -> nse(s, o; log_transform=true),
    "LogKGE" => (s, o; kw...) -> kge(s, o; log_transform=true),
    "PBIAS"  => pbias,
    "R2"     => r_squared,
    "RMSE"   => rmse,
)

# ==============================================================================
# 批量计算接口
# ==============================================================================

"""
    compute_metrics(sim, obs, metric_names) -> Dict

批量计算指标。自动检测是否建议 log 变换 (Strategy 3)。
"""
function compute_metrics(sim::AbstractVector, obs::AbstractVector,
                         metric_names::Vector{String}=String["NSE","KGE","PBIAS","RMSE","R2"])
    results = Dict{String,Any}()

    for name in metric_names
        if haskey(METRIC_FUNCTIONS, name)
            try
                results[name] = METRIC_FUNCTIONS[name](sim, obs)
            catch e
                results[name] = "error: $(sprint(showerror, e))"
            end
        else
            results[name] = "unknown_metric"
        end
    end

    # Strategy 3: 自动检测数据量级跨度
    pos_obs = filter(>(0), obs)
    if length(pos_obs) > 0
        ratio = maximum(pos_obs) / minimum(pos_obs)
        results["_magnitude_ratio"] = ratio
        results["_log_transform_recommended"] = ratio > 100.0
    end

    return results
end

"""根据目标名获取单个指标值（供优化器内部调用）"""
function get_metric_value(sim::AbstractVector, obs::AbstractVector, metric_name::String)
    haskey(METRIC_FUNCTIONS, metric_name) || throw(ArgumentError("未知指标: $metric_name"))
    return METRIC_FUNCTIONS[metric_name](sim, obs)
end

"""判断指标是否越大越好（用于优化方向）"""
function is_higher_better(metric_name::String)
    return metric_name in ("NSE", "KGE", "LogNSE", "LogKGE", "R2")
end

# ==============================================================================
# 加权组合指标 (DEV.md 模块 B: calc_weighted_metric)
# ==============================================================================

"""
    weighted_metric(sim, obs, metrics_weights) -> Float64

多目标加权单目标化。将多个指标按权重组合为单一标量。

参数:
- metrics_weights: Dict{String, Float64}，指标名 -> 权重
  例如 Dict("KGE" => 0.6, "LogKGE" => 0.4)

所有指标先归一化到 [0,1] 方向（越大越好），再加权求和。
"""
function weighted_metric(sim::AbstractVector, obs::AbstractVector,
                         metrics_weights::Dict{String,Float64})
    total_weight = sum(values(metrics_weights))
    total_weight > 0 || throw(ArgumentError("权重之和必须大于 0"))

    weighted_sum = 0.0
    for (name, weight) in metrics_weights
        val = get_metric_value(sim, obs, name)
        # 归一化方向：越大越好的指标直接用，越小越好的取负
        normalized = is_higher_better(name) ? val : -val
        weighted_sum += weight * normalized
    end
    return weighted_sum / total_weight
end

# ==============================================================================
# 数据变换 (DEV.md 模块 A DataOps: transform_data)
# ==============================================================================

"""
    transform_data(data, method; kwargs...) -> (transformed, params)

对数据进行变换。返回变换后的数据和变换参数（用于逆变换）。

方法:
- :log  — 自然对数变换 (带 floor 保护)
- :log10 — 常用对数变换
- :boxcox — Box-Cox 变换 (需指定 lambda，默认自动估计)
"""
function transform_data(data::AbstractVector, method::Symbol; lambda::Float64=NaN)
    if method == :log
        floor_val = 1e-6
        transformed = log.(max.(data, floor_val))
        return transformed, Dict{String,Any}("method" => "log", "floor" => floor_val)
    elseif method == :log10
        floor_val = 1e-6
        transformed = log10.(max.(data, floor_val))
        return transformed, Dict{String,Any}("method" => "log10", "floor" => floor_val)
    elseif method == :boxcox
        # Box-Cox: y = (x^λ - 1) / λ  (λ ≠ 0), y = ln(x) (λ = 0)
        if isnan(lambda)
            lambda = _estimate_boxcox_lambda(data)
        end
        floor_val = 1e-6
        x = max.(data, floor_val)
        transformed = if abs(lambda) < 1e-10
            log.(x)
        else
            (x .^ lambda .- 1.0) ./ lambda
        end
        return transformed, Dict{String,Any}("method" => "boxcox", "lambda" => lambda, "floor" => floor_val)
    else
        throw(ArgumentError("未知变换方法: $(method)。可选: :log, :log10, :boxcox"))
    end
end

"""
    inverse_transform_data(data, params) -> Vector{Float64}

逆变换，将变换后的数据恢复到原始空间。
"""
function inverse_transform_data(data::AbstractVector, params::Dict{String,Any})
    method = params["method"]
    if method == "log"
        return exp.(data)
    elseif method == "log10"
        return 10.0 .^ data
    elseif method == "boxcox"
        lambda = params["lambda"]
        if abs(lambda) < 1e-10
            return exp.(data)
        else
            return (data .* lambda .+ 1.0) .^ (1.0 / lambda)
        end
    else
        throw(ArgumentError("未知逆变换方法: $(method)"))
    end
end

"""估计 Box-Cox 最优 lambda (简化：基于对数似然的网格搜索)"""
function _estimate_boxcox_lambda(data::AbstractVector)
    pos_data = filter(>(0), data)
    isempty(pos_data) && return 0.0

    best_lambda = 0.0
    best_ll = -Inf
    for lam in -2.0:0.1:2.0
        transformed = if abs(lam) < 1e-10
            log.(pos_data)
        else
            (pos_data .^ lam .- 1.0) ./ lam
        end
        # 对数似然 (正态假设)
        n = length(transformed)
        sigma2 = var(transformed)
        sigma2 <= 0 && continue
        ll = -n / 2 * log(sigma2) + (lam - 1) * sum(log.(pos_data))
        if ll > best_ll
            best_ll = ll
            best_lambda = lam
        end
    end
    return best_lambda
end

end # module
