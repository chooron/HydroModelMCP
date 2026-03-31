module SensitivityAnalysis

using ..GlobalSensitivity
using ..HydroModels
using ..HydroModelLibrary
using ..ComponentArrays
using ..DataInterpolations
using ..Statistics
using ..Metrics
using ..Simulation: _execute_core

export run_sensitivity

"""
    run_sensitivity(model_name, forcing_nt, obs; kwargs...) -> Dict

对水文模型参数进行全局敏感性分析 (Strategy 1)。

识别哪些参数对模型输出有显著影响（应校准），哪些可以固定为默认值。

参数:
- model_name: 模型名称
- forcing_nt: 驱动数据 NamedTuple
- obs: 观测值向量
- method: "morris" 或 "sobol"
- n_samples: 采样数
- objective: 评价指标名
- threshold: 敏感性阈值
- solver_type / interp_type: 求解器和插值器类型
"""
function run_sensitivity(model_name::String, forcing_nt::NamedTuple,
                         obs::AbstractVector;
                         method::String="morris",
                         n_samples::Int=100,
                         objective::String="NSE",
                         threshold::Float64=0.1,
                         solver_type::String="DISCRETE",
                         interp_type::String="LINEAR",
                         param_bounds_override::Union{Nothing,Dict{String,Vector{Float64}}}=nothing,
                         param_names_filter::Union{Nothing,Vector{String}}=nothing)
    # 1. 加载模型
    valid_name = String.(HydroModelLibrary.AVAILABLE_MODELS)
    idx = findfirst(m -> lowercase(m) == lowercase(strip(model_name)), valid_name)
    isnothing(idx) && throw(ArgumentError("模型 '$(model_name)' 未找到"))
    canonical = valid_name[idx]

    model_module = HydroModelLibrary.load_model(Symbol(canonical), reload=false)
    model = Base.invokelatest(m -> m.model, model_module)

    # 2. 获取参数信息和 bounds
    all_param_names = string.(HydroModels.get_param_names(model))
    wrapper_params = Base.invokelatest(m -> m.model_parameters, model_module)

    # 注意：wrapper_params 的顺序不一定与 get_param_names(model) 一致。
    # 必须按参数名匹配边界，避免错配导致大量无效参数组合。
    bounds_by_name = Dict{String,Tuple{Float64,Float64}}()
    for p in wrapper_params
        pname = string(p)
        bounds = try
            b = HydroModels.getbounds(p)
            (Float64(b[1]), Float64(b[2]))
        catch
            (0.0, 10.0)  # 默认范围
        end
        bounds_by_name[pname] = bounds
    end

    all_lb = Float64[]
    all_ub = Float64[]
    for pname in all_param_names
        # 优先使用 param_bounds_override，否则按参数名使用模型默认范围
        bounds = if !isnothing(param_bounds_override) && haskey(param_bounds_override, pname)
            b = param_bounds_override[pname]
            (Float64(b[1]), Float64(b[2]))
        else
            get(bounds_by_name, pname, (0.0, 10.0))
        end

        bounds[1] <= bounds[2] ||
            throw(ArgumentError("参数 '$pname' 边界无效: lower=$(bounds[1]) > upper=$(bounds[2])"))

        push!(all_lb, bounds[1])
        push!(all_ub, bounds[2])
    end

    # 应用参数名筛选 (param_names_filter)
    active_indices = if !isnothing(param_names_filter)
        filter(i -> all_param_names[i] in param_names_filter, 1:length(all_param_names))
    else
        collect(1:length(all_param_names))
    end

    param_names = all_param_names[active_indices]
    lb = all_lb[active_indices]
    ub = all_ub[active_indices]

    # 默认参数值（用于固定未分析的参数）
    default_params = [(all_lb[i] + all_ub[i]) / 2.0 for i in 1:length(all_param_names)]

    # 3. 初始化状态
    state_names = HydroModels.get_state_names(model)
    init_states = ComponentVector(NamedTuple{Tuple(state_names)}(zeros(length(state_names))))

    solver_sym = Symbol(uppercase(solver_type))
    interp_sym = Symbol(uppercase(interp_type))

    # 4. 构建目标函数: active_params_vector -> scalar
    function objective_fn(active_vec::AbstractVector)
        try
            # 组合完整参数向量（固定参数用默认值）
            full_vec = copy(default_params)
            for (j, idx) in enumerate(active_indices)
                full_vec[idx] = active_vec[j]
            end
            pnames = HydroModels.get_param_names(model)
            pv = ComponentVector(; params=ComponentVector(NamedTuple{Tuple(pnames)}(full_vec)))
            out = _execute_core(model, forcing_nt, pv, init_states,
                                solver_sym, interp_sym, Dict{String,Any}())
            sim = Float64.(out)
            n = min(length(sim), length(obs))
            return Metrics.get_metric_value(sim[1:n], obs[1:n], objective)
        catch
            return -Inf  # 模型崩溃时返回最差值
        end
    end

    # 5. 执行敏感性分析
    sensitivities = if method == "morris"
        _run_morris(objective_fn, lb, ub, n_samples)
    elseif method == "sobol"
        _run_sobol(objective_fn, lb, ub, n_samples)
    else
        throw(ArgumentError("未知敏感性分析方法: $(method)。可选: morris, sobol"))
    end

    # 6. 分类参数
    calibratable = String[]
    fixed = String[]
    for (i, s) in enumerate(sensitivities)
        if s >= threshold
            push!(calibratable, param_names[i])
        else
            push!(fixed, param_names[i])
        end
    end

    return Dict{String,Any}(
        "model_name"    => canonical,
        "param_names"   => param_names,
        "param_bounds"  => Dict(param_names[i] => [lb[i], ub[i]] for i in eachindex(param_names)),
        "sensitivities" => sensitivities,
        "calibratable"  => calibratable,
        "fixed"         => fixed,
        "method"        => method,
        "threshold"     => threshold,
        "n_samples"     => n_samples,
        "objective"     => objective
    )
end

# ==============================================================================
# Morris Method
# ==============================================================================

function _run_morris(f::Function, lb::Vector{Float64}, ub::Vector{Float64}, n_samples::Int)
    method = GlobalSensitivity.Morris(num_trajectory=n_samples)
    result = GlobalSensitivity.gsa(f, method, [[lb[i], ub[i]] for i in eachindex(lb)])
    # Morris 返回 means_star (绝对均值) 作为敏感性度量
    # means_star 是一个矩阵，需要转换为向量
    mu_star = vec(result.means_star)
    # 归一化到 [0, 1]
    max_val = maximum(abs.(mu_star))
    return max_val > 0 ? abs.(mu_star) ./ max_val : zeros(length(lb))
end

# ==============================================================================
# Sobol Method
# ==============================================================================

function _run_sobol(f::Function, lb::Vector{Float64}, ub::Vector{Float64}, n_samples::Int)
    method = GlobalSensitivity.Sobol(nboot=20)
    # 抑制 QuasiMonteCarlo 的随机化警告
    # GlobalSensitivity v2.10.0 尚未暴露随机化配置选项
    result = Base.with_logger(Base.NullLogger()) do
        GlobalSensitivity.gsa(f, method, [[lb[i], ub[i]] for i in eachindex(lb)],
                             samples=n_samples)
    end
    # Sobol 返回 ST (Total-order indices) 作为敏感性度量
    st = result.ST
    return Float64.(st)
end

end # module
