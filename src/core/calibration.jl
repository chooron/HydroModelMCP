module Calibration

using ..Optimization
using ..OptimizationBBO
using ..OptimizationMetaheuristics
using ..Metaheuristics
using ..OptimizationStrategies
using ..HydroModels
using ..HydroModelLibrary
using ..ComponentArrays
using ..DataInterpolations
using ..Random
using ..Statistics
using ..Metrics
using ..Sampling
using ..Simulation: _execute_core

export calibrate_model, calibrate_multiobjective, diagnose_calibration, diagnose_multiobjective, diagnose_calibration_full

const OBJECTIVE_DIRECTION = Dict(
    "NSE" => :max,
    "KGE" => :max,
    "LogNSE" => :max,
    "LogKGE" => :max,
    "R2" => :max,
    "PBIAS" => :abs_min,
    "RMSE" => :min,
)

function _objective_score(metric_name::String, metric_value::Float64)
    direction = get(OBJECTIVE_DIRECTION, metric_name, :max)
    if direction == :max
        return metric_value
    elseif direction == :min
        return -metric_value
    elseif direction == :abs_min
        return -abs(metric_value)
    end

    return metric_value
end

function _objective_loss(metric_name::String, metric_value::Float64)
    return -_objective_score(metric_name, metric_value)
end

function _score_to_metric(metric_name::String, score::Float64)
    direction = get(OBJECTIVE_DIRECTION, metric_name, :max)
    if direction == :max
        return score
    elseif direction == :min
        return -score
    elseif direction == :abs_min
        return -score
    end

    return score
end

function _evaluate_objective(metric_name::String, sim::AbstractVector, obs::AbstractVector; log_transform::Bool = false)
    metric_value = if metric_name == "LogNSE"
        Metrics.nse(sim, obs; log_transform = true)
    elseif metric_name == "LogKGE"
        Metrics.kge(sim, obs; log_transform = true)
    elseif metric_name == "NSE"
        Metrics.nse(sim, obs; log_transform = log_transform)
    elseif metric_name == "KGE"
        Metrics.kge(sim, obs; log_transform = log_transform)
    elseif metric_name == "RMSE"
        Metrics.rmse(sim, obs; log_transform = log_transform)
    elseif metric_name == "R2"
        Metrics.r_squared(sim, obs; log_transform = log_transform)
    elseif metric_name == "MAE"
        Metrics.mae(sim, obs; log_transform = log_transform)
    elseif metric_name == "Bias"
        Metrics.bias(sim, obs; log_transform = log_transform)
    else
        Metrics.get_metric_value(sim, obs, metric_name)
    end

    return metric_value, _objective_loss(metric_name, metric_value)
end

function _infer_log_transform(obs::AbstractVector, objective_name::String, requested_log_transform::Bool)
    if requested_log_transform
        return true, Dict{String,Any}(
            "enabled" => true,
            "mode" => "manual",
            "reason" => "user_requested",
            "objective" => objective_name,
        )
    end

    ratio = 1.0
    positive_obs = filter(>(0), Float64.(obs))
    if !isempty(positive_obs)
        ratio = maximum(positive_obs) / max(minimum(positive_obs), 1e-12)
    end

    objective_requires_log = objective_name in ("LogNSE", "LogKGE")
    enabled = objective_requires_log || ratio > 100.0
    reason = if objective_requires_log
        "objective_requires_log"
    elseif enabled
        "auto_magnitude_ratio"
    else
        "native_scale_sufficient"
    end

    return enabled, Dict{String,Any}(
        "enabled" => enabled,
        "mode" => "auto",
        "reason" => reason,
        "magnitude_ratio" => ratio,
        "objective" => objective_name,
    )
end

function _build_constraint_functions(
    all_param_names::Vector{Symbol},
    fixed_values_sym::Dict{Symbol,Float64},
    cal_names::Vector{Symbol},
    constraints_cfg,
)
    if constraints_cfg === nothing
        return (x -> 0.0), String[]
    end

    cal_lookup = Dict(cal_names[i] => i for i in eachindex(cal_names))
    fixed_lookup = Dict(fixed_values_sym)

    function full_param_value(name::Symbol, candidate::Vector{Float64})
        if haskey(cal_lookup, name)
            return candidate[cal_lookup[name]]
        elseif haskey(fixed_lookup, name)
            return fixed_lookup[name]
        end
        return NaN
    end

    checks = Function[]
    labels = String[]

    if haskey(constraints_cfg, "pie_share")
        pie = constraints_cfg["pie_share"]
        pie_params = Symbol.(String.(pie["parameters"]))
        total = Float64(pie["total"])
        push!(checks, function(candidate::Vector{Float64})
            values = [full_param_value(name, candidate) for name in pie_params]
            any(isnan, values) && return 1e3
            return abs(sum(values) - total)
        end)
        push!(labels, "pie_share")
    end

    if haskey(constraints_cfg, "delta_method")
        inequalities = constraints_cfg["delta_method"]["inequalities"]
        for (lower_name, upper_name) in inequalities
            lower_sym = Symbol(lower_name)
            upper_sym = Symbol(upper_name)
            push!(checks, function(candidate::Vector{Float64})
                lower_val = full_param_value(lower_sym, candidate)
                upper_val = full_param_value(upper_sym, candidate)
                (isnan(lower_val) || isnan(upper_val)) && return 1e3
                return max(0.0, lower_val - upper_val + 1e-8)
            end)
            push!(labels, string(lower_name, "<", upper_name))
        end
    end

    function penalty(candidate::Vector{Float64})
        value = 0.0
        for check_fn in checks
            violation = check_fn(candidate)
            value += violation^2
        end
        return value
    end

    return penalty, labels
end

function _params_to_component_vector(all_param_names::Vector{Symbol}, full_params::Vector{Float64})
    return ComponentVector(; params = ComponentVector(
        NamedTuple{Tuple(all_param_names)}(Tuple(full_params))
    ))
end

function _build_full_params_template(
    all_param_names::Vector{Symbol},
    cal_names::Vector{Symbol},
    fixed_values_sym::Dict{Symbol,Float64},
)
    cal_lookup = Dict(cal_names[i] => i for i in eachindex(cal_names))
    function materialize(candidate::Vector{Float64})
        out = Vector{Float64}(undef, length(all_param_names))
        for (idx, pname) in enumerate(all_param_names)
            if haskey(cal_lookup, pname)
                out[idx] = candidate[cal_lookup[pname]]
            else
                out[idx] = fixed_values_sym[pname]
            end
        end
        return out
    end
    return materialize
end

function _calibration_objective(
    model,
    forcing_nt::NamedTuple,
    obs::AbstractVector,
    all_param_names::Vector{Symbol},
    cal_names::Vector{Symbol},
    fixed_values_sym::Dict{Symbol,Float64},
    init_states,
    solver_sym::Symbol,
    interp_sym::Symbol,
    config_dict::AbstractDict,
    objective_name::String,
    constraint_penalty::Function;
    log_transform::Bool = false,
    warm_up::Int = 1,
)
    materialize = _build_full_params_template(all_param_names, cal_names, fixed_values_sym)

    return function(candidate::Vector{Float64})
        full_params = materialize(candidate)
        params_vec = _params_to_component_vector(all_param_names, full_params)

        sim = _execute_core(
            model,
            forcing_nt,
            params_vec,
            init_states,
            solver_sym,
            interp_sym,
            config_dict,
        )

        sim_trim, obs_trim = _warmup_slice(sim, obs, warm_up)
        metric_value, loss = _evaluate_objective(objective_name, sim_trim, obs_trim; log_transform = log_transform)
        penalty = constraint_penalty(candidate)
        return loss + penalty * 1e3, metric_value, penalty
    end
end

function _run_optimizer(
    objective_loss::Function,
    algorithm::String,
    lb::Vector{Float64},
    ub::Vector{Float64};
    maxiters::Int,
    seed::Union{Nothing,Int} = nothing,
    x0::Union{Nothing,Vector{Float64}} = nothing,
)
    if isempty(lb)
        f0 = objective_loss(Float64[])
        return Dict{String,Any}(
            "x_best" => Float64[],
            "f_best" => f0,
            "history" => Float64[f0],
            "evaluations" => 1,
            "iterations" => 0,
        )
    end

    optimizer = OptimizationStrategies.resolve_optimizer(algorithm)

    n_params = length(lb)
    initial = isnothing(x0) ? [lb[i] + rand() * (ub[i] - lb[i]) for i in 1:n_params] : copy(x0)

    function wrapped(u, _)
        return objective_loss(collect(u))
    end

    optf = Optimization.OptimizationFunction(wrapped)
    prob = Optimization.OptimizationProblem(optf, initial, nothing; lb = lb, ub = ub)
    sol = Optimization.solve(prob, optimizer; maxiters = maxiters)

    return Dict{String,Any}(
        "x_best" => Float64.(collect(sol.u)),
        "f_best" => Float64(sol.objective),
        "history" => Float64[],
        "evaluations" => maxiters,
        "iterations" => maxiters,
    )
end

# ==============================================================================
# 辅助：加载模型并提取参数信息
# ==============================================================================

function _load_model_and_bounds(model_name::String)
    all_models = String.(HydroModelLibrary.AVAILABLE_MODELS)
    idx = findfirst(m -> lowercase(m) == lowercase(strip(model_name)), all_models)
    isnothing(idx) && throw(ArgumentError("模型 '$(model_name)' 未找到"))
    canonical = all_models[idx]

    model_module = HydroModelLibrary.load_model(Symbol(canonical), reload=false)
    model = Base.invokelatest(m -> m.model, model_module)
    wrapper_params = Base.invokelatest(m -> m.model_parameters, model_module)

    param_names = Symbol.(string.(HydroModels.get_param_names(model)))
    bounds_by_name = Dict{Symbol,Tuple{Float64,Float64}}()
    for p in wrapper_params
        pname = Symbol(string(p))
        b = try
            raw = HydroModels.getbounds(p)
            (Float64(raw[1]), Float64(raw[2]))
        catch
            (0.0, 10.0)
        end
        bounds_by_name[pname] = b
    end

    bounds_list = Tuple{Float64,Float64}[]
    for pname in param_names
        if haskey(bounds_by_name, pname)
            push!(bounds_list, bounds_by_name[pname])
        else
            push!(bounds_list, (0.0, 10.0))
        end
    end

    return model, canonical, param_names, bounds_list
end

function _init_states_for(model)
    snames = HydroModels.get_state_names(model)
    ComponentVector(NamedTuple{Tuple(snames)}(zeros(length(snames))))
end

function _resolve_solver_symbol(solver_type::String)
    normalized = uppercase(strip(solver_type))
    normalized in ("ODE", "DISCRETE", "MUTABLE", "IMMUTABLE") || (normalized = "DISCRETE")
    return Symbol(normalized)
end

function _resolve_interpolation_symbol(interp_type::String)
    normalized = uppercase(strip(interp_type))
    normalized in ("LINEAR", "CONSTANT", "DIRECT") || (normalized = "LINEAR")
    return Symbol(normalized)
end

function _warmup_slice(sim::AbstractVector, obs::AbstractVector, warm_up::Int)
    n = min(length(sim), length(obs))
    n > 0 || throw(ArgumentError("Simulation and observation are empty after alignment"))
    start_idx = clamp(warm_up, 1, n)
    if n >= 2 && (n - start_idx + 1) < 2
        start_idx = n - 1
    end
    return sim[start_idx:n], obs[start_idx:n]
end

# ==============================================================================
# 单目标校准 (Strategy 8)
# ==============================================================================

"""
    calibrate_model(model_name, forcing_nt, obs; kwargs...) -> Dict

执行单目标参数自动校准。使用 HydroModelsOptimizationExt 扩展。

返回 Dict 包含:
- best_params: 最优参数
- best_objective: 最优目标函数值
- convergence_trace: 收敛轨迹
- all_trials: 多次试验结果 (n_trials > 1 时)
"""
function calibrate_model(model_name::String, forcing_nt::NamedTuple,
                         obs::AbstractVector;
                         algorithm::String="BBO",
                          maxiters::Int=1000,
                          objective::String="KGE",
                          log_transform::Bool=false,
                          fixed_params::Dict{String,Float64}=Dict{String,Float64}(),
                          param_bounds::Union{Nothing,Dict{String,Vector{Float64}}}=nothing,
                          constraints=nothing,
                          n_trials::Int=1,
                          solver_type::String="DISCRETE",
                          interp_type::String="LINEAR",
                          init_states_dict::Union{Nothing,Dict{String,Float64}}=nothing,
                          warm_up::Int=1,
                          budget::String="medium",
                          seed::Union{Nothing,Int}=nothing,
                          log_transform_mode::String="auto")

    model, canonical, all_param_names, default_bounds = _load_model_and_bounds(model_name)

    # --- 分离 calibratable vs fixed 参数 ---
    cal_indices = Int[]
    cal_names = Symbol[]
    fixed_values_sym = Dict{Symbol,Float64}()

    for (i, pname) in enumerate(all_param_names)
        sname = string(pname)
        if haskey(fixed_params, sname)
            fixed_values_sym[pname] = fixed_params[sname]
        else
            push!(cal_indices, i)
            push!(cal_names, pname)
        end
    end

    # --- 构建 bounds (仅针对可校准参数) ---
    lb = Float64[]
    ub = Float64[]
    for i in cal_indices
        pname = string(all_param_names[i])
        if !isnothing(param_bounds) && haskey(param_bounds, pname)
            push!(lb, Float64(param_bounds[pname][1]))
            push!(ub, Float64(param_bounds[pname][2]))
        else
            push!(lb, default_bounds[i][1])
            push!(ub, default_bounds[i][2])
        end
        lb[end] <= ub[end] || throw(ArgumentError("Invalid bounds for parameter '$pname': lower > upper"))
    end

    # --- 初始状态 ---
    states = if !isnothing(init_states_dict)
        snames = HydroModels.get_state_names(model)
        vals = [get(init_states_dict, string(s), 0.0) for s in snames]
        ComponentVector(NamedTuple{Tuple(snames)}(vals))
    else
        _init_states_for(model)
    end

    input_names = HydroModels.get_input_names(model)
    missing_vars = setdiff(input_names, keys(forcing_nt))
    if !isempty(missing_vars)
        throw(ArgumentError("输入数据缺失变量: $missing_vars"))
    end
    solver_sym = _resolve_solver_symbol(solver_type)
    interp_sym = _resolve_interpolation_symbol(interp_type)
    config_dict = Dict{String,Any}()

    requested_algorithm = OptimizationStrategies.normalize_algorithm_name(algorithm)
    chosen_algorithm = requested_algorithm == "AUTO" ?
        OptimizationStrategies.recommend_algorithm(budget = budget, n_parameters = length(cal_names)) :
        requested_algorithm
    chosen_backend = OptimizationStrategies.resolve_backend_algorithm(chosen_algorithm)

    auto_log_enabled, auto_log_report = _infer_log_transform(obs, objective, log_transform)
    effective_log_transform = lowercase(strip(log_transform_mode)) == "manual" ? log_transform : auto_log_enabled

    constraint_penalty, constraint_labels = _build_constraint_functions(
        all_param_names,
        fixed_values_sym,
        cal_names,
        constraints,
    )

    objective_eval = _calibration_objective(
        model,
        forcing_nt,
        obs,
        all_param_names,
        cal_names,
        fixed_values_sym,
        states,
        solver_sym,
        interp_sym,
        config_dict,
        objective,
        constraint_penalty;
        log_transform = effective_log_transform,
        warm_up = warm_up,
    )

    # --- 多次独立试验 ---
    all_trials = Dict{String,Any}[]
    best_result = nothing
    best_loss = Inf

    for trial in 1:n_trials
        x0_samples = Sampling.generate_samples(
            collect(zip(lb, ub)); method="lhs", n_samples=1
        )
        x0 = Float64.(x0_samples[:, 1])

        initial_loss, initial_metric, initial_penalty = objective_eval(x0)
        objective_loss = candidate -> first(objective_eval(candidate))

        trial_seed = isnothing(seed) ? nothing : seed + trial - 1
        opt = _run_optimizer(
            objective_loss,
            chosen_algorithm,
            lb,
            ub;
            maxiters = maxiters,
            seed = trial_seed,
            x0 = x0,
        )

        best_candidate = Float64.(opt["x_best"])
        final_loss, final_metric, final_penalty = objective_eval(best_candidate)

        history_loss = if isempty(opt["history"])
            Float64[initial_loss, final_loss]
        else
            Float64.(opt["history"])
        end
        history_metric = [_score_to_metric(objective, -v) for v in history_loss]
        history_metric[end] = final_metric

        materialize = _build_full_params_template(all_param_names, cal_names, fixed_values_sym)
        best_full = materialize(best_candidate)
        sim_best = _execute_core(
            model,
            forcing_nt,
            _params_to_component_vector(all_param_names, best_full),
            states,
            solver_sym,
            interp_sym,
            config_dict,
        )
        sim_trim, obs_trim = _warmup_slice(sim_best, obs, warm_up)
        aux_metrics = Metrics.compute_metrics(sim_trim, obs_trim, ["NSE", "KGE", "LogNSE", "LogKGE", "RMSE", "PBIAS", "R2"])

        trial_params = Dict{String,Float64}()
        for (j, name) in enumerate(cal_names)
            trial_params[string(name)] = best_candidate[j]
        end

        total_improvement = if get(OBJECTIVE_DIRECTION, objective, :max) == :max
            final_metric - initial_metric
        elseif get(OBJECTIVE_DIRECTION, objective, :max) == :min
            initial_metric - final_metric
        else
            abs(initial_metric) - abs(final_metric)
        end
        history_window = min(length(history_metric), 12)
        tail = history_metric[end-history_window+1:end]
        plateau_detected = history_window >= 3 && (maximum(tail) - minimum(tail)) <= max(1e-6, abs(tail[end]) * 0.005)
        still_improving = history_window >= 3 && !plateau_detected && abs(tail[end] - tail[1]) > max(1e-6, abs(tail[1]) * 0.01)

        trial_result = Dict{String,Any}(
            "trial" => trial,
            "objective_value" => final_metric,
            "raw_minimized" => final_loss,
            "constraint_penalty" => final_penalty,
            "objective_history" => history_metric,
            "objective_history_loss" => history_loss,
            "metric_snapshot" => aux_metrics,
            "algorithm_used" => chosen_algorithm,
            "algorithm_backend" => chosen_backend,
            "params" => trial_params,
            "objective_summary" => Dict{String,Any}(
                "initial_value" => initial_metric,
                "final_value" => final_metric,
                "best_value" => final_metric,
                "n_iterations" => get(opt, "iterations", maxiters),
                "evaluations" => get(opt, "evaluations", maxiters),
                "total_improvement" => total_improvement,
                "improvement_rate" => history_window >= 2 ? total_improvement / max(history_window - 1, 1) : 0.0,
                "plateau_detected" => plateau_detected,
                "still_improving" => still_improving,
                "last_100_improvement" => history_window >= 2 ? (tail[end] - tail[1]) : 0.0,
                "initial_penalty" => initial_penalty,
                "final_penalty" => final_penalty,
            )
        )
        push!(all_trials, trial_result)

        if final_loss < best_loss
            best_loss = final_loss
            best_result = trial_result
        end
    end

    # --- 组装最终结果 ---
    best_params_full = Dict{String,Float64}()
    for (pname, val) in fixed_values_sym
        best_params_full[string(pname)] = val
    end
    for (k, v) in best_result["params"]
        best_params_full[k] = v
    end

    return Dict{String,Any}(
        "model_name"       => canonical,
        "best_params"      => best_params_full,
        "calibrated_params"=> best_result["params"],
        "fixed_params"     => Dict(string(k) => v for (k, v) in fixed_values_sym),
        "best_objective"   => best_result["objective_value"],
        "objective_name"   => objective,
        "algorithm"        => chosen_algorithm,
        "algorithm_backend" => chosen_backend,
        "requested_algorithm" => requested_algorithm,
        "algorithm_strategy8_note" => "DDS/SCE are strategy aliases mapped to stable library backends (DDS->DE, SCE->PSO)",
        "maxiters"         => maxiters,
        "n_trials"         => n_trials,
        "budget"           => budget,
        "log_transform"    => effective_log_transform,
        "log_transform_report" => auto_log_report,
        "constraints_applied" => constraints !== nothing,
        "constraint_labels" => constraint_labels,
        "all_trials"       => all_trials,
        "convergence_trace" => get(best_result, "objective_history", Float64[]),
        "param_bounds"     => Dict(string(cal_names[i]) => [lb[i], ub[i]] for i in eachindex(cal_names)),
    )
end

# ==============================================================================
# 多目标校准 (Strategy 9)
# ==============================================================================

"""
    calibrate_multiobjective(model_name, forcing_nt, obs; kwargs...) -> Dict

执行多目标参数校准，生成 Pareto 前沿。
"""
function calibrate_multiobjective(model_name::String, forcing_nt::NamedTuple,
                                  obs::AbstractVector;
                                  objectives::Vector{String}=["KGE", "LogKGE"],
                                  algorithm::String="NSGA2",
                                  maxiters::Int=30,
                                  population_size::Int=16,
                                  fixed_params::Dict{String,Float64}=Dict{String,Float64}(),
                                  param_bounds::Union{Nothing,Dict{String,Vector{Float64}}}=nothing,
                                  solver_type::String="DISCRETE",
                                  interp_type::String="LINEAR",
                                  init_states_dict::Union{Nothing,Dict{String,Float64}}=nothing)

    model, canonical, all_param_names, default_bounds = _load_model_and_bounds(model_name)

    # 分离参数
    cal_indices = Int[]
    cal_names = Symbol[]
    fixed_values = Dict{Symbol,Float64}()
    for (i, pname) in enumerate(all_param_names)
        sname = string(pname)
        if haskey(fixed_params, sname)
            fixed_values[pname] = fixed_params[sname]
        else
            push!(cal_indices, i)
            push!(cal_names, pname)
        end
    end

    lb = Float64[]
    ub = Float64[]
    for i in cal_indices
        pname = string(all_param_names[i])
        if !isnothing(param_bounds) && haskey(param_bounds, pname)
            push!(lb, param_bounds[pname][1])
            push!(ub, param_bounds[pname][2])
        else
            push!(lb, default_bounds[i][1])
            push!(ub, default_bounds[i][2])
        end
    end

    states = if !isnothing(init_states_dict)
        snames = HydroModels.get_state_names(model)
        vals = [get(init_states_dict, string(s), 0.0) for s in snames]
        ComponentVector(NamedTuple{Tuple(snames)}(vals))
    else
        _init_states_for(model)
    end

    # 准备输入矩阵（与单目标类似）
    input_names = HydroModels.get_input_names(model)
    missing_vars = setdiff(input_names, keys(forcing_nt))
    if !isempty(missing_vars)
        throw(ArgumentError("输入数据缺失变量: $missing_vars"))
    end
    input_matrix = stack([Float64.(forcing_nt[n]) for n in input_names], dims=1)

    solver_val = if uppercase(solver_type) == "DISCRETE"
        HydroModels.DiscreteSolver
    elseif uppercase(solver_type) == "MUTABLE"
        HydroModels.MutableSolver
    elseif uppercase(solver_type) == "IMMUTABLE"
        HydroModels.ImmutableSolver
    else
        HydroModels.MutableSolver
    end

    interp_val = if uppercase(interp_type) == "LINEAR"
        Val(DataInterpolations.LinearInterpolation)
    elseif uppercase(interp_type) == "CONSTANT"
        Val(DataInterpolations.ConstantInterpolation)
    else
        Val(DataInterpolations.LinearInterpolation)
    end

    # 多目标函数: 返回 (f, g, h) 三元组
    # f: 目标函数向量, g: 不等式约束, h: 等式约束
    function multi_obj_fn(θ)
        try
            full_params = zeros(length(all_param_names))
            ci = 1
            for (i, pname) in enumerate(all_param_names)
                if haskey(fixed_values, pname)
                    full_params[i] = fixed_values[pname]
                else
                    full_params[i] = θ[ci]
                    ci += 1
                end
            end

            pv = ComponentVector(; params=ComponentVector(
                NamedTuple{Tuple(all_param_names)}(full_params)
            ))
            # 使用扩展接口运行模型
            output = model(input_matrix, pv;
                          timeidx=collect(1:size(input_matrix, 2)),
                          initstates=states)
            sim = Float64.(output[end, :])
            n = min(length(sim), length(obs))

            vals = Float64[]
            for obj_name in objectives
                mv = Metrics.get_metric_value(sim[1:n], obs[1:n], obj_name)
                # 统一最小化：越大越好的取负
                push!(vals, Metrics.is_higher_better(obj_name) ? -mv : mv)
            end
            # 返回 (目标函数, 不等式约束, 等式约束)
            # 无约束问题：g 和 h 为空数组
            return (vals, Float64[], Float64[])
        catch
            return (fill(Inf, length(objectives)), Float64[], Float64[])
        end
    end

    # 直接使用 Metaheuristics 包，避免 OptimizationMetaheuristics 的兼容性问题
    bounds = Matrix{Float64}(undef, length(lb), 2)
    for i in eachindex(lb)
        bounds[i, 1] = lb[i]
        bounds[i, 2] = ub[i]
    end

    # 选择算法
    mh_alg = if uppercase(algorithm) == "NSGA2"
        Metaheuristics.NSGA2(N=population_size)
    elseif uppercase(algorithm) == "NSGA3"
        Metaheuristics.NSGA3(N=population_size)
    else
        Metaheuristics.NSGA2(N=population_size)
    end

    # 设置迭代次数
    mh_alg.options.iterations = maxiters

    # 运行优化 - 直接传递函数和边界
    mh_result = Metaheuristics.optimize(multi_obj_fn, bounds, mh_alg)

    # 解析 Pareto 前沿
    pareto_front = Dict{String,Any}[]

    # 从 Metaheuristics 结果中提取 Pareto 前沿
    try
        # 获取最终种群
        if hasproperty(mh_result, :population)
            pop = mh_result.population
            for ind in pop
                params_dict = Dict(string(cal_names[j]) => ind.x[j] for j in eachindex(cal_names))
                obj_dict = Dict{String,Float64}()
                for (k, obj_name) in enumerate(objectives)
                    raw = ind.f[k]
                    obj_dict[obj_name] = Metrics.is_higher_better(obj_name) ? -raw : raw
                end
                push!(pareto_front, Dict("params" => params_dict, "objectives" => obj_dict))
            end
        elseif hasproperty(mh_result, :best_sol)
            # 单个最优解
            best = mh_result.best_sol
            params_dict = Dict(string(cal_names[j]) => best.x[j] for j in eachindex(cal_names))
            obj_dict = Dict{String,Float64}()
            for (k, obj_name) in enumerate(objectives)
                raw = best.f[k]
                obj_dict[obj_name] = Metrics.is_higher_better(obj_name) ? -raw : raw
            end
            push!(pareto_front, Dict("params" => params_dict, "objectives" => obj_dict))
        end
    catch e
        @warn "提取 Pareto 前沿时出错: $(e)，尝试备用方法"
        # Fallback: 使用 best_sol
        if hasproperty(mh_result, :best_sol)
            best = mh_result.best_sol
            params_dict = Dict(string(cal_names[j]) => best.x[j] for j in eachindex(cal_names))
            obj_dict = Dict{String,Float64}()
            for (k, obj_name) in enumerate(objectives)
                raw = best.f[k]
                obj_dict[obj_name] = Metrics.is_higher_better(obj_name) ? -raw : raw
            end
            push!(pareto_front, Dict("params" => params_dict, "objectives" => obj_dict))
        end
    end

    return Dict{String,Any}(
        "model_name"    => canonical,
        "objectives"    => objectives,
        "algorithm"     => algorithm,
        "maxiters"      => maxiters,
        "population_size" => population_size,
        "pareto_front"  => pareto_front,
        "n_solutions"   => length(pareto_front),
        "param_bounds"  => Dict(string(cal_names[i]) => [lb[i], ub[i]] for i in eachindex(cal_names)),
    )
end

# ==============================================================================
# 校准诊断 (Strategy 10)
# ==============================================================================

"""
    diagnose_calibration(result; kwargs...) -> Dict

诊断校准结果质量。执行 4 项检查:
1. 收敛性: 多次试验结果的一致性
2. 边界触达: 最优参数是否卡在边界上
3. 目标平台期: 优化是否已充分收敛
4. Hat-trick: 三项全通过
"""
function diagnose_calibration(result::Dict{String,Any};
                              boundary_tolerance::Float64=0.01,
                              convergence_threshold::Float64=0.05,
                              plateau_window::Int=50)

    diagnostics = Dict{String,Any}()
    recommendations = String[]

    # --- Check 1: 收敛性 (多次试验) ---
    all_trials = get(result, "all_trials", Dict{String,Any}[])
    if length(all_trials) > 1
        obj_values = [t["objective_value"] for t in all_trials]
        mean_obj = mean(obj_values)
        spread = std(obj_values)
        cv = abs(mean_obj) > 1e-10 ? spread / abs(mean_obj) : spread

        convergence_passed = cv < convergence_threshold
        diagnostics["convergence"] = Dict{String,Any}(
            "passed" => convergence_passed,
            "spread" => spread,
            "cv" => cv,
            "n_trials" => length(all_trials),
            "objective_values" => obj_values,
        )
        if !convergence_passed
            push!(recommendations, "INCREASE_BUDGET: 多次试验结果差异较大 (CV=$(round(cv, digits=4)))，建议增加 maxiters 或 n_trials")
        end
    else
        diagnostics["convergence"] = Dict{String,Any}(
            "passed" => true,
            "message" => "仅单次试验，无法评估收敛性。建议设置 n_trials >= 3",
        )
        push!(recommendations, "建议使用 n_trials >= 3 以评估收敛性")
    end

    # --- Check 2: 边界触达 ---
    param_bounds = get(result, "param_bounds", Dict{String,Any}())
    best_params = get(result, "calibrated_params", Dict{String,Any}())
    at_bound = String[]

    for (pname, val) in best_params
        if haskey(param_bounds, pname)
            bounds = param_bounds[pname]
            range_width = bounds[2] - bounds[1]
            tol = range_width * boundary_tolerance
            if val <= bounds[1] + tol || val >= bounds[2] - tol
                push!(at_bound, pname)
            end
        end
    end

    boundary_passed = isempty(at_bound)
    diagnostics["boundaries"] = Dict{String,Any}(
        "passed" => boundary_passed,
        "at_bound" => at_bound,
    )
    if !boundary_passed
        push!(recommendations, "WIDEN_RANGE: 参数 $(join(at_bound, ", ")) 触达边界，建议扩大其取值范围")
    end

    boundary_adjustments = Dict{String,Any}()
    for pname in at_bound
        if haskey(param_bounds, pname)
            bounds = param_bounds[pname]
            lo = Float64(bounds[1])
            hi = Float64(bounds[2])
            width = hi - lo
            margin = max(width * 0.2, 1e-6)
            boundary_adjustments[pname] = Dict(
                "current_bounds" => [lo, hi],
                "suggested_bounds" => [lo - margin, hi + margin],
                "rule" => "expand_by_20_percent",
            )
        end
    end
    diagnostics["boundary_adjustments"] = boundary_adjustments

    # --- Check 3: 目标平台期 ---
    plateau_passed = true
    if length(all_trials) >= 2
        sorted_objs = sort([t["objective_value"] for t in all_trials], rev=true)
        improvement = abs(sorted_objs[1] - sorted_objs[end])
        plateau_passed = improvement < abs(sorted_objs[1]) * 0.01
        diagnostics["plateau"] = Dict{String,Any}(
            "passed" => plateau_passed,
            "best_worst_gap" => improvement,
        )
        if !plateau_passed
            push!(recommendations, "INCREASE_BUDGET: 目标函数仍有明显改善空间，建议增加迭代次数")
        end
    else
        diagnostics["plateau"] = Dict{String,Any}(
            "passed" => true,
            "message" => "单次试验，无法判断平台期",
        )
    end

    # --- Check 3.5: 参数 Spread (Identifiability) ---
    unidentifiable = String[]
    if length(all_trials) > 1
        param_names_set = keys(all_trials[1]["params"])
        param_spread_info = Dict{String,Any}()
        for pname in param_names_set
            vals = [t["params"][pname] for t in all_trials]
            pstd = std(vals)
            pmean = mean(vals)
            pcv = abs(pmean) > 1e-10 ? pstd / abs(pmean) : pstd
            param_spread_info[pname] = Dict("std" => pstd, "cv" => pcv)
            if pcv > 0.5  # CV > 50% 视为不可识别
                push!(unidentifiable, pname)
            end
        end
        diagnostics["parameter_spread"] = Dict{String,Any}(
            "details" => param_spread_info,
            "unidentifiable" => unidentifiable,
        )
        if !isempty(unidentifiable)
            push!(recommendations, "UNIDENTIFIABLE: 参数 $(join(unidentifiable, ", ")) 在多次试验中变异较大，可能不可识别，建议固定或缩小范围")
        end
    end

    # --- Check 4: Hat-trick ---
    conv_ok = get(get(diagnostics, "convergence", Dict()), "passed", false)
    bound_ok = get(get(diagnostics, "boundaries", Dict()), "passed", false)
    plat_ok = get(get(diagnostics, "plateau", Dict()), "passed", false)
    hat_trick = conv_ok && bound_ok && plat_ok

    diagnostics["hat_trick"] = hat_trick
    diagnostics["recommendations"] = recommendations

    if hat_trick
        diagnostics["summary"] = "校准质量良好：收敛一致、参数未触边界、目标函数已收敛。"
    else
        diagnostics["summary"] = "校准存在问题，请参考 recommendations 进行调整。"
    end

    return diagnostics
end

# ==============================================================================
# 多目标校准诊断 (Strategy 9 + 10: Pareto 退化检测)
# ==============================================================================

"""
    diagnose_multiobjective(result) -> Dict

诊断多目标校准结果。检测 Pareto 前沿是否退化（简并为单点或直线）。

退化判定:
- 单点退化: 所有解在目标空间中几乎重合
- 直线退化: 所有解在目标空间中近似共线（目标不冲突）
"""
function diagnose_multiobjective(result::Dict{String,Any})
    diagnostics = Dict{String,Any}()
    recommendations = String[]

    pareto_front = get(result, "pareto_front", Dict{String,Any}[])
    objectives = get(result, "objectives", String[])
    n_solutions = length(pareto_front)

    diagnostics["n_solutions"] = n_solutions

    if n_solutions < 2
        diagnostics["degeneracy"] = Dict{String,Any}(
            "status" => "insufficient_solutions",
            "message" => "Pareto 前沿解数量不足 ($(n_solutions))，无法判断退化",
        )
        push!(recommendations, "增加种群大小或迭代次数以获得更多 Pareto 解")
        diagnostics["recommendations"] = recommendations
        return diagnostics
    end

    # 提取目标值矩阵
    obj_matrix = zeros(n_solutions, length(objectives))
    for (i, sol) in enumerate(pareto_front)
        objs = sol["objectives"]
        for (j, oname) in enumerate(objectives)
            obj_matrix[i, j] = get(objs, oname, NaN)
        end
    end

    # Check 1: 单点退化 — 所有解几乎重合
    obj_ranges = [maximum(obj_matrix[:, j]) - minimum(obj_matrix[:, j]) for j in 1:length(objectives)]
    obj_means = [abs(mean(obj_matrix[:, j])) for j in 1:length(objectives)]
    relative_ranges = [obj_means[j] > 1e-10 ? obj_ranges[j] / obj_means[j] : obj_ranges[j]
                       for j in 1:length(objectives)]

    point_degenerate = all(r < 0.01 for r in relative_ranges)

    # Check 2: 直线退化 — 目标之间高度相关（不冲突）
    line_degenerate = false
    if length(objectives) == 2 && n_solutions >= 3
        corr = cor(obj_matrix[:, 1], obj_matrix[:, 2])
        line_degenerate = abs(corr) > 0.95
        diagnostics["objective_correlation"] = corr
    end

    if point_degenerate
        diagnostics["degeneracy"] = Dict{String,Any}(
            "status" => "point_degenerate",
            "message" => "Pareto 前沿退化为单点，所有解几乎相同",
            "relative_ranges" => Dict(objectives[j] => relative_ranges[j] for j in eachindex(objectives)),
        )
        push!(recommendations, "目标可能不冲突，考虑使用单目标优化或更换目标组合")
    elseif line_degenerate
        diagnostics["degeneracy"] = Dict{String,Any}(
            "status" => "line_degenerate",
            "message" => "Pareto 前沿近似为直线，目标高度相关",
            "relative_ranges" => Dict(objectives[j] => relative_ranges[j] for j in eachindex(objectives)),
        )
        push!(recommendations, "两个目标高度相关，考虑替换其中一个以获得真正的权衡关系")
    else
        diagnostics["degeneracy"] = Dict{String,Any}(
            "status" => "normal",
            "message" => "Pareto 前沿形态正常",
            "relative_ranges" => Dict(objectives[j] => relative_ranges[j] for j in eachindex(objectives)),
        )
    end

    # 目标范围统计
    diagnostics["objective_ranges"] = Dict(
        objectives[j] => Dict("min" => minimum(obj_matrix[:, j]),
                              "max" => maximum(obj_matrix[:, j]),
                              "range" => obj_ranges[j])
        for j in eachindex(objectives)
    )

    diagnostics["recommendations"] = recommendations
    return diagnostics
end

# ==============================================================================
# 完整诊断分析 (Strategy 10 增强版)
# ==============================================================================

"""转换参数边界格式：从嵌套对象到数组"""
function _convert_bounds_format(parameter_bounds::Dict)
    bounds = Dict{String,Vector{Float64}}()
    for (pname, bound_dict) in parameter_bounds
        if bound_dict isa Dict
            bounds[pname] = [Float64(bound_dict["lower"]), Float64(bound_dict["upper"])]
        else
            bounds[pname] = Float64.(bound_dict)
        end
    end
    return bounds
end

"""平台期检测：分析目标函数历史"""
function _detect_plateau(objective_history::Vector{Float64}; window::Int=10, threshold::Float64=0.01)
    if length(objective_history) < window
        return false, 0.0
    end

    recent = objective_history[end-window+1:end]
    best_val = minimum(recent)
    worst_val = maximum(recent)
    improvement_rate = abs(worst_val - best_val) / (abs(worst_val) + 1e-10)

    return improvement_rate < threshold, improvement_rate
end

"""持续改进检测：比较最近和之前的改进"""
function _is_still_improving(objective_history::Vector{Float64}; window::Int=10, threshold::Float64=0.01)
    if length(objective_history) < 2 * window
        return true
    end

    recent = objective_history[end-window+1:end]
    previous = objective_history[end-2*window+1:end-window]

    recent_best = minimum(recent)
    previous_best = minimum(previous)

    improvement = abs(previous_best - recent_best) / (abs(previous_best) + 1e-10)
    return improvement > threshold
end

"""推荐动作决策逻辑"""
function _recommend_action(diagnostics::Dict{String,Any})
    if diagnostics["hat_trick_achieved"]
        return "converged", "校准已达到最佳状态：收敛一致、参数未触边界、目标函数已收敛"
    end

    if !isempty(diagnostics["parameters_at_boundary"])
        params_str = join(diagnostics["parameters_at_boundary"], ", ")
        return "widen_ranges", "参数 $params_str 触及边界，建议扩大其取值范围"
    end

    if diagnostics["convergence_status"] != "converged"
        cv = diagnostics["objective_spread"]
        return "increase_budget", "多次试验结果差异较大 (CV=$(round(cv, digits=4)))，建议增加 maxiters 或 n_trials"
    end

    if diagnostics["is_still_improving"]
        return "increase_budget", "目标函数仍在持续改进，建议增加迭代次数"
    end

    if diagnostics["is_plateaued"] && !diagnostics["hat_trick_achieved"]
        return "restart", "目标函数已停滞但未达到理想状态，建议重新开始或尝试其他算法"
    end

    return "converged", "校准质量可接受"
end

"""
    diagnose_calibration_full(trial_results, parameter_bounds; kwargs...) -> Dict

完整的校准诊断分析（Strategy 10增强版）。

# 参数
- `trial_results`: 试验结果列表，每个包含 best_objective, best_parameters, objective_history
- `parameter_bounds`: 参数边界字典 {"param": {"lower": x, "upper": y}}
- `objective_threshold`: 好拟合阈值（默认0.7，表示目标<0.3为好拟合）
- `boundary_tolerance`: 边界容差（默认0.01）
- `convergence_threshold`: 收敛阈值（默认0.1）
- `plateau_window`: 平台期检测窗口（默认10）

# 返回
包含以下字段的字典：
- convergence_status: "converged", "improving", "plateaued"
- objective_spread: 目标函数变异系数
- parameters_at_boundary: 触及边界的参数列表
- is_plateaued: 是否进入平台期
- is_still_improving: 是否仍在改进
- hat_trick_achieved: 是否达成Hat-trick
- recommended_action: 推荐操作
- reasoning: 推理解释
- statistics: 统计信息
"""
function diagnose_calibration_full(trial_results::Vector,
                                   parameter_bounds::Dict;
                                   objective_threshold::Float64=0.7,
                                   boundary_tolerance::Float64=0.01,
                                   convergence_threshold::Float64=0.1,
                                   plateau_window::Int=10)

    if isempty(trial_results)
        throw(ArgumentError("trial_results 不能为空"))
    end

    diagnostics = Dict{String,Any}()
    diagnostics["history_source"] = "observed"
    diagnostics["diagnostic_confidence"] = "high"

    # 转换参数边界格式
    bounds = _convert_bounds_format(parameter_bounds)

    # 1. 统计计算
    objectives = [Float64(t["best_objective"]) for t in trial_results]
    mean_obj = mean(objectives)
    std_obj = length(objectives) > 1 ? std(objectives) : 0.0
    cv_obj = abs(mean_obj) > 1e-10 ? std_obj / abs(mean_obj) : std_obj

    diagnostics["statistics"] = Dict{String,Any}(
        "mean_objective" => mean_obj,
        "std_objective" => std_obj,
        "cv_objective" => cv_obj,
        "min_objective" => minimum(objectives),
        "max_objective" => maximum(objectives),
        "n_trials" => length(objectives)
    )

    # 2. 收敛性分析
    if length(trial_results) > 1
        if cv_obj < convergence_threshold
            diagnostics["convergence_status"] = "converged"
        else
            diagnostics["convergence_status"] = "improving"
        end
    else
        diagnostics["convergence_status"] = "single_trial"
    end

    diagnostics["objective_spread"] = cv_obj

    # 3. 参数边界检查
    params_at_boundary = String[]
    for trial in trial_results
        best_params = trial["best_parameters"]
        for (pname, value) in best_params
            if haskey(bounds, pname)
                bound = bounds[pname]
                lower, upper = bound[1], bound[2]
                range_size = upper - lower
                tol = boundary_tolerance * range_size

                if abs(Float64(value) - lower) < tol || abs(Float64(value) - upper) < tol
                    if !(pname in params_at_boundary)
                        push!(params_at_boundary, pname)
                    end
                end
            end
        end
    end
    diagnostics["parameters_at_boundary"] = params_at_boundary

    # 4. 平台期和持续改进检测
    is_plateaued = false
    is_still_improving = false

    for trial in trial_results
        if haskey(trial, "objective_history")
            history = Float64.(trial["objective_history"])
            if !isempty(history)
                plateaued, _ = _detect_plateau(history; window=plateau_window)
                improving = _is_still_improving(history; window=plateau_window)

                if plateaued
                    is_plateaued = true
                end
                if improving
                    is_still_improving = true
                end
            end
        end
    end

    diagnostics["is_plateaued"] = is_plateaued
    diagnostics["is_still_improving"] = is_still_improving

    # 5. Hat-trick检查
    model_fit_good = mean_obj < (1.0 - objective_threshold)
    objectives_consistent = cv_obj < convergence_threshold
    no_boundary_hit = isempty(params_at_boundary)
    converged = is_plateaued && !is_still_improving

    hat_trick = model_fit_good && objectives_consistent && no_boundary_hit && converged
    diagnostics["hat_trick_achieved"] = hat_trick

    # 6. 推荐动作
    action, reasoning = _recommend_action(diagnostics)
    diagnostics["recommended_action"] = action
    diagnostics["reasoning"] = reasoning

    return diagnostics
end

end # module
