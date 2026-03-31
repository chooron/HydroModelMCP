module Calibration

using ..Optimization
using ..OptimizationBBO
using ..OptimizationMetaheuristics
using ..Metaheuristics
using ..HydroModels
using ..HydroModelLibrary
using ..ComponentArrays
using ..DataInterpolations
using ..Statistics
using ..Metrics
using ..Sampling
using ..Simulation: _execute_core

export calibrate_model, calibrate_multiobjective, diagnose_calibration, diagnose_multiobjective, diagnose_calibration_full

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

# ==============================================================================
# 选择优化算法 (Strategy 8)
# ==============================================================================

function _resolve_algorithm(name::String)
    alg_map = Dict(
        "BBO"        => BBO_adaptive_de_rand_1_bin_radiuslimited(),
        "DE"         => BBO_de_rand_1_bin(),
        "PSO"        => OptimizationMetaheuristics.PSO(),
        "CMAES"      => OptimizationMetaheuristics.CGSA(),
        "ECA"        => OptimizationMetaheuristics.ECA(),
    )
    haskey(alg_map, uppercase(name)) || throw(ArgumentError("未知算法: $(name)。可选: BBO, DE, PSO, CMAES, ECA"))
    return alg_map[uppercase(name)]
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
                         n_trials::Int=1,
                         solver_type::String="DISCRETE",
                         interp_type::String="LINEAR",
                         init_states_dict::Union{Nothing,Dict{String,Float64}}=nothing,
                         warm_up::Int=1)

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
            push!(lb, param_bounds[pname][1])
            push!(ub, param_bounds[pname][2])
        else
            push!(lb, default_bounds[i][1])
            push!(ub, default_bounds[i][2])
        end
    end

    # --- 初始状态 ---
    states = if !isnothing(init_states_dict)
        snames = HydroModels.get_state_names(model)
        vals = [get(init_states_dict, string(s), 0.0) for s in snames]
        ComponentVector(NamedTuple{Tuple(snames)}(vals))
    else
        _init_states_for(model)
    end

    # --- 准备输入矩阵 ---
    input_names = HydroModels.get_input_names(model)
    missing_vars = setdiff(input_names, keys(forcing_nt))
    if !isempty(missing_vars)
        throw(ArgumentError("输入数据缺失变量: $missing_vars"))
    end
    input_matrix = stack([Float64.(forcing_nt[n]) for n in input_names], dims=1)

    # --- 求解器和插值器 ---
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

    # --- 优化算法 ---
    alg = _resolve_algorithm(algorithm)
    higher_better = Metrics.is_higher_better(objective)

    # --- 多次独立试验 ---
    all_trials = Dict{String,Any}[]
    best_result = nothing
    best_obj = Inf

    for trial in 1:n_trials
        # LHS 初始点
        x0_samples = Sampling.generate_samples(
            collect(zip(lb, ub)); method="lhs", n_samples=1
        )
        x0 = x0_samples[:, 1]

        # 将 fixed_params 包装为 ComponentVector 格式（扩展期望的格式）
        fixed_params_cv = if !isempty(fixed_values_sym)
            ComponentVector(params=NamedTuple{Tuple(keys(fixed_values_sym))}(values(fixed_values_sym)))
        else
            nothing
        end

        # 使用 OptimizationProblem 扩展
        # 注意：扩展会使用 default_pas 作为初始点，但我们通过 x0 覆盖它
        prob = OptimizationProblem(
            model, input_matrix, obs;
            metric=objective,
            warm_up=warm_up,
            fixed_params=fixed_params_cv,
            lb_pas=lb,
            ub_pas=ub,
            default_initstates=states,
            solver=solver_val,
            interpolator=interp_val
        )

        # 重建问题以使用 LHS 初始点
        prob = Optimization.remake(prob, u0=x0)

        # 求解
        sol = solve(prob, alg; maxiters=maxiters)

        obj_val = sol.objective
        trial_params = Dict{String,Float64}()
        for (j, name) in enumerate(cal_names)
            trial_params[string(name)] = sol.u[j]
        end

        trial_result = Dict{String,Any}(
            "trial" => trial,
            "objective_value" => higher_better ? -obj_val : obj_val,
            "raw_minimized" => obj_val,
            "params" => trial_params,
            "objective_summary" => Dict{String,Any}(
                "initial_value" => 1.0,
                "final_value" => higher_better ? -obj_val : obj_val,
                "best_value" => higher_better ? -obj_val : obj_val,
                "n_iterations" => maxiters,
                "total_improvement" => 0.0,
                "improvement_rate" => 0.0,
                "plateau_detected" => false,
                "still_improving" => false,
                "last_100_improvement" => 0.0
            )
        )
        push!(all_trials, trial_result)

        if obj_val < best_obj
            best_obj = obj_val
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
        "algorithm"        => algorithm,
        "maxiters"         => maxiters,
        "n_trials"         => n_trials,
        "all_trials"       => all_trials,
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
                                  maxiters::Int=1000,
                                  population_size::Int=50,
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

    # 转换参数边界格式
    bounds = _convert_bounds_format(parameter_bounds)

    # 1. 统计计算
    objectives = [Float64(t["best_objective"]) for t in trial_results]
    mean_obj = mean(objectives)
    std_obj = std(objectives)
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
