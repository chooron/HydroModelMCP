using ModelContextProtocol
using JSON3
using .Schemas
import UUIDs
import Dates

# ==============================================================================
# 1. 计算评价指标工具
# ==============================================================================
compute_metrics_tool = MCPTool(
    name = "compute_metrics",
    description = "计算模拟结果与观测数据之间的评价指标。支持 NSE, KGE, LogNSE, LogKGE, PBIAS, R2, RMSE。自动检测是否建议对数变换(Strategy 3)。",
    input_schema = Dict{String,Any}(
        "type" => "object",
        "properties" => Dict{String,Any}(
            "simulation" => DATA_SOURCE_SCHEMA,
            "observation" => OBSERVATION_SCHEMA,
            "sim_column" => SIM_COLUMN_SCHEMA,
            "obs_column" => OBS_COLUMN_SCHEMA,
            "metrics" => METRICS_SCHEMA
        ),
        "required" => ["simulation", "observation", "sim_column", "obs_column"]
    ),
    handler = function(params)
        # 验证必需参数
        validation_error = validate_required_params(params, ["simulation", "observation", "sim_column", "obs_column"])
        if !isnothing(validation_error)
            return create_error_response(validation_error)
        end

        try
            # 加载模拟和观测数据
            sim_config = params["simulation"]
            obs_config = params["observation"]
            sim_col = params["sim_column"]
            obs_col = params["obs_column"]

            sim_data, _ = DataLoader.load_data(Val(Symbol(sim_config["source_type"])), sim_config)
            obs_data, _ = DataLoader.load_data(Val(Symbol(obs_config["source_type"])), obs_config)

            sim_vec = Float64.(sim_data[Symbol(sim_col)])
            obs_vec = Float64.(obs_data[Symbol(obs_col)])

            metric_names = get(params, "metrics", String["NSE","KGE","LogNSE","LogKGE","PBIAS","R2","RMSE"])
            n = min(length(sim_vec), length(obs_vec))
            result = Metrics.compute_metrics(sim_vec[1:n], obs_vec[1:n], metric_names)

            return TextContent(text = JSON3.write(result))
        catch e
            return create_error_response(e)
        end
    end
)

# ==============================================================================
# 2. 数据划分工具
# ==============================================================================
split_data_tool = MCPTool(
    name = "split_data",
    description = "将时间序列数据划分为校准集和验证集。支持三种策略：recent_first(最新数据优先,适合预测部署)、split_sample(经典分割,适合过程理解)、use_all(全部校准,适合最终部署)。",
    input_schema = Dict{String,Any}(
        "type" => "object",
        "properties" => Dict{String,Any}(
            "data_source" => DATA_SOURCE_SCHEMA,
            "obs_column" => OBS_COLUMN_SCHEMA,
            "method" => SPLIT_METHOD_SCHEMA,
            "ratio" => RATIO_SCHEMA,
            "warmup" => WARMUP_SCHEMA
        ),
        "required" => ["data_source", "obs_column"]
    ),
    handler = function(params)
        # 验证必需参数
        validation_error = validate_required_params(params, ["data_source", "obs_column"])
        if !isnothing(validation_error)
            return create_error_response(validation_error)
        end

        # 验证枚举参数
        if haskey(params, "method")
            enum_error = validate_enum_param(params, "method",
                ["recent_first", "split_sample", "use_all"], "split_sample")
            if !isnothing(enum_error)
                return create_error_response(enum_error)
            end
        end

        try
            config = params["data_source"]
            obs_col = params["obs_column"]
            data, _ = DataLoader.load_data(Val(Symbol(config["source_type"])), config)
            obs_vec = Float64.(data[Symbol(obs_col)])

            method = get(params, "method", "split_sample")
            ratio = get(params, "ratio", 0.7)
            warmup = get(params, "warmup", 365)

            result = DataSplitter.split_data(obs_vec, data;
                                             method=method, ratio=Float64(ratio), warmup=Int(warmup))
            # 移除 NamedTuple (不可直接 JSON 序列化)，只保留元信息
            serializable = Dict{String,Any}(
                "method"      => result["method"],
                "warmup"      => result["warmup"],
                "total_length"=> result["total_length"],
                "cal_length"  => result["cal_length"],
                "val_length"  => result["val_length"],
                "cal_indices" => result["cal_indices"],
                "val_indices" => result["val_indices"],
            )
            return TextContent(text = JSON3.write(serializable))
        catch e
            return create_error_response(e)
        end
    end
)

# ==============================================================================
# 3. 敏感性分析工具
# ==============================================================================
sensitivity_tool = MCPTool(
    name = "run_sensitivity",
    description = "对水文模型参数进行全局敏感性分析(Strategy 1)。识别哪些参数对输出有显著影响(应校准)，哪些可固定为默认值。支持 Morris 和 Sobol 方法。",
    input_schema = Dict{String,Any}(
        "type" => "object",
        "properties" => Dict{String,Any}(
            "model_name" => MODEL_NAME_SCHEMA,
            "forcing" => FORCING_SCHEMA,
            "observation" => OBSERVATION_SCHEMA,
            "obs_column" => OBS_COLUMN_SCHEMA,
            "method" => SENSITIVITY_METHOD_SCHEMA,
            "n_samples" => N_SAMPLES_SCHEMA,
            "objective" => OBJECTIVE_SCHEMA,
            "threshold" => THRESHOLD_SCHEMA,
            "solver" => SOLVER_SIMPLE_SCHEMA,
            "interpolator" => INTERPOLATOR_SIMPLE_SCHEMA
        ),
        "required" => ["model_name", "forcing", "observation", "obs_column"]
    ),
    handler = function(params)
        # 验证必需参数
        validation_error = validate_required_params(params, ["model_name", "forcing", "observation", "obs_column"])
        if !isnothing(validation_error)
            return create_error_response(validation_error)
        end

        # 验证枚举参数
        if haskey(params, "method")
            enum_error = validate_enum_param(params, "method",
                ["morris", "sobol"], "morris")
            if !isnothing(enum_error)
                return create_error_response(enum_error)
            end
        end

        try
            forcing_config = params["forcing"]
            obs_config = params["observation"]
            forcing_nt, _ = DataLoader.load_data(Val(Symbol(forcing_config["source_type"])), forcing_config)
            obs_data, _ = DataLoader.load_data(Val(Symbol(obs_config["source_type"])), obs_config)
            obs_vec = Float64.(obs_data[Symbol(params["obs_column"])])

            method = get(params, "method", "morris")
            default_n = method == "sobol" ? 1000 : 100

            result = SensitivityAnalysis.run_sensitivity(
                params["model_name"], forcing_nt, obs_vec;
                method = method,
                n_samples = Int(get(params, "n_samples", default_n)),
                objective = get(params, "objective", "NSE"),
                threshold = Float64(get(params, "threshold", 0.1)),
                solver_type = get(params, "solver", "DISCRETE"),
                interp_type = get(params, "interpolator", "LINEAR")
            )
            return TextContent(text = JSON3.write(result))
        catch e
            return create_error_response(e)
        end
    end
)

# ==============================================================================
# 4. 参数采样工具
# ==============================================================================
sampling_tool = MCPTool(
    name = "generate_samples",
    description = "为水文模型生成参数样本集(Strategy 5)。支持 LHS(拉丁超立方)和 Sobol 序列，确保参数空间均匀覆盖。可用于参数探索或校准初始化。",
    input_schema = Dict{String,Any}(
        "type" => "object",
        "properties" => Dict{String,Any}(
            "model_name" => MODEL_NAME_SCHEMA,
            "n_samples" => SAMPLING_N_SAMPLES_SCHEMA,
            "method" => SAMPLING_METHOD_SCHEMA,
            "param_bounds" => PARAM_BOUNDS_SCHEMA
        ),
        "required" => ["model_name"]
    ),
    handler = function(params)
        # 验证必需参数
        validation_error = validate_required_params(params, ["model_name"])
        if !isnothing(validation_error)
            return create_error_response(validation_error)
        end

        # 验证枚举参数
        if haskey(params, "method")
            enum_error = validate_enum_param(params, "method",
                ["lhs", "sobol", "random"], "lhs")
            if !isnothing(enum_error)
                return create_error_response(enum_error)
            end
        end

        try
            model_name = params["model_name"]
            _, canonical, param_names, default_bounds = Calibration._load_model_and_bounds(model_name)

            custom_bounds = get(params, "param_bounds", nothing)
            bounds = Tuple{Float64,Float64}[]
            for (i, pname) in enumerate(param_names)
                sname = string(pname)
                if !isnothing(custom_bounds) && haskey(custom_bounds, sname)
                    b = custom_bounds[sname]
                    push!(bounds, (Float64(b[1]), Float64(b[2])))
                else
                    push!(bounds, default_bounds[i])
                end
            end

            n_samples = Int(get(params, "n_samples", 100))
            method = get(params, "method", "lhs")
            samples_matrix = Sampling.generate_samples(bounds; method=method, n_samples=n_samples)

            # 转为可序列化格式: [{param1: v1, param2: v2, ...}, ...]
            samples_list = Dict{String,Float64}[]
            for j in 1:n_samples
                d = Dict{String,Float64}()
                for (i, pname) in enumerate(param_names)
                    d[string(pname)] = samples_matrix[i, j]
                end
                push!(samples_list, d)
            end

            result = Dict{String,Any}(
                "model_name" => canonical,
                "method" => method,
                "n_samples" => n_samples,
                "param_names" => string.(param_names),
                "bounds" => Dict(string(param_names[i]) => [bounds[i][1], bounds[i][2]] for i in eachindex(param_names)),
                "samples" => samples_list
            )
            return TextContent(text = JSON3.write(result))
        catch e
            return create_error_response(e)
        end
    end
)

# ==============================================================================
# 5. 单目标校准工具
# ==============================================================================
calibrate_tool = MCPTool(
    name = "calibrate_model",
    description = "执行水文模型参数自动校准(单目标优化)。支持多种优化算法(BBO/DE/PSO等)，可指定固定参数、自定义范围。返回最优参数、目标函数值和收敛信息。",
    input_schema = Dict{String,Any}(
        "type" => "object",
        "properties" => Dict{String,Any}(
            "model_name" => MODEL_NAME_SCHEMA,
            "forcing" => FORCING_SCHEMA,
            "observation" => OBSERVATION_SCHEMA,
            "obs_column" => OBS_COLUMN_SCHEMA,
            "objective" => OBJECTIVE_SCHEMA,
            "algorithm" => ALGORITHM_SCHEMA,
            "maxiters" => MAXITERS_SCHEMA,
            "n_trials" => N_TRIALS_SCHEMA,
            "log_transform" => LOG_TRANSFORM_SCHEMA,
            "fixed_params" => FIXED_PARAMS_SCHEMA,
            "param_bounds" => PARAM_BOUNDS_SCHEMA,
            "solver" => SOLVER_SCHEMA,
            "interpolator" => INTERPOLATOR_SCHEMA,
            "init_states" => INIT_STATES_SCHEMA
        ),
        "required" => ["model_name", "forcing", "observation", "obs_column"]
    ),
    handler = function(params)
        # 验证必需参数
        validation_error = validate_required_params(params, ["model_name", "forcing", "observation", "obs_column"])
        if !isnothing(validation_error)
            return create_error_response(validation_error)
        end

        # 验证枚举参数
        if haskey(params, "objective")
            enum_error = validate_enum_param(params, "objective",
                ["KGE","NSE","LogNSE","LogKGE","PBIAS","R2","RMSE"], "KGE")
            if !isnothing(enum_error)
                return create_error_response(enum_error)
            end
        end

        if haskey(params, "algorithm")
            enum_error = validate_enum_param(params, "algorithm",
                ["BBO","DE","PSO","CMAES","ECA"], "BBO")
            if !isnothing(enum_error)
                return create_error_response(enum_error)
            end
        end

        try
            forcing_config = params["forcing"]
            obs_config = params["observation"]
            forcing_nt, _ = DataLoader.load_data(Val(Symbol(forcing_config["source_type"])), forcing_config)
            obs_data, _ = DataLoader.load_data(Val(Symbol(obs_config["source_type"])), obs_config)
            obs_vec = Float64.(obs_data[Symbol(params["obs_column"])])

            # 解析可选参数
            fixed = Dict{String,Float64}()
            if haskey(params, "fixed_params")
                for (k, v) in params["fixed_params"]
                    fixed[string(k)] = Float64(v)
                end
            end

            pb = nothing
            if haskey(params, "param_bounds")
                pb = Dict{String,Vector{Float64}}()
                for (k, v) in params["param_bounds"]
                    pb[string(k)] = Float64.(v)
                end
            end

            ist = nothing
            if haskey(params, "init_states")
                ist = Dict{String,Float64}()
                for (k, v) in params["init_states"]
                    ist[string(k)] = Float64(v)
                end
            end

            result = Calibration.calibrate_model(
                params["model_name"], forcing_nt, obs_vec;
                algorithm = get(params, "algorithm", "BBO"),
                maxiters = Int(get(params, "maxiters", 1000)),
                objective = get(params, "objective", "KGE"),
                log_transform = Bool(get(params, "log_transform", false)),
                fixed_params = fixed,
                param_bounds = pb,
                n_trials = Int(get(params, "n_trials", 1)),
                solver_type = get(params, "solver", "DISCRETE"),
                interp_type = get(params, "interpolator", "LINEAR"),
                init_states_dict = ist
            )
            return TextContent(text = JSON3.write(result))
        catch e
            return create_error_response(e)
        end
    end
)

# ==============================================================================
# 6. 多目标校准工具
# ==============================================================================
calibrate_multi_tool = MCPTool(
    name = "calibrate_multiobjective",
    description = "执行多目标参数校准(Strategy 9)，生成 Pareto 前沿。适用于需要同时优化多个指标(如高流 KGE + 低流 LogKGE)的场景。",
    input_schema = Dict{String,Any}(
        "type" => "object",
        "properties" => Dict{String,Any}(
            "model_name" => MODEL_NAME_SCHEMA,
            "forcing" => FORCING_SCHEMA,
            "observation" => OBSERVATION_SCHEMA,
            "obs_column" => OBS_COLUMN_SCHEMA,
            "objectives" => OBJECTIVES_SCHEMA,
            "algorithm" => MULTI_ALGORITHM_SCHEMA,
            "maxiters" => MAXITERS_SCHEMA,
            "population_size" => POPULATION_SIZE_SCHEMA,
            "fixed_params" => FIXED_PARAMS_SCHEMA,
            "param_bounds" => PARAM_BOUNDS_SCHEMA,
            "solver" => SOLVER_SCHEMA,
            "interpolator" => INTERPOLATOR_SCHEMA,
            "init_states" => INIT_STATES_SCHEMA
        ),
        "required" => ["model_name", "forcing", "observation", "obs_column", "objectives"]
    ),
    handler = function(params)
        # 验证必需参数
        validation_error = validate_required_params(params, ["model_name", "forcing", "observation", "obs_column", "objectives"])
        if !isnothing(validation_error)
            return create_error_response(validation_error)
        end

        # 验证枚举参数
        if haskey(params, "algorithm")
            enum_error = validate_enum_param(params, "algorithm",
                ["NSGA2", "NSGA3"], "NSGA2")
            if !isnothing(enum_error)
                return create_error_response(enum_error)
            end
        end

        try
            forcing_config = params["forcing"]
            obs_config = params["observation"]
            forcing_nt, _ = DataLoader.load_data(Val(Symbol(forcing_config["source_type"])), forcing_config)
            obs_data, _ = DataLoader.load_data(Val(Symbol(obs_config["source_type"])), obs_config)
            obs_vec = Float64.(obs_data[Symbol(params["obs_column"])])

            fixed = Dict{String,Float64}()
            if haskey(params, "fixed_params")
                for (k, v) in params["fixed_params"]
                    fixed[string(k)] = Float64(v)
                end
            end

            pb = nothing
            if haskey(params, "param_bounds")
                pb = Dict{String,Vector{Float64}}()
                for (k, v) in params["param_bounds"]
                    pb[string(k)] = Float64.(v)
                end
            end

            ist = nothing
            if haskey(params, "init_states")
                ist = Dict{String,Float64}()
                for (k, v) in params["init_states"]
                    ist[string(k)] = Float64(v)
                end
            end

            result = Calibration.calibrate_multiobjective(
                params["model_name"], forcing_nt, obs_vec;
                objectives = String.(params["objectives"]),
                algorithm = get(params, "algorithm", "NSGA2"),
                maxiters = Int(get(params, "maxiters", 1000)),
                population_size = Int(get(params, "population_size", 50)),
                fixed_params = fixed,
                param_bounds = pb,
                solver_type = get(params, "solver", "DISCRETE"),
                interp_type = get(params, "interpolator", "LINEAR"),
                init_states_dict = ist
            )
            return TextContent(text = JSON3.write(result))
        catch e
            return create_error_response(e)
        end
    end
)

# ==============================================================================
# 7. 校准诊断工具
# ==============================================================================
diagnose_tool = MCPTool(
    name = "diagnose_calibration",
    description = "诊断校准结果质量(Strategy 10)。检查收敛性、参数边界触达、目标函数平台期，并给出改进建议(增加预算/扩大范围等)。输入为 calibrate_model 的返回结果。",
    input_schema = Dict{String,Any}(
        "type" => "object",
        "properties" => Dict{String,Any}(
            "calibration_result" => CALIBRATION_RESULT_SCHEMA,
            "boundary_tolerance" => BOUNDARY_TOLERANCE_SCHEMA,
            "convergence_threshold" => CONVERGENCE_THRESHOLD_SCHEMA,
            "plateau_window" => PLATEAU_WINDOW_SCHEMA
        ),
        "required" => ["calibration_result"]
    ),
    handler = function(params)
        # 验证必需参数
        validation_error = validate_required_params(params, ["calibration_result"])
        if !isnothing(validation_error)
            return create_error_response(validation_error)
        end

        try
            cal_result = params["calibration_result"]
            # 确保是 Dict{String,Any}
            if !(cal_result isa Dict)
                cal_result = Dict{String,Any}(string(k) => v for (k, v) in cal_result)
            end

            result = Calibration.diagnose_calibration(
                cal_result;
                boundary_tolerance = Float64(get(params, "boundary_tolerance", 0.01)),
                convergence_threshold = Float64(get(params, "convergence_threshold", 0.05)),
                plateau_window = Int(get(params, "plateau_window", 50))
            )
            return TextContent(text = JSON3.write(result))
        catch e
            return create_error_response(e)
        end
    end
)

# ==============================================================================
# 8. 目标函数配置工具 (DEV.md Tool 2: configure_objectives)
# ==============================================================================
configure_objectives_tool = MCPTool(
    name = "configure_objectives",
    description = "根据业务目标自动推荐合适的评价指标组合。输入业务场景描述，返回推荐的目标函数、是否建议 log 变换、以及加权方案。对应论文 Strategy 7 的映射表。",
    input_schema = Dict{String,Any}(
        "type" => "object",
        "properties" => Dict{String,Any}(
            "goal" => Dict{String,Any}(
                "type" => "string",
                "enum" => ["general_fit", "peak_flows", "low_flows", "water_balance", "dynamics", "custom"],
                "description" => "业务目标: general_fit(综合拟合), peak_flows(洪峰), low_flows(枯水), water_balance(水量平衡), dynamics(动态过程), custom(自定义)"
            ),
            "custom_metrics" => METRICS_SCHEMA,
            "custom_weights" => Dict{String,Any}(
                "type" => "array",
                "items" => Dict{String,Any}("type" => "number"),
                "description" => "自定义权重列表 (仅 goal=custom 时使用，与 custom_metrics 对应)"
            )
        ),
        "required" => ["goal"]
    ),
    handler = function(params)
        # 验证必需参数
        validation_error = validate_required_params(params, ["goal"])
        if !isnothing(validation_error)
            return create_error_response(validation_error)
        end

        # 验证枚举参数
        enum_error = validate_enum_param(params, "goal",
            ["general_fit", "peak_flows", "low_flows", "water_balance", "dynamics", "custom"], nothing)
        if !isnothing(enum_error)
            return create_error_response(enum_error)
        end

        try
            goal = params["goal"]

            # Strategy 7 映射表
            config = if goal == "general_fit"
                Dict{String,Any}(
                    "primary_metric" => "KGE",
                    "secondary_metrics" => ["NSE", "PBIAS"],
                    "log_transform" => false,
                    "weights" => Dict("KGE" => 1.0),
                    "description" => "综合拟合：KGE 兼顾相关性、变异性和偏差"
                )
            elseif goal == "peak_flows"
                Dict{String,Any}(
                    "primary_metric" => "NSE",
                    "secondary_metrics" => ["KGE", "RMSE"],
                    "log_transform" => false,
                    "weights" => Dict("NSE" => 0.7, "KGE" => 0.3),
                    "description" => "洪峰模拟：NSE 对大值敏感，适合捕捉峰值"
                )
            elseif goal == "low_flows"
                Dict{String,Any}(
                    "primary_metric" => "LogKGE",
                    "secondary_metrics" => ["LogNSE", "PBIAS"],
                    "log_transform" => true,
                    "weights" => Dict("LogKGE" => 0.7, "PBIAS" => 0.3),
                    "description" => "枯水期：对数变换后的指标对低流量更敏感"
                )
            elseif goal == "water_balance"
                Dict{String,Any}(
                    "primary_metric" => "PBIAS",
                    "secondary_metrics" => ["KGE", "R2"],
                    "log_transform" => false,
                    "weights" => Dict("PBIAS" => 0.6, "KGE" => 0.4),
                    "description" => "水量平衡：PBIAS 直接衡量总量偏差"
                )
            elseif goal == "dynamics"
                Dict{String,Any}(
                    "primary_metric" => "R2",
                    "secondary_metrics" => ["KGE", "NSE"],
                    "log_transform" => false,
                    "weights" => Dict("R2" => 0.6, "KGE" => 0.4),
                    "description" => "动态过程：R² 衡量时间序列的相关性和模式匹配"
                )
            elseif goal == "custom"
                metrics = get(params, "custom_metrics", String["KGE"])
                weights_arr = get(params, "custom_weights", ones(length(metrics)))
                w_dict = Dict(metrics[i] => Float64(weights_arr[i]) for i in eachindex(metrics))
                Dict{String,Any}(
                    "primary_metric" => metrics[1],
                    "secondary_metrics" => length(metrics) > 1 ? metrics[2:end] : String[],
                    "log_transform" => false,
                    "weights" => w_dict,
                    "description" => "自定义指标组合"
                )
            else
                throw(ArgumentError("未知目标: $(goal)"))
            end

            config["goal"] = goal
            return TextContent(text = JSON3.write(config))
        catch e
            return create_error_response(e)
        end
    end
)

# ==============================================================================
# 9. 校准初始化设置工具 (DEV.md Tool 1: init_calibration_setup)
# ==============================================================================
init_calibration_setup_tool = MCPTool(
    name = "init_calibration_setup",
    description = "一键初始化校准配置(DEV.md Tool 1)。执行敏感性分析筛选参数，检测数据量级，推荐算法和采样策略。返回完整的校准配置建议，可直接用于 calibrate_model。",
    input_schema = Dict{String,Any}(
        "type" => "object",
        "properties" => Dict{String,Any}(
            "model_name" => MODEL_NAME_SCHEMA,
            "forcing" => FORCING_SCHEMA,
            "observation" => OBSERVATION_SCHEMA,
            "obs_column" => OBS_COLUMN_SCHEMA,
            "goal" => GOAL_SCHEMA,
            "budget" => BUDGET_SCHEMA,
            "sensitivity_samples" => SENSITIVITY_SAMPLES_SCHEMA,
            "solver" => SOLVER_SIMPLE_SCHEMA,
            "interpolator" => INTERPOLATOR_SIMPLE_SCHEMA
        ),
        "required" => ["model_name", "forcing", "observation", "obs_column"]
    ),
    handler = function(params)
        # 验证必需参数
        validation_error = validate_required_params(params, ["model_name", "forcing", "observation", "obs_column"])
        if !isnothing(validation_error)
            return create_error_response(validation_error)
        end

        # 验证枚举参数
        if haskey(params, "goal")
            enum_error = validate_enum_param(params, "goal",
                ["general_fit", "peak_flows", "low_flows", "water_balance", "dynamics"], "general_fit")
            if !isnothing(enum_error)
                return create_error_response(enum_error)
            end
        end

        if haskey(params, "budget")
            enum_error = validate_enum_param(params, "budget",
                ["low", "medium", "high"], "medium")
            if !isnothing(enum_error)
                return create_error_response(enum_error)
            end
        end

        try
            forcing_config = params["forcing"]
            obs_config = params["observation"]
            forcing_nt, _ = DataLoader.load_data(Val(Symbol(forcing_config["source_type"])), forcing_config)
            obs_data, _ = DataLoader.load_data(Val(Symbol(obs_config["source_type"])), obs_config)
            obs_vec = Float64.(obs_data[Symbol(params["obs_column"])])

            model_name = params["model_name"]
            goal = get(params, "goal", "general_fit")
            budget = get(params, "budget", "medium")
            n_sens = Int(get(params, "sensitivity_samples", 50))
            solver_type = get(params, "solver", "DISCRETE")
            interp_type = get(params, "interpolator", "LINEAR")

            # Step 1: 敏感性分析
            sens_result = SensitivityAnalysis.run_sensitivity(
                model_name, forcing_nt, obs_vec;
                method="morris", n_samples=n_sens, objective="NSE",
                threshold=0.1, solver_type=solver_type, interp_type=interp_type
            )

            # Step 2: 数据量级检测 (Strategy 3)
            pos_obs = filter(>(0), obs_vec)
            magnitude_ratio = length(pos_obs) > 0 ? maximum(pos_obs) / minimum(pos_obs) : 1.0
            log_recommended = magnitude_ratio > 100.0

            # Step 3: 算法推荐 (Strategy 8)
            n_calibratable = length(sens_result["calibratable"])
            algorithm = if budget == "low" || n_calibratable > 10
                "BBO"  # 高维或低预算用 BBO
            elseif budget == "high"
                "PSO"  # 高预算用 PSO
            else
                "BBO"  # 中等预算用 BBO
            end

            maxiters = if budget == "low"
                500
            elseif budget == "high"
                5000
            else
                1000
            end

            # Step 4: 目标函数推荐
            objective_config = if goal == "general_fit"
                Dict("metric" => "KGE", "log_transform" => false)
            elseif goal == "peak_flows"
                Dict("metric" => "NSE", "log_transform" => false)
            elseif goal == "low_flows"
                Dict("metric" => "LogKGE", "log_transform" => log_recommended)
            elseif goal == "water_balance"
                Dict("metric" => "PBIAS", "log_transform" => false)
            else
                Dict("metric" => "KGE", "log_transform" => false)
            end

            # Step 5: 构建固定参数 Dict
            fixed_params = Dict{String,Any}()
            for pname in sens_result["fixed"]
                fixed_params[pname] = "default"  # Agent 应从模型获取默认值
            end

            setup = Dict{String,Any}(
                "model_name" => sens_result["model_name"],
                "sensitivity" => Dict{String,Any}(
                    "calibratable" => sens_result["calibratable"],
                    "fixed" => sens_result["fixed"],
                    "sensitivities" => Dict(
                        sens_result["param_names"][i] => sens_result["sensitivities"][i]
                        for i in eachindex(sens_result["param_names"])
                    ),
                ),
                "data_analysis" => Dict{String,Any}(
                    "n_observations" => length(obs_vec),
                    "magnitude_ratio" => magnitude_ratio,
                    "log_transform_recommended" => log_recommended,
                ),
                "recommended_config" => Dict{String,Any}(
                    "algorithm" => algorithm,
                    "maxiters" => maxiters,
                    "n_trials" => 3,
                    "objective" => objective_config["metric"],
                    "log_transform" => objective_config["log_transform"],
                    "fixed_params" => fixed_params,
                ),
                "goal" => goal,
                "budget" => budget,
            )

            return TextContent(text = JSON3.write(setup))
        catch e
            return create_error_response(e)
        end
    end
)

export compute_metrics_tool, split_data_tool, sensitivity_tool,
       sampling_tool, calibrate_tool, calibrate_multi_tool, diagnose_tool,
       configure_objectives_tool, init_calibration_setup_tool
