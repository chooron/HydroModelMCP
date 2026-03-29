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

            metric_names = _normalize_metric_names(
                get(params, "metrics", String["NSE", "KGE", "LogNSE", "LogKGE", "PBIAS", "R2", "RMSE", "MAE", "Bias"]),
            )
            n = min(length(sim_vec), length(obs_vec))
            result = Metrics.compute_metrics(sim_vec[1:n], obs_vec[1:n], metric_names)

            return TextContent(text = JSON3.write(result))
        catch e
            return create_error_response(e)
        end
    end
)

function _normalize_metric_names(metric_names_input)
    metric_names_input isa AbstractVector ||
        throw(ArgumentError("metrics must be an array of strings"))

    alias_map = Dict(
        "NSE" => "NSE",
        "KGE" => "KGE",
        "LOGNSE" => "LogNSE",
        "LOGKGE" => "LogKGE",
        "PBIAS" => "PBIAS",
        "R2" => "R2",
        "RMSE" => "RMSE",
        "MAE" => "MAE",
        "BIAS" => "Bias",
    )

    normalized = String[]
    for raw_name in metric_names_input
        canonical = get(alias_map, uppercase(strip(string(raw_name))), string(raw_name))
        push!(normalized, canonical)
    end

    return normalized
end

function _metric_candidates(role::String)
    role == "simulated" && return ["Result", "simulated", "simulation", "runoff", "Q"]
    return ["flow(mm)", "flow", "runoff", "discharge", "streamflow", "Q", "observed"]
end

function _infer_metric_column(df, role::String, explicit_column)
    if !(explicit_column === nothing)
        return String(explicit_column), String[]
    end

    warnings = String[]
    column_names = String.(names(df))
    for candidate in _metric_candidates(role)
        candidate in column_names || continue
        push!(warnings, "Inferred $(role) column '$candidate'")
        return candidate, warnings
    end

    for name in names(df)
        if eltype(df[!, name]) <: Number
            inferred = String(name)
            push!(warnings, "Fell back to first numeric $(role) column '$inferred'")
            return inferred, warnings
        end
    end

    throw(ArgumentError("Could not infer a $(role) column from the CSV"))
end

function _load_metric_series(source_like, role::String)
    source = normalize_string_dict(source_like)
    warnings = String[]

    if haskey(source, "data_handle")
        stored = get_data(string(source["data_handle"]))
        if stored isa AbstractVector
            return Float64.(stored), nothing, nothing, warnings
        elseif stored isa AbstractDict
            explicit_column = get(source, "column", nothing)
            key = if role == "simulated"
                haskey(stored, "simulated") ? "simulated" : explicit_column
            else
                haskey(stored, "obs") ? "obs" : explicit_column
            end
            isnothing(key) && throw(ArgumentError("data_handle does not contain $(role) series"))
            return Float64.(stored[key]), nothing, String(key), warnings
        end
        throw(ArgumentError("Unsupported data_handle payload for $(role)"))
    end

    source_type = lowercase(string(get(source, "source_type", "csv")))
    source_type == "csv" || throw(ArgumentError("Only CSV sources are supported for $(role)"))
    haskey(source, "path") || throw(ArgumentError("$(role) source must include 'path'"))

    path = resolve_workspace_path(string(source["path"]); must_exist = true)
    df = CSV.read(path, DataFrame)
    column_name, infer_warnings = _infer_metric_column(df, role, get(source, "column", nothing))
    append!(warnings, infer_warnings)
    return Float64.(df[!, Symbol(column_name)]), path, column_name, warnings
end

function _write_metrics_artifact(output_dir::String, base_name::String, payload::Dict{String,Any})
    normalized_output_dir = normpath(output_dir)
    metrics_dir = lowercase(basename(normalized_output_dir)) == "metrics" ?
                  normalized_output_dir :
                  joinpath(normalized_output_dir, "metrics")
    mkpath(metrics_dir)
    timestamp = Dates.format(now(), "yyyymmddHHMMSS")
    output_path = joinpath(metrics_dir, "$(base_name)_metrics_$(timestamp).json")
    open(output_path, "w") do io
        JSON3.write(io, payload)
    end
    return output_path
end

compute_metrics_tool = MCPTool(
    name = "compute_metrics",
    description = "Compute evaluation metrics from simulated and observed sources, using file paths or in-memory data handles.",
    input_schema = Dict{String,Any}(
        "type" => "object",
        "properties" => Dict{String,Any}(
            "simulated" => Dict("type" => "object", "description" => "Simulated source descriptor."),
            "observed" => Dict("type" => "object", "description" => "Observed source descriptor."),
            "metrics" => METRICS_SCHEMA,
            "output_dir" => Dict(
                "type" => "string",
                "description" => "Workspace-relative output directory for metrics artifacts.",
                "default" => "./result"
            )
        ),
        "required" => ["simulated", "observed"]
    ),
    handler = function(params)
        validation_error = validate_required_params(params, ["simulated", "observed"])
        if !isnothing(validation_error)
            return create_error_response(validation_error)
        end

        try
            sim_vec, sim_path, sim_column, sim_warnings = _load_metric_series(params["simulated"], "simulated")
            obs_vec, obs_path, obs_column, obs_warnings = _load_metric_series(params["observed"], "observed")

            warnings = String[]
            append!(warnings, sim_warnings)
            append!(warnings, obs_warnings)

            n = min(length(sim_vec), length(obs_vec))
            if length(sim_vec) != length(obs_vec)
                push!(warnings, "Series lengths differ; truncated to $n samples")
            end

            metric_names = _normalize_metric_names(
                get(params, "metrics", String["NSE", "KGE", "RMSE", "PBIAS", "R2", "MAE", "Bias"]),
            )
            metric_values = Metrics.compute_metrics(sim_vec[1:n], obs_vec[1:n], metric_names)

            payload = Dict{String,Any}(
                "status" => "success",
                "metrics" => metric_values,
                "sample_size" => n,
                "warnings" => warnings,
                "simulated_column" => sim_column,
                "observed_column" => obs_column
            )

            output_dir = resolve_workspace_path(string(get(params, "output_dir", "./result")))
            base_name = isnothing(sim_path) ? "simulation" : splitext(basename(sim_path))[1]
            payload["output_path"] = _write_metrics_artifact(output_dir, base_name, payload)
            !isnothing(sim_path) && (payload["simulated_path"] = sim_path)
            !isnothing(obs_path) && (payload["observed_path"] = obs_path)

            return create_json_response(payload)
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
# 3b. sensitivity_analysis 工具（data_handle 接口，兼容 Python HydroAgent 协议）
# ==============================================================================
sensitivity_analysis_tool = MCPTool(
    name = "sensitivity_analysis",
    description = "对水文模型参数进行全局敏感性分析。通过 data_handle 引用已加载数据（需先调用 load_hydro_csv），分析哪些参数对输出有显著影响。支持 morris 和 sobol 方法。",
    input_schema = Dict{String,Any}(
        "type" => "object",
        "properties" => Dict{String,Any}(
            "data_handle" => Dict{String,Any}(
                "type" => "string",
                "description" => "通过 load_hydro_csv 获取的数据句柄（需包含 forcing 数据）"
            ),
            "model_name" => Dict{String,Any}(
                "type" => "string",
                "description" => "水文模型名称（如 exphydro、gr4j 等）"
            ),
            "model_type" => Dict{String,Any}(
                "type" => "string",
                "description" => "model_name 的别名，两者取其一即可"
            ),
            "parameter_names" => Dict{String,Any}(
                "type" => "array",
                "items" => Dict{String,Any}("type" => "string"),
                "description" => "要分析的参数名列表（可选，默认分析全部参数）"
            ),
            "parameter_bounds" => Dict{String,Any}(
                "type" => "object",
                "description" => "自定义参数边界，格式: {\"param\": [lower, upper]}（可选，默认使用模型内置边界）"
            ),
            "parameter_ranges" => Dict{String,Any}(
                "type" => "object",
                "description" => "自定义参数范围（parameter_bounds 别名），格式: {\"param\": {\"lower\": x, \"upper\": y}} 或 [lower, upper]"
            ),
            "method" => Dict{String,Any}(
                "type" => "string",
                "enum" => ["morris", "sobol"],
                "description" => "敏感性分析方法: morris（快速筛选）或 sobol（精确分解），默认 morris",
                "default" => "morris"
            ),
            "n_samples" => Dict{String,Any}(
                "type" => "integer",
                "description" => "采样数，morris 默认 100，sobol 默认 1000"
            ),
            "objective" => Dict{String,Any}(
                "type" => "string",
                "description" => "目标评价指标",
                "default" => "NSE"
            ),
            "threshold" => Dict{String,Any}(
                "type" => "number",
                "description" => "敏感性阈值（归一化后，>=此值为敏感参数），默认 0.1"
            )
        ),
        "required" => ["data_handle"]
    ),
    handler = function(params)
        validation_error = validate_required_params(params, ["data_handle"])
        if !isnothing(validation_error)
            return create_error_response(validation_error)
        end

        # 解析 model_name（支持 model_type 别名）
        model_name = get(params, "model_name", get(params, "model_type", nothing))
        if isnothing(model_name)
            return create_error_response("缺少必需参数: model_name 或 model_type")
        end

        # 验证枚举
        if haskey(params, "method")
            enum_error = validate_enum_param(params, "method", ["morris", "sobol"], "morris")
            if !isnothing(enum_error)
                return create_error_response(enum_error)
            end
        end

        try
            # 从 data_handle 获取 obs + forcing
            handle = params["data_handle"]
            if !has_data(handle)
                return create_error_response("data_handle '$handle' 不存在，请先调用 load_hydro_csv 加载数据")
            end
            stored = get_data(handle)
            if !(stored isa Dict) || !haskey(stored, "obs") || !haskey(stored, "forcing_nt")
                return create_error_response("data_handle '$handle' 不含 forcing 数据，请使用 load_hydro_csv 加载完整数据集")
            end
            obs_vec    = Float64.(stored["obs"])
            forcing_nt = stored["forcing_nt"]

            # 处理可选参数
            method = get(params, "method", "morris")
            default_n = method == "sobol" ? 1000 : 100
            n_samples  = Int(get(params, "n_samples", default_n))
            threshold  = Float64(get(params, "threshold", 0.1))
            objective  = get(params, "objective", "NSE")

            # 自定义参数边界（parameter_bounds 和 parameter_ranges 均支持）
            pb_override = nothing
            raw_pb_sa = haskey(params, "parameter_ranges") ? params["parameter_ranges"] :
                        haskey(params, "parameter_bounds") ? params["parameter_bounds"] : nothing
            if !isnothing(raw_pb_sa)
                pb_override = Dict{String,Vector{Float64}}()
                for (k, v) in raw_pb_sa
                    if v isa AbstractVector
                        pb_override[string(k)] = Float64.(v)
                    elseif v isa Dict || (hasproperty(v, :keys))
                        lo = Float64(get(v, "lower", get(v, :lower, 0.0)))
                        hi = Float64(get(v, "upper", get(v, :upper, 1.0)))
                        pb_override[string(k)] = [lo, hi]
                    end
                end
            end

            # 参数名筛选
            pnames_filter = nothing
            if haskey(params, "parameter_names") && !isempty(params["parameter_names"])
                pnames_filter = String.(params["parameter_names"])
            end

            result = SensitivityAnalysis.run_sensitivity(
                model_name, forcing_nt, obs_vec;
                method                = method,
                n_samples             = n_samples,
                objective             = objective,
                threshold             = threshold,
                param_bounds_override = pb_override,
                param_names_filter    = pnames_filter
            )

            # 转换为规范输出格式
            sensitivity_scores = Dict{String,Float64}(
                result["param_names"][i] => result["sensitivities"][i]
                for i in eachindex(result["param_names"])
            )

            output = Dict{String,Any}(
                "sensitive_parameters"   => result["calibratable"],
                "insensitive_parameters" => result["fixed"],
                "sensitivity_scores"     => sensitivity_scores,
                "method"                 => result["method"],
                "method_used"            => result["method"],
                "threshold"              => result["threshold"],
                "model_name"             => result["model_name"],
                "n_samples"              => result["n_samples"]
            )
            return TextContent(text = JSON3.write(output))
        catch e
            return create_error_response(e)
        end
    end
)
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
    description = "执行水文模型参数自动校准(单目标优化)。支持两种数据输入方式：(1) data_handle（推荐，通过 load_hydro_csv 预加载）；(2) 直接传入 forcing + observation 配置。支持多种算法(BBO/DE/PSO等)。返回最优参数、目标函数值和收敛信息。",
    input_schema = Dict{String,Any}(
        "type" => "object",
        "properties" => Dict{String,Any}(
            # --- 数据输入方式 1：data_handle（与 Python HydroAgent 协议兼容）---
            "data_handle" => Dict{String,Any}(
                "type" => "string",
                "description" => "通过 load_hydro_csv 获取的数据句柄（与 model_type 配合使用）"
            ),
            "model_type" => Dict{String,Any}(
                "type" => "string",
                "description" => "模型类型名称（model_name 的别名，与 data_handle 配合使用）"
            ),
            # --- 数据输入方式 2：直接传入配置（原有接口）---
            "model_name" => MODEL_NAME_SCHEMA,
            "forcing" => FORCING_SCHEMA,
            "observation" => OBSERVATION_SCHEMA,
            "obs_column" => OBS_COLUMN_SCHEMA,
            # --- 通用校准参数 ---
            "objective" => OBJECTIVE_SCHEMA,
            "algorithm" => ALGORITHM_SCHEMA,
            "maxiters" => MAXITERS_SCHEMA,
            "budget" => Dict{String,Any}(
                "type" => "integer",
                "description" => "最大迭代次数（maxiters 的别名，两者取其一）"
            ),
            "n_trials" => N_TRIALS_SCHEMA,
            "trial_id" => Dict{String,Any}(
                "type" => "integer",
                "description" => "本次试验的 ID（用于多次并行试验的追踪），将原样回传到输出"
            ),
            "sampling_method" => Dict{String,Any}(
                "type" => "string",
                "enum" => ["LHS", "sobol", "random"],
                "description" => "初始采样方法（当前版本由算法内部处理，此参数记录用）"
            ),
            "log_transform" => LOG_TRANSFORM_SCHEMA,
            "use_log_transform" => Dict{String,Any}(
                "type" => "boolean",
                "description" => "是否对观测数据做对数变换（log_transform 的别名）",
                "default" => false
            ),
            "fixed_params" => FIXED_PARAMS_SCHEMA,
            "param_bounds" => PARAM_BOUNDS_SCHEMA,
            "parameter_bounds" => Dict{String,Any}(
                "type" => "object",
                "description" => "参数边界（param_bounds 的别名），格式: {\"param\": [lower, upper]}"
            ),
            "parameter_ranges" => Dict{String,Any}(
                "type" => "object",
                "description" => "参数范围（parameter_bounds 的别名），格式: {\"param\": {\"lower\": x, \"upper\": y}} 或 [lower, upper]"
            ),
            "max_iterations" => Dict{String,Any}(
                "type" => "integer",
                "description" => "最大迭代次数（maxiters/budget 的别名）",
                "default" => 1000
            ),
            "random_seed" => Dict{String,Any}(
                "type" => "integer",
                "description" => "随机种子（用于复现，当前版本记录但不强制传入优化器）"
            ),
            "solver" => SOLVER_SCHEMA,
            "interpolator" => INTERPOLATOR_SCHEMA,
            "init_states" => INIT_STATES_SCHEMA
        ),
        "required" => []  # 通过运行时验证，支持两种接口
    ),
    handler = function(params)
        # ---- 解析数据来源 ----
        use_handle = haskey(params, "data_handle")

        # 解析 model_name（支持 model_type 别名）
        model_name = get(params, "model_name", get(params, "model_type", nothing))
        if isnothing(model_name)
            return create_error_response("缺少参数: model_name 或 model_type")
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
                    ["BBO","DE","PSO","CMAES","ECA","dds","DDS"], "BBO")
            if !isnothing(enum_error)
                return create_error_response(enum_error)
            end
        end

        try
            # ---- 加载 forcing + obs ----
            forcing_nt = nothing
            obs_vec    = nothing

            if use_handle
                handle = params["data_handle"]
                if !has_data(handle)
                    return create_error_response("data_handle '$handle' 不存在，请先调用 load_hydro_csv 加载数据")
                end
                stored = get_data(handle)
                if !(stored isa Dict) || !haskey(stored, "obs") || !haskey(stored, "forcing_nt")
                    return create_error_response("data_handle '$handle' 不含 forcing 数据，请使用 load_hydro_csv 加载完整数据集")
                end
                obs_vec    = Float64.(stored["obs"])
                forcing_nt = stored["forcing_nt"]
            else
                # 原有直接配置接口
                validation_error = validate_required_params(params, ["forcing", "observation", "obs_column"])
                if !isnothing(validation_error)
                    return create_error_response("未找到 data_handle，尝试直接配置接口，但: " * validation_error)
                end
                forcing_config = params["forcing"]
                obs_config     = params["observation"]
                forcing_nt, _ = DataLoader.load_data(Val(Symbol(forcing_config["source_type"])), forcing_config)
                obs_data, _   = DataLoader.load_data(Val(Symbol(obs_config["source_type"])), obs_config)
                obs_vec = Float64.(obs_data[Symbol(params["obs_column"])])
            end

            # ---- 解析校准参数 ----
            fixed = Dict{String,Float64}()
            if haskey(params, "fixed_params")
                for (k, v) in params["fixed_params"]
                    fixed[string(k)] = Float64(v)
                end
            end

            # 合并 param_bounds / parameter_bounds / parameter_ranges（多种别名支持）
            pb = nothing
            raw_pb = haskey(params, "param_bounds")       ? params["param_bounds"] :
                     haskey(params, "parameter_bounds")   ? params["parameter_bounds"] :
                     haskey(params, "parameter_ranges")   ? params["parameter_ranges"] : nothing
            if !isnothing(raw_pb)
                pb = Dict{String,Vector{Float64}}()
                for (k, v) in raw_pb
                    if v isa AbstractVector
                        pb[string(k)] = Float64.(v)
                    elseif v isa Dict || (hasproperty(v, :keys))
                        lo = Float64(get(v, "lower", get(v, :lower, 0.0)))
                        hi = Float64(get(v, "upper", get(v, :upper, 1.0)))
                        pb[string(k)] = [lo, hi]
                    end
                end
            end

            ist = nothing
            if haskey(params, "init_states")
                ist = Dict{String,Float64}()
                for (k, v) in params["init_states"]
                    ist[string(k)] = Float64(v)
                end
            end

            # budget / max_iterations 是 maxiters 的别名
            maxiters_val = Int(get(params, "maxiters",
                             get(params, "max_iterations",
                             get(params, "budget", 1000))))

            # use_log_transform 是 log_transform 的别名
            log_transform_val = Bool(get(params, "log_transform",
                                     get(params, "use_log_transform", false)))

            # dds / DDS 映射到 BBO（Dynamically Dimensioned Search 近似实现）
            algorithm_val = let raw_alg = get(params, "algorithm", "BBO")
                uppercase(raw_alg) in ["DDS"] ? "BBO" : raw_alg
            end

            t_start = time()
            result = Calibration.calibrate_model(
                model_name, forcing_nt, obs_vec;
                algorithm     = algorithm_val,
                maxiters      = maxiters_val,
                objective     = get(params, "objective", "KGE"),
                log_transform = log_transform_val,
                fixed_params  = fixed,
                param_bounds  = pb,
                n_trials      = Int(get(params, "n_trials", 1)),
                solver_type   = get(params, "solver", "DISCRETE"),
                interp_type   = get(params, "interpolator", "LINEAR"),
                init_states_dict = ist
            )
            runtime_seconds = time() - t_start

            # ---- 输出格式扩展：兼容 Python HydroAgent 协议 ----
            # best_parameters / final_parameters（best_params 的别名）
            if haskey(result, "best_params")
                result["best_parameters"]  = result["best_params"]
                result["final_parameters"] = result["best_params"]
            end
            # iterations_used（maxiters 的别名）
            result["iterations_used"] = maxiters_val
            # runtime_seconds
            result["runtime_seconds"] = round(runtime_seconds; digits=2)
            # trial_id（如果传入了）
            if haskey(params, "trial_id")
                result["trial_id"] = params["trial_id"]
            end
            # converged：以 best_objective 是否超过阈值作简单判断
            best_obj_val = get(result, "best_objective", 0.0)
            result["converged"] = best_obj_val >= 0.7
            # objective_summary：从 all_trials 第一项提升到顶层
            if haskey(result, "all_trials") && !isempty(result["all_trials"])
                first_trial = result["all_trials"][1]
                if haskey(first_trial, "objective_summary")
                    result["objective_summary"] = first_trial["objective_summary"]
                end
                if haskey(first_trial, "objective_history")
                    result["convergence_curve"] = first_trial["objective_history"]
                end
            end

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

# ==============================================================================
# 10. 完整诊断分析工具
# ==============================================================================
compute_diagnostics_full_tool = MCPTool(
    name = "compute_diagnostics_full",
    description = "完整的校准诊断分析。分析多次试验结果，检查收敛性、参数边界触达、平台期，推荐下一步操作。支持 final_parameters（calibrate_model 输出格式）和 best_parameters 两种字段名。",
    input_schema = Dict{String,Any}(
        "type" => "object",
        "properties" => Dict{String,Any}(
            "trial_results" => Dict{String,Any}(
                "type" => "array",
                "items" => Dict{String,Any}(
                    "type" => "object",
                    "properties" => Dict{String,Any}(
                        "trial_id" => Dict("type" => "integer"),
                        "best_objective" => Dict("type" => "number"),
                        "best_parameters" => Dict("type" => "object",
                            "description" => "最优参数字典（与 final_parameters 二选一）"),
                        "final_parameters" => Dict("type" => "object",
                            "description" => "最优参数字典（best_parameters 的别名）"),
                        "objective_history" => Dict("type" => "array", "items" => Dict("type" => "number")),
                        "iterations_used" => Dict("type" => "integer")
                    )
                ),
                "description" => "试验结果列表（来自 calibrate_model 的输出）"
            ),
            "parameter_bounds" => Dict{String,Any}(
                "type" => "object",
                "description" => "参数边界，格式: {\"param\": {\"lower\": x, \"upper\": y}} 或 {\"param\": [lower, upper]}"
            ),
            "data_handle" => Dict{String,Any}(
                "type" => "string",
                "description" => "（可选）数据句柄，仅用于上下文记录，不影响诊断计算"
            ),
            "model_type" => Dict{String,Any}(
                "type" => "string",
                "description" => "（可选）模型类型，仅用于上下文记录"
            ),
            "objective_threshold" => Dict{String,Any}(
                "type" => "number",
                "description" => "好拟合阈值，默认 0.7"
            )
        ),
        "required" => ["trial_results"]
    ),
    handler = function(params)
        validation_error = validate_required_params(params, ["trial_results"])
        if !isnothing(validation_error)
            return create_error_response(validation_error)
        end

        try
            # ---- 规范化 trial_results ----
            # 将 final_parameters 统一映射到 best_parameters
            raw_trials = params["trial_results"]
            normalized_trials = Dict{String,Any}[]
            for trial in raw_trials
                t = Dict{String,Any}(string(k) => v for (k, v) in trial)
                # final_parameters → best_parameters 兼容
                if haskey(t, "final_parameters") && !haskey(t, "best_parameters")
                    t["best_parameters"] = t["final_parameters"]
                end
                # convergence_curve → objective_history 兼容
                if haskey(t, "convergence_curve") && !haskey(t, "objective_history")
                    t["objective_history"] = t["convergence_curve"]
                end
                # 若无 objective_history，但有 objective_summary，合成一个虚拟历史
                # 使核心函数能正确检测 is_plateaued / is_still_improving
                if !haskey(t, "objective_history") && haskey(t, "objective_summary")
                    os = t["objective_summary"]
                    plateau   = Bool(get(os, "plateau_detected", get(os, :plateau_detected, false)))
                    improving = Bool(get(os, "still_improving",  get(os, :still_improving,  false)))
                    best_obj_v = Float64(get(t, "best_objective", 0.5))
                    if plateau && !improving
                        # 平台期：最后 50 步完全不变
                        t["objective_history"] = vcat(
                            range(max(best_obj_v - 0.3, 0.0), best_obj_v, length=50),
                            fill(best_obj_v, 50)
                        )
                    elseif improving
                        # 仍在改进：末尾仍有下降趋势（内部用的是最小化，所以历史值递减）
                        t["objective_history"] = range(best_obj_v + 0.2, best_obj_v, length=100)
                    end
                end
                push!(normalized_trials, t)
            end

            # ---- 处理 parameter_bounds（支持数组格式 [lower, upper]） ----
            param_bounds = if haskey(params, "parameter_bounds")
                raw_bounds = Dict{String,Any}(string(k) => v for (k, v) in params["parameter_bounds"])
                # 将 [lower, upper] 格式转换为 {"lower": x, "upper": y}
                converted = Dict{String,Any}()
                for (pname, bound) in raw_bounds
                    if bound isa AbstractVector || bound isa Vector
                        converted[pname] = Dict("lower" => Float64(bound[1]), "upper" => Float64(bound[2]))
                    else
                        converted[pname] = bound
                    end
                end
                converted
            else
                # 若未提供 bounds，从 trial_results 中推断（以参数值范围扩大 20% 作为 bounds）
                all_params = String[]
                for t in normalized_trials
                    if haskey(t, "best_parameters")
                        for k in keys(t["best_parameters"])
                            k ∉ all_params && push!(all_params, string(k))
                        end
                    end
                end
                inferred = Dict{String,Any}()
                for pname in all_params
                    vals = [Float64(t["best_parameters"][pname])
                            for t in normalized_trials if haskey(t, "best_parameters") && haskey(t["best_parameters"], pname)]
                    if !isempty(vals)
                        lo, hi = minimum(vals), maximum(vals)
                        margin = max((hi - lo) * 0.2, abs(lo) * 0.1 + 1e-6)
                        inferred[pname] = Dict("lower" => lo - margin, "upper" => hi + margin)
                    end
                end
                inferred
            end

            result = Calibration.diagnose_calibration_full(
                normalized_trials,
                param_bounds;
                objective_threshold = Float64(get(params, "objective_threshold", 0.7))
            )

            # ---- 补充输出字段：best_trial_id + parameter_uncertainty ----
            if !isempty(normalized_trials)
                objectives = [Float64(t["best_objective"]) for t in normalized_trials]
                best_idx = argmax(objectives)
                # trial_id 字段（若有）或用索引
                result["best_trial_id"] = if haskey(normalized_trials[best_idx], "trial_id")
                    normalized_trials[best_idx]["trial_id"]
                else
                    best_idx
                end

                # 参数不确定性统计
                all_pnames = String[]
                for t in normalized_trials
                    for k in keys(get(t, "best_parameters", Dict()))
                        string(k) ∉ all_pnames && push!(all_pnames, string(k))
                    end
                end
                param_uncertainty = Dict{String,Any}()
                for pname in all_pnames
                    vals = [Float64(t["best_parameters"][pname])
                            for t in normalized_trials
                            if haskey(t, "best_parameters") && haskey(t["best_parameters"], pname)]
                    if length(vals) >= 2
                        param_uncertainty[pname] = Dict(
                            "mean" => mean(vals),
                            "std"  => std(vals),
                            "min"  => minimum(vals),
                            "max"  => maximum(vals)
                        )
                    elseif length(vals) == 1
                        param_uncertainty[pname] = Dict("mean" => vals[1], "std" => 0.0,
                                                        "min" => vals[1], "max" => vals[1])
                    end
                end
                result["parameter_uncertainty"] = param_uncertainty
            end

            # 可选上下文字段原样回传
            if haskey(params, "data_handle")
                result["data_handle"] = params["data_handle"]
            end
            if haskey(params, "model_type")
                result["model_type"] = params["model_type"]
            end

            return TextContent(text = JSON3.write(result))
        catch e
            return create_error_response(e)
        end
    end
)

export compute_metrics_tool, split_data_tool, sensitivity_tool, sensitivity_analysis_tool,
       sampling_tool, calibrate_tool, calibrate_multi_tool, diagnose_tool,
       configure_objectives_tool, init_calibration_setup_tool, compute_diagnostics_full_tool
