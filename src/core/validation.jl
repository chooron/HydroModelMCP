"""
验证期运行模块

提供在率定期和验证期分别运行模型并计算性能指标的功能。
"""
module Validation

using ..Simulation
using ..Metrics
using ..DataLoader
using ..DataSplitter
using Dates

"""
    run_validation(args::AbstractDict) -> Dict

在率定期和验证期分别运行模型并计算指标。

# 参数
- `model_name::String`: 模型名称
- `parameters::Dict`: 模型参数
- `forcing::Dict`: 驱动数据配置
- `observation::Dict`: 观测数据配置
- `obs_column::String`: 观测值列名
- `calibration_period::Dict`: 率定期（start, end）
- `validation_period::Dict`: 验证期（start, end）
- `metrics::Vector{String}`: 评价指标列表（默认["NSE", "KGE", "RMSE"]）

# 返回
- `calibration_metrics::Dict`: 率定期指标
- `validation_metrics::Dict`: 验证期指标
- `parameters::Dict`: 使用的参数
- `calibration_period::Dict`: 率定期时间范围
- `validation_period::Dict`: 验证期时间范围
"""
function run_validation(args::AbstractDict)
    # 1. 提取参数
    model_name = args["model_name"]
    parameters = args["parameters"]
    forcing_config = args["forcing"]
    obs_config = args["observation"]
    obs_column = args["obs_column"]

    cal_period = args["calibration_period"]
    val_period = args["validation_period"]

    metrics_list = get(args, "metrics", ["NSE", "KGE", "RMSE"])

    # 2. 加载完整数据
    forcing_nt, forcing_meta = DataLoader.process_io(
        Val(Symbol(forcing_config["source_type"])),
        forcing_config,
        :load
    )

    obs_nt, obs_meta = DataLoader.process_io(
        Val(Symbol(obs_config["source_type"])),
        obs_config,
        :load
    )

    obs_vec = Float64.(obs_nt[Symbol(obs_column)])

    # 3. 解析时间范围
    if haskey(forcing_meta, "dates")
        dates = Date.(forcing_meta["dates"])
    else
        # 如果没有日期信息，使用索引
        dates = nothing
    end

    # 4. 分割数据
    if !isnothing(dates)
        cal_start = Date(cal_period["start"])
        cal_end = Date(cal_period["end"])
        val_start = Date(val_period["start"])
        val_end = Date(val_period["end"])

        cal_indices = findall(d -> cal_start <= d <= cal_end, dates)
        val_indices = findall(d -> val_start <= d <= val_end, dates)
    else
        # 使用索引范围
        cal_indices = cal_period["start_index"]:cal_period["end_index"]
        val_indices = val_period["start_index"]:val_period["end_index"]
    end

    # 5. 率定期模拟
    cal_forcing_nt = NamedTuple{keys(forcing_nt)}(
        Tuple([forcing_nt[k][cal_indices] for k in keys(forcing_nt)])
    )

    cal_sim_args = Dict(
        "model_name" => model_name,
        "params" => parameters,
        "forcing" => Dict(
            "source_type" => "json",
            "data" => Dict(string(k) => v for (k, v) in pairs(cal_forcing_nt))
        ),
        "solver" => get(args, "solver", "DISCRETE"),
        "interpolator" => get(args, "interpolator", "LINEAR")
    )

    if haskey(args, "input_mapping")
        cal_sim_args["input_mapping"] = args["input_mapping"]
    end

    cal_result = Simulation.run_simulation(cal_sim_args)
    cal_sim = cal_result["result"]
    cal_obs = obs_vec[cal_indices]

    # 6. 验证期模拟
    val_forcing_nt = NamedTuple{keys(forcing_nt)}(
        Tuple([forcing_nt[k][val_indices] for k in keys(forcing_nt)])
    )

    val_sim_args = Dict(
        "model_name" => model_name,
        "params" => parameters,
        "forcing" => Dict(
            "source_type" => "json",
            "data" => Dict(string(k) => v for (k, v) in pairs(val_forcing_nt))
        ),
        "solver" => get(args, "solver", "DISCRETE"),
        "interpolator" => get(args, "interpolator", "LINEAR")
    )

    if haskey(args, "input_mapping")
        val_sim_args["input_mapping"] = args["input_mapping"]
    end

    val_result = Simulation.run_simulation(val_sim_args)
    val_sim = val_result["result"]
    val_obs = obs_vec[val_indices]

    # 7. 计算指标
    cal_metrics = Metrics.compute_metrics(
        cal_sim, cal_obs, metrics_list
    )

    val_metrics = Metrics.compute_metrics(
        val_sim, val_obs, metrics_list
    )

    return Dict(
        "calibration_metrics" => cal_metrics,
        "validation_metrics" => val_metrics,
        "parameters" => parameters,
        "calibration_period" => cal_period,
        "validation_period" => val_period,
        "model_name" => model_name
    )
end

end  # module Validation
