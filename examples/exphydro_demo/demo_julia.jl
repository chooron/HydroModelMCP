# ExpHydro 模型演示 - Julia 直接调用版本
# 不通过 MCP 服务器，直接调用核心功能

using Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))
using HydroModelMCP
using JSON3
using CSV
using DataFrames
using HydroModels
using HydroModelLibrary
using ComponentArrays
using DataInterpolations

println("=" ^ 60)
println("ExpHydro 模型演示 - Julia 直接调用")
println("=" ^ 60)

# 数据路径
data_path = joinpath(@__DIR__, "../../data/03604000.csv")

# 步骤 1: 获取模型信息
println("\n[步骤 1] 获取模型信息...")
model_info = HydroModelMCP.Discovery.get_model_info("exphydro")
println("模型: ", model_info["model_name"])
println("输入变量: ", model_info["inputs"])
println("输出变量: ", model_info["outputs"])

# 步骤 2: 获取参数边界
println("\n[步骤 2] 获取参数边界...")
params_info = HydroModelMCP.Discovery.get_parameters_detail("exphydro")
println("参数边界:")
for param_info in params_info
    bounds = param_info["bounds"]
    if !isnothing(bounds)
        println("  $(param_info["name"]): [$(bounds[1]), $(bounds[2])] $(param_info["unit"])")
    else
        println("  $(param_info["name"]): 无边界 $(param_info["unit"])")
    end
end

# 步骤 3: 读取数据
println("\n[步骤 3] 读取数据...")
df = CSV.read(data_path, DataFrame)
forcing_nt = (
    P = df[!, "prcp(mm/day)"],
    T = df[!, "tmean(C)"],
    Ep = df[!, "pet(mm)"]
)
observed = df[!, "flow(mm)"]
println("数据长度: ", length(observed))

# 步骤 4: 加载模型
println("\n[步骤 4] 加载 exphydro 模型...")
model_module = HydroModelLibrary.load_model(:exphydro, reload=false)
model = Base.invokelatest(m -> m.model, model_module)
println("模型加载成功")

# 步骤 5: 运行初始模拟（使用随机参数）
println("\n[步骤 5] 运行初始模拟（使用随机参数）...")
random_params = HydroModelLibrary.get_random_params(:exphydro)

# 准备输入数据
input_names = HydroModels.get_input_names(model)
input_matrix = stack([Float64.(forcing_nt[n]) for n in input_names], dims=1)

# 配置求解器
hydro_config = HydroModels.HydroConfig(
    solver=HydroModels.ODESolver,
    interpolator=Val(HydroModels.LinearInterpolation)
)

# 运行模拟
result_matrix = model(
    input_matrix,
    random_params;
    config=hydro_config
)

initial_sim = result_matrix[end, :]  # 取最后一行（流量）
println("模拟完成! 时间步数: ", length(initial_sim))

# 计算初始性能
initial_metrics = HydroModelMCP.Metrics.compute_metrics(
    initial_sim, observed,
    ["NSE", "KGE", "RMSE"]
)
println("初始性能 (随机参数):")
for (metric, value) in initial_metrics
    println("  $metric: $(round(value, digits=4))")
end

# 步骤 6: 数据分割
println("\n[步骤 6] 分割数据集...")
split_result = HydroModelMCP.DataSplitter.split_data(
    observed, forcing_nt;
    method="split_sample",
    ratio=0.7
)
println("训练集: $(split_result["cal_indices"][1]) 到 $(split_result["cal_indices"][2])")
println("验证集: $(split_result["val_indices"][1]) 到 $(split_result["val_indices"][2])")

# 提取训练和验证数据
train_forcing = split_result["cal_forcing"]
test_forcing = split_result["val_forcing"]
train_obs = split_result["cal_obs"]
test_obs = split_result["val_obs"]

# 步骤 7: 参数率定
println("\n[步骤 7] 运行参数率定（这可能需要一些时间）...")
println("使用 BBO 算法，最大迭代次数: 50")
calib_result = HydroModelMCP.Calibration.calibrate_model(
    "exphydro",
    train_forcing,
    train_obs;
    algorithm="BBO",
    maxiters=50,
    objective="KGE",
    solver_type="ODE",
    interp_type="LINEAR"
)

println("\n率定完成!")
println("最优参数:")
for (param, value) in pairs(calib_result["best_params"])
    println("  $param: $(round(value, digits=6))")
end
println("\n训练集 KGE: $(round(calib_result["best_objective"], digits=4))")

# 步骤 8: 在验证集上测试
println("\n[步骤 8] 在验证集上测试...")
test_input_matrix = stack([Float64.(test_forcing[n]) for n in input_names], dims=1)

# 将 Dict 转换为 ComponentVector，包装在 params 键下
best_params_nt = NamedTuple{Tuple(Symbol.(keys(calib_result["best_params"])))}(values(calib_result["best_params"]))
best_params_cv = ComponentVector(params=best_params_nt)

test_result = model(
    test_input_matrix,
    best_params_cv;
    config=hydro_config
)

test_sim = test_result[end, :]

# 计算验证集指标
test_metrics = HydroModelMCP.Metrics.compute_metrics(
    test_sim, test_obs,
    ["NSE", "KGE", "RMSE", "PBIAS", "R2"]
)

println("\n验证集性能:")
for (metric, value) in test_metrics
    println("  $metric: $(round(value, digits=4))")
end

# 步骤 9: 计算全数据集性能
println("\n[步骤 9] 计算全数据集性能...")
full_result = model(
    input_matrix,
    best_params_cv;
    config=hydro_config
)
full_sim = full_result[end, :]

full_metrics = HydroModelMCP.Metrics.compute_metrics(
    full_sim, observed,
    ["NSE", "KGE", "RMSE", "PBIAS", "R2"]
)

println("\n全数据集性能:")
for (metric, value) in full_metrics
    println("  $metric: $(round(value, digits=4))")
end

# 保存结果
println("\n[步骤 10] 保存结果...")
output_file = joinpath(@__DIR__, "calibration_results.json")
result_dict = Dict(
    "model" => "exphydro",
    "data_file" => data_path,
    "data_length" => length(observed),
    "train_period" => "$(split_result["cal_indices"][1]) - $(split_result["cal_indices"][2])",
    "test_period" => "$(split_result["val_indices"][1]) - $(split_result["val_indices"][2])",
    "best_parameters" => Dict(string(k) => v for (k, v) in pairs(calib_result["best_params"])),
    "train_objective" => calib_result["best_objective"],
    "initial_metrics" => initial_metrics,
    "test_metrics" => test_metrics,
    "full_metrics" => full_metrics
)

open(output_file, "w") do io
    JSON3.pretty(io, result_dict)
end
println("结果已保存到: $output_file")

# 步骤 11: 集合模拟测试
println("\n[步骤 11] 测试集合模拟功能...")
println("生成5个参数集进行集合模拟...")

# 生成参数集（基于最优参数添加扰动）
parameter_sets = []
for i in 1:5
    perturbed_params = Dict()
    for (param, value) in pairs(calib_result["best_params"])
        # 添加±10%的随机扰动
        perturbation = value * (rand() * 0.2 - 0.1)
        perturbed_params[string(param)] = value + perturbation
    end
    push!(parameter_sets, perturbed_params)
end

# 准备集合模拟参数
ensemble_args = Dict(
    "model_name" => "exphydro",
    "parameter_sets" => parameter_sets,
    "forcing" => Dict(
        "source_type" => "json",
        "data" => Dict(
            "P" => forcing_nt.P,
            "T" => forcing_nt.T,
            "Ep" => forcing_nt.Ep
        )
    ),
    "solver" => "ODE",
    "interpolator" => "LINEAR",
    "parallel" => true
)

# 运行集合模拟
ensemble_result = HydroModelMCP.Ensemble.run_ensemble(ensemble_args)

println("集合模拟完成!")
println("  集合成员数: ", ensemble_result["n_members"])
println("  结果ID: ", ensemble_result["result_id"])
println("  集合均值范围: [$(round(minimum(ensemble_result["ensemble_mean"]), digits=2)), $(round(maximum(ensemble_result["ensemble_mean"]), digits=2))]")
println("  集合标准差范围: [$(round(minimum(ensemble_result["ensemble_std"]), digits=2)), $(round(maximum(ensemble_result["ensemble_std"]), digits=2))]")

# 计算集合均值的性能
ensemble_mean_metrics = HydroModelMCP.Metrics.compute_metrics(
    ensemble_result["ensemble_mean"], observed,
    ["NSE", "KGE", "RMSE"]
)
println("\n集合均值性能:")
for (metric, value) in ensemble_mean_metrics
    println("  $metric: $(round(value, digits=4))")
end

# 步骤 12: 验证期运行测试
println("\n[步骤 12] 测试验证期运行功能...")

# 准备验证期运行参数
validation_args = Dict(
    "model_name" => "exphydro",
    "parameters" => Dict(string(k) => v for (k, v) in pairs(calib_result["best_params"])),
    "forcing" => Dict(
        "source_type" => "json",
        "data" => Dict(
            "P" => forcing_nt.P,
            "T" => forcing_nt.T,
            "Ep" => forcing_nt.Ep
        )
    ),
    "observation" => Dict(
        "source_type" => "json",
        "data" => Dict("flow" => observed)
    ),
    "obs_column" => "flow",
    "calibration_period" => Dict(
        "start_index" => split_result["cal_indices"][1],
        "end_index" => split_result["cal_indices"][2]
    ),
    "validation_period" => Dict(
        "start_index" => split_result["val_indices"][1],
        "end_index" => split_result["val_indices"][2]
    ),
    "metrics" => ["NSE", "KGE", "RMSE", "PBIAS", "R2"],
    "solver" => "ODE",
    "interpolator" => "LINEAR"
)

# 运行验证期测试
validation_result = HydroModelMCP.Validation.run_validation(validation_args)

println("验证期运行完成!")
println("\n率定期性能:")
for (metric, value) in validation_result["calibration_metrics"]
    println("  $metric: $(round(value, digits=4))")
end

println("\n验证期性能:")
for (metric, value) in validation_result["validation_metrics"]
    println("  $metric: $(round(value, digits=4))")
end

# 计算性能下降
nse_drop = validation_result["calibration_metrics"]["NSE"] - validation_result["validation_metrics"]["NSE"]
kge_drop = validation_result["calibration_metrics"]["KGE"] - validation_result["validation_metrics"]["KGE"]
println("\n性能下降分析:")
println("  NSE下降: $(round(nse_drop, digits=4)) ($(round(nse_drop/validation_result["calibration_metrics"]["NSE"]*100, digits=2))%)")
println("  KGE下降: $(round(kge_drop, digits=4)) ($(round(kge_drop/validation_result["calibration_metrics"]["KGE"]*100, digits=2))%)")

# 保存扩展结果
println("\n[步骤 13] 保存扩展结果...")
extended_output_file = joinpath(@__DIR__, "extended_results.json")
extended_result_dict = Dict(
    "basic_results" => result_dict,
    "ensemble_simulation" => Dict(
        "n_members" => ensemble_result["n_members"],
        "result_id" => ensemble_result["result_id"],
        "ensemble_mean_metrics" => ensemble_mean_metrics,
        "ensemble_std_range" => [minimum(ensemble_result["ensemble_std"]), maximum(ensemble_result["ensemble_std"])]
    ),
    "validation_run" => Dict(
        "calibration_metrics" => validation_result["calibration_metrics"],
        "validation_metrics" => validation_result["validation_metrics"],
        "performance_drop" => Dict(
            "NSE" => nse_drop,
            "KGE" => kge_drop
        )
    )
)

open(extended_output_file, "w") do io
    JSON3.pretty(io, extended_result_dict)
end
println("扩展结果已保存到: $extended_output_file")

println("\n" * "=" ^ 60)
println("演示完成! (包含新功能测试)")
println("=" ^ 60)
