"""
集合模拟模块

提供批量运行多个参数集的模拟功能，支持并行执行和集合统计量计算。
"""
module Ensemble

using ..Simulation
using Statistics
using UUIDs
using Base.Threads

"""
    run_ensemble(args::AbstractDict) -> Dict

执行集合模拟。

# 参数
- `model_name::String`: 模型名称
- `parameter_sets::Vector{Dict}`: 参数集列表
- `forcing::Dict`: 驱动数据配置
- `input_mapping::Dict`: 输入变量映射（可选）
- `solver::String`: 求解器类型（默认"DISCRETE"）
- `interpolator::String`: 插值器类型（默认"LINEAR"）
- `parallel::Bool`: 是否并行执行（默认true）

# 返回
- `n_members::Int`: 集合成员数量
- `ensemble_members::Vector{Dict}`: 每个成员的结果
- `ensemble_mean::Vector{Float64}`: 集合均值
- `ensemble_std::Vector{Float64}`: 集合标准差
- `result_id::String`: 结果ID（用于存储）
"""
function run_ensemble(args::AbstractDict)
    # 1. 提取参数
    model_name = args["model_name"]
    parameter_sets = args["parameter_sets"]
    forcing_config = args["forcing"]

    n_members = length(parameter_sets)

    # 2. 准备模拟参数
    base_args = Dict(
        "model_name" => model_name,
        "forcing" => forcing_config,
        "solver" => get(args, "solver", "DISCRETE"),
        "interpolator" => get(args, "interpolator", "LINEAR")
    )

    if haskey(args, "input_mapping")
        base_args["input_mapping"] = args["input_mapping"]
    end

    # 3. 执行集合模拟
    ensemble_results = Vector{Dict}(undef, n_members)

    if get(args, "parallel", true) && n_members > 1
        # 并行执行
        @threads for i in 1:n_members
            sim_args = copy(base_args)
            sim_args["params"] = parameter_sets[i]

            result = Simulation.run_simulation(sim_args)

            ensemble_results[i] = Dict(
                "member_id" => i - 1,  # 0-based索引
                "parameters" => parameter_sets[i],
                "simulated_flow" => result["result"]
            )
        end
    else
        # 串行执行
        for i in 1:n_members
            sim_args = copy(base_args)
            sim_args["params"] = parameter_sets[i]

            result = Simulation.run_simulation(sim_args)

            ensemble_results[i] = Dict(
                "member_id" => i - 1,
                "parameters" => parameter_sets[i],
                "simulated_flow" => result["result"]
            )
        end
    end

    # 4. 计算集合统计量
    flow_matrix = hcat([r["simulated_flow"] for r in ensemble_results]...)
    ensemble_mean = vec(mean(flow_matrix, dims=2))
    ensemble_std = vec(std(flow_matrix, dims=2))

    # 5. 生成结果ID
    result_id = string(uuid4())

    return Dict(
        "n_members" => n_members,
        "ensemble_members" => ensemble_results,
        "ensemble_mean" => ensemble_mean,
        "ensemble_std" => ensemble_std,
        "result_id" => result_id,
        "model_name" => model_name
    )
end

end  # module Ensemble
