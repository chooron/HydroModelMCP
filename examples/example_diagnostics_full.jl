# ============================================================================
# compute_diagnostics_full 工具使用示例
# ============================================================================
#
# 本示例展示如何使用 compute_diagnostics_full 工具进行完整的校准诊断分析
#

using JSON3

include("../src/HydroModelMCP.jl")
using .HydroModelMCP.Calibration

println("=" ^ 70)
println("compute_diagnostics_full 工具使用示例")
println("=" ^ 70)

# ============================================================================
# 示例 1: 收敛良好的场景
# ============================================================================
println("\n示例 1: 收敛良好的场景")
println("-" ^ 70)

trial_results_converged = [
    Dict(
        "trial_id" => 1,
        "best_objective" => 0.25,
        "best_parameters" => Dict("x1" => 100.0, "x2" => 0.5, "x3" => 2.0),
        "objective_history" => [0.8, 0.7, 0.6, 0.5, 0.4, 0.35, 0.3, 0.28, 0.26, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
        "iterations_used" => 1000
    ),
    Dict(
        "trial_id" => 2,
        "best_objective" => 0.26,
        "best_parameters" => Dict("x1" => 102.0, "x2" => 0.51, "x3" => 2.1),
        "objective_history" => [0.8, 0.7, 0.6, 0.5, 0.4, 0.35, 0.3, 0.28, 0.27, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26],
        "iterations_used" => 1000
    ),
    Dict(
        "trial_id" => 3,
        "best_objective" => 0.24,
        "best_parameters" => Dict("x1" => 98.0, "x2" => 0.49, "x3" => 1.9),
        "objective_history" => [0.8, 0.7, 0.6, 0.5, 0.4, 0.35, 0.3, 0.27, 0.25, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24],
        "iterations_used" => 1000
    )
]

parameter_bounds = Dict(
    "x1" => Dict("lower" => 50.0, "upper" => 200.0),
    "x2" => Dict("lower" => 0.0, "upper" => 1.0),
    "x3" => Dict("lower" => 0.0, "upper" => 5.0)
)

result1 = diagnose_calibration_full(
    trial_results_converged,
    parameter_bounds;
    objective_threshold=0.7
)

println("收敛状态: ", result1["convergence_status"])
println("目标函数变异系数: ", round(result1["objective_spread"], digits=4))
println("是否进入平台期: ", result1["is_plateaued"])
println("是否仍在改进: ", result1["is_still_improving"])
println("Hat-trick达成: ", result1["hat_trick_achieved"])
println("推荐动作: ", result1["recommended_action"])
println("推理: ", result1["reasoning"])
println("\n统计信息:")
for (k, v) in result1["statistics"]
    println("  $k: ", round(v, digits=4))
end

# ============================================================================
# 示例 2: 参数触及边界
# ============================================================================
println("\n\n示例 2: 参数触及边界")
println("-" ^ 70)

trial_results_boundary = [
    Dict(
        "trial_id" => 1,
        "best_objective" => 0.3,
        "best_parameters" => Dict("x1" => 199.5, "x2" => 0.5, "x3" => 2.0),
        "objective_history" => [0.8, 0.7, 0.6, 0.5, 0.4, 0.35, 0.3],
        "iterations_used" => 500
    )
]

result2 = diagnose_calibration_full(trial_results_boundary, parameter_bounds)

println("收敛状态: ", result2["convergence_status"])
println("触及边界的参数: ", result2["parameters_at_boundary"])
println("推荐动作: ", result2["recommended_action"])
println("推理: ", result2["reasoning"])

# ============================================================================
# 示例 3: 持续改进中
# ============================================================================
println("\n\n示例 3: 持续改进中")
println("-" ^ 70)

trial_results_improving = [
    Dict(
        "trial_id" => 1,
        "best_objective" => 0.1,
        "best_parameters" => Dict("x1" => 100.0, "x2" => 0.5, "x3" => 2.0),
        "objective_history" => [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.12, 0.11, 0.105, 0.1],
        "iterations_used" => 500
    )
]

result3 = diagnose_calibration_full(trial_results_improving, parameter_bounds)

println("收敛状态: ", result3["convergence_status"])
println("是否仍在改进: ", result3["is_still_improving"])
println("推荐动作: ", result3["recommended_action"])
println("推理: ", result3["reasoning"])

# ============================================================================
# 示例 4: 平台期但未收敛
# ============================================================================
println("\n\n示例 4: 平台期但未收敛")
println("-" ^ 70)

trial_results_plateau = [
    Dict(
        "trial_id" => 1,
        "best_objective" => 0.6,
        "best_parameters" => Dict("x1" => 100.0, "x2" => 0.5, "x3" => 2.0),
        "objective_history" => [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.62, 0.61, 0.605, 0.602, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6],
        "iterations_used" => 500
    )
]

result4 = diagnose_calibration_full(trial_results_plateau, parameter_bounds)

println("收敛状态: ", result4["convergence_status"])
println("是否进入平台期: ", result4["is_plateaued"])
println("是否仍在改进: ", result4["is_still_improving"])
println("推荐动作: ", result4["recommended_action"])
println("推理: ", result4["reasoning"])

# ============================================================================
# JSON 输出示例
# ============================================================================
println("\n\n" * "=" ^ 70)
println("JSON 输出示例（用于 Python 端集成）")
println("=" ^ 70)
println(JSON3.write(result1, allow_inf=true))

println("\n✅ 所有示例运行完成！")
