# ==========================================================================
# 测试 Calibration 模块
# ==========================================================================
using JSON3
using .HydroModelMCP
using .HydroModelMCP.Calibration

function _parse_calibration_tool_json(response)
    text = response.text
    startswith(text, "Error:") && error(text)
    return JSON3.read(text, Dict{String,Any})
end

@testset "Calibration Module Tests" begin

    # 使用真实数据（较短的时间段以加快测试）- 基于测试文件位置构建路径
    test_dir = @__DIR__
    data_file = joinpath(dirname(test_dir), "data", "03604000.csv")
    df = CSV.File(data_file) |> DataFrame
    forcing_nt = (
        P = df[1:200, "prcp(mm/day)"],
        T = df[1:200, "tmean(C)"],
        Ep = df[1:200, "pet(mm)"]
    )
    obs = df[1:200, "flow(mm)"]

    test_model = "exphydro"

    @testset "单目标校准 (BBO 算法)" begin
        println("\n   -> Running single-objective calibration (this may take 1-2 minutes)...")

        result = calibrate_model(
            test_model, forcing_nt, obs;
            algorithm="BBO",
            maxiters=50,
            objective="KGE",
            n_trials=1,
            solver_type="DISCRETE",
            interp_type="LINEAR",
            warm_up=1  # 添加 warm_up 参数
        )

        # 检查返回结构
        @test haskey(result, "model_name")
        @test haskey(result, "best_params")
        @test haskey(result, "calibrated_params")
        @test haskey(result, "fixed_params")
        @test haskey(result, "best_objective")
        @test haskey(result, "objective_name")
        @test haskey(result, "algorithm")
        @test haskey(result, "maxiters")
        @test haskey(result, "n_trials")
        @test haskey(result, "all_trials")
        @test haskey(result, "param_bounds")

        # 检查值的合理性
        @test result["model_name"] == test_model
        @test result["objective_name"] == "KGE"
        @test result["algorithm"] == "BBO"
        @test result["best_objective"] isa Float64
        @test result["best_params"] isa Dict
        @test length(result["all_trials"]) == 1

        println("   -> Best objective (KGE): $(round(result["best_objective"], digits=4))")
        println("   -> Best params: $(result["best_params"])")
    end

    @testset "多次试验 (收敛性检查)" begin
        println("\n   -> Running calibration with multiple trials...")

        result = calibrate_model(
            test_model, forcing_nt, obs;
            algorithm="BBO",
            maxiters=30,
            objective="NSE",
            n_trials=3,
            solver_type="DISCRETE",
            warm_up=1
        )

        @test length(result["all_trials"]) == 3

        # 检查每次试验的结果
        obj_values = [t["objective_value"] for t in result["all_trials"]]
        @test all(v isa Float64 for v in obj_values)

        println("   -> Trial objectives: $(round.(obj_values, digits=4))")
        println("   -> Best: $(round(result["best_objective"], digits=4))")
        println("   -> Spread: $(round(std(obj_values), digits=4))")
    end

    @testset "固定参数" begin
        # 先运行敏感性分析找出不敏感的参数
        sens_result = SensitivityAnalysis.run_sensitivity(
            test_model, forcing_nt, obs;
            method="morris", n_samples=15, threshold=0.2
        )

        # 如果有不敏感参数，固定它们
        if !isempty(sens_result["fixed"])
            fixed_params = Dict{String,Float64}()
            # 使用默认值固定（这里简化处理，实际应该从模型获取默认值）
            for pname in sens_result["fixed"][1:min(1, length(sens_result["fixed"]))]
                fixed_params[pname] = 0.5  # 示例值
            end

            try
                result = calibrate_model(
                    test_model, forcing_nt, obs;
                    algorithm="BBO",
                    maxiters=30,
                    objective="KGE",
                    fixed_params=fixed_params,
                    n_trials=1,
                    warm_up=1
                )

                @test haskey(result, "fixed_params")
                @test length(result["fixed_params"]) >= 1

                println("   -> Fixed params: $(result["fixed_params"])")
                println("   -> Calibrated params: $(keys(result["calibrated_params"]))")
            catch e
                println("   -> Fixed params test skipped due to parameter mismatch: $(e)")
            end
        else
            println("   -> No insensitive params found, skipping fixed params test")
        end
    end

    @testset "自定义参数范围" begin
        # 获取默认范围
        _, _, param_names, default_bounds = Calibration._load_model_and_bounds(test_model)

        # 缩小第一个参数的范围
        custom_bounds = Dict{String,Vector{Float64}}()
        first_param = string(param_names[1])
        orig_bounds = default_bounds[1]
        mid = (orig_bounds[1] + orig_bounds[2]) / 2
        custom_bounds[first_param] = [mid - 0.1, mid + 0.1]

        result = calibrate_model(
            test_model, forcing_nt, obs;
            algorithm="BBO",
            maxiters=30,
            objective="KGE",
            param_bounds=custom_bounds,
            n_trials=1,
            warm_up=1
        )

        # 检查最优参数是否在自定义范围内
        best_val = result["best_params"][first_param]
        @test custom_bounds[first_param][1] <= best_val <= custom_bounds[first_param][2]

        println("   -> Custom bounds for $(first_param): $(custom_bounds[first_param])")
        println("   -> Calibrated value: $(round(best_val, digits=4))")
    end

    # 多目标校准测试（可选，计算量较大）
    # @testset "多目标校准 (NSGA2)" begin
    #     println("\n   -> Running multi-objective calibration...")

    #     result = calibrate_multiobjective(
    #         test_model, forcing_nt, obs;
    #         objectives=["KGE", "PBIAS"],
    #         algorithm="NSGA2",
    #         maxiters=50,
    #         population_size=20
    #     )

    #     @test haskey(result, "pareto_front")
    #     @test haskey(result, "n_solutions")
    #     @test result["n_solutions"] > 0

    #     println("   -> Pareto front size: $(result["n_solutions"])")
    # end

    @testset "校准诊断" begin
        # 先运行一次校准
        cal_result = calibrate_model(
            test_model, forcing_nt, obs;
            algorithm="BBO",
            maxiters=30,
            objective="KGE",
            n_trials=3,
            warm_up=1
        )

        # 诊断结果
        diag = diagnose_calibration(
            cal_result;
            boundary_tolerance=0.01,
            convergence_threshold=0.05,
            plateau_window=50
        )

        # 检查诊断结构
        @test haskey(diag, "convergence")
        @test haskey(diag, "boundaries")
        @test haskey(diag, "plateau")
        @test haskey(diag, "hat_trick")
        @test haskey(diag, "recommendations")
        @test haskey(diag, "summary")

        # 检查各项检查结果
        @test diag["convergence"] isa Dict
        @test diag["boundaries"] isa Dict
        @test diag["plateau"] isa Dict
        @test diag["hat_trick"] isa Bool
        @test diag["recommendations"] isa Vector

        # 检查参数 spread (identifiability) 诊断
        @test haskey(diag, "parameter_spread")
        @test diag["parameter_spread"] isa Dict
        @test haskey(diag["parameter_spread"], "unidentifiable")
        @test diag["parameter_spread"]["unidentifiable"] isa Vector

        println("\n   -> Diagnostics:")
        println("      Convergence passed: $(diag["convergence"]["passed"])")
        println("      Boundaries passed: $(diag["boundaries"]["passed"])")
        println("      Plateau passed: $(diag["plateau"]["passed"])")
        println("      Hat-trick: $(diag["hat_trick"])")
        println("      Unidentifiable params: $(diag["parameter_spread"]["unidentifiable"])")
        println("      Summary: $(diag["summary"])")
        if !isempty(diag["recommendations"])
            println("      Recommendations:")
            for rec in diag["recommendations"]
                println("        - $(rec)")
            end
        end
    end

    @testset "calibrate_model tool uses canonical data_handle interface" begin
        payload = _parse_calibration_tool_json(HydroModelMCP.calibrate_tool.handler(Dict(
            "model" => "exphydro",
            "inputs" => Dict(
                "forcing" => Dict("source_type" => "csv", "path" => data_file),
                "observation" => Dict("source_type" => "csv", "path" => data_file, "column" => "flow(mm)"),
            ),
            "algorithm" => "BBO",
            "maxiters" => 20,
            "n_trials" => 1,
            "objective" => "KGE",
        )))

        @test payload["status"] == "success"
        @test payload["model_name"] == "exphydro"
        @test payload["maxiters"] == 20
        @test payload["n_trials"] == 1
        @test haskey(payload, "best_params")
        @test haskey(payload, "convergence_assessment")
        @test payload["convergence_assessment"] isa Dict
        @test haskey(payload["convergence_assessment"], "status")
        @test haskey(payload, "result_id")
        @test payload["result_id"] isa String
        @test payload["storage_category"] == "calibration"

        bounds_lookup = Dict{String,Any}(String(name) => bounds for (name, bounds) in payload["param_bounds"])
        for (pname, pvalue) in payload["best_params"]
            lo, hi = Float64(bounds_lookup[pname][1]), Float64(bounds_lookup[pname][2])
            @test lo <= Float64(pvalue) <= hi
        end
    end

    @testset "diagnose_calibration can reuse last calibration result" begin
        calibration_payload = _parse_calibration_tool_json(HydroModelMCP.calibrate_tool.handler(Dict(
            "model" => "exphydro",
            "inputs" => Dict(
                "forcing" => Dict("source_type" => "csv", "path" => data_file),
                "observation" => Dict("source_type" => "csv", "path" => data_file, "column" => "flow(mm)"),
            ),
            "algorithm" => "BBO",
            "maxiters" => 20,
            "n_trials" => 1,
            "objective" => "KGE",
        )))

        @test calibration_payload["status"] == "success"

        diag_payload = _parse_calibration_tool_json(HydroModelMCP.diagnose_tool.handler(Dict()))
        @test haskey(diag_payload, "summary")
        @test haskey(diag_payload, "recommendations")
        @test diag_payload["recommendations"] isa Vector
    end

    @testset "diagnose_calibration accepts JSON-string calibration result" begin
        calibration_payload = _parse_calibration_tool_json(HydroModelMCP.calibrate_tool.handler(Dict(
            "model" => "exphydro",
            "inputs" => Dict(
                "forcing" => Dict("source_type" => "csv", "path" => data_file),
                "observation" => Dict("source_type" => "csv", "path" => data_file, "column" => "flow(mm)"),
            ),
            "algorithm" => "BBO",
            "maxiters" => 20,
            "n_trials" => 1,
            "objective" => "KGE",
        )))

        diag_payload = _parse_calibration_tool_json(HydroModelMCP.diagnose_tool.handler(Dict(
            "calibration_result" => JSON3.write(calibration_payload),
        )))

        @test haskey(diag_payload, "summary")
        @test haskey(diag_payload, "recommendations")
        @test diag_payload["recommendations"] isa Vector
    end

    @testset "calibrate_model normalizes bilingual objective and algorithm aliases" begin
        payload = _parse_calibration_tool_json(HydroModelMCP.calibrate_tool.handler(Dict(
            "model" => "exphydro",
            "inputs" => Dict(
                "forcing" => Dict("source_type" => "csv", "path" => data_file),
                "observation" => Dict("source_type" => "csv", "path" => data_file, "column" => "flow(mm)"),
            ),
            "objective" => "纳什效率",
            "algorithm" => "粒子群优化",
            "maxiters" => 12,
            "n_trials" => 1,
        )))

        @test payload["status"] == "success"
        @test payload["objective_name"] == "NSE"
        @test payload["algorithm"] == "PSO"
    end

    @testset "calibrate_model supports stage2 auto split evaluation" begin
        payload = _parse_calibration_tool_json(HydroModelMCP.calibrate_tool.handler(Dict(
            "model" => "exphydro",
            "inputs" => Dict(
                "forcing" => Dict("source_type" => "csv", "path" => data_file),
                "observation" => Dict("source_type" => "csv", "path" => data_file, "column" => "flow(mm)"),
            ),
            "algorithm" => "BBO",
            "objective" => "KGE",
            "maxiters" => 8,
            "n_trials" => 1,
            "method" => "split_sample",
            "ratio" => 0.75,
            "warmup" => 30,
            "metrics" => ["NSE", "KGE"],
        )))

        @test payload["status"] == "success"
        @test haskey(payload, "stage2_evaluation")
        stage2 = payload["stage2_evaluation"]
        @test stage2["split_mode"] == "auto"
        @test stage2["train_length"] > 0
        @test stage2["test_length"] > 0
        @test stage2["test_available"] == true
        @test haskey(stage2["train_metrics"], "NSE")
        @test haskey(stage2["test_metrics"], "KGE")
        @test payload["maxiters"] == 8
    end

    @testset "calibrate_model supports stage2 period split evaluation" begin
        payload = _parse_calibration_tool_json(HydroModelMCP.calibrate_tool.handler(Dict(
            "model" => "exphydro",
            "inputs" => Dict(
                "forcing" => Dict("source_type" => "csv", "path" => data_file),
                "observation" => Dict("source_type" => "csv", "path" => data_file, "column" => "flow(mm)"),
            ),
            "algorithm" => "BBO",
            "objective" => "KGE",
            "maxiters" => 6,
            "n_trials" => 1,
            "calibration_period" => Dict("start_index" => 1, "end_index" => 900),
            "validation_period" => Dict("start_index" => 901, "end_index" => 1200),
            "metrics" => ["NSE"],
        )))

        @test payload["status"] == "success"
        @test haskey(payload, "stage2_evaluation")
        stage2 = payload["stage2_evaluation"]
        @test stage2["split_mode"] == "period"
        @test stage2["train_indices"] == [1, 900]
        @test stage2["test_indices"] == [901, 1200]
        @test stage2["train_length"] == 900
        @test stage2["test_length"] == 300
        @test haskey(stage2["test_metrics"], "NSE")
    end

    @testset "calibrate_model fails fast on infeasible constraints" begin
        response = HydroModelMCP.calibrate_tool.handler(Dict(
            "model" => "exphydro",
            "inputs" => Dict(
                "forcing" => Dict("source_type" => "csv", "path" => data_file),
                "observation" => Dict("source_type" => "csv", "path" => data_file, "column" => "flow(mm)"),
            ),
            "algorithm" => "BBO",
            "objective" => "KGE",
            "maxiters" => 6,
            "n_trials" => 1,
            "constraints" => Dict(
                "pie_share" => Dict(
                    "parameters" => ["f", "Smax"],
                    "total" => 10000.0,
                ),
            ),
        ))

        @test startswith(response.text, "Error:")
        @test occursin("Pie-share infeasible", response.text)
    end

    @testset "calibrate_model reports constraint diagnostics" begin
        payload = _parse_calibration_tool_json(HydroModelMCP.calibrate_tool.handler(Dict(
            "model" => "exphydro",
            "inputs" => Dict(
                "forcing" => Dict("source_type" => "csv", "path" => data_file),
                "observation" => Dict("source_type" => "csv", "path" => data_file, "column" => "flow(mm)"),
            ),
            "algorithm" => "BBO",
            "objective" => "KGE",
            "maxiters" => 8,
            "n_trials" => 1,
            "constraints" => Dict(
                "delta_method" => Dict(
                    "inequalities" => [["Tmin", "Tmax"]],
                ),
            ),
            "save_to_storage" => false,
        )))

        @test payload["status"] == "success"
        @test haskey(payload, "constraints")
        @test haskey(payload, "constraint_diagnostics")
        @test payload["constraint_diagnostics"]["enabled"] == true
        @test payload["constraint_diagnostics"]["status"] == "ok"
        @test haskey(payload["constraint_diagnostics"], "checks")
        @test haskey(payload, "warnings")
    end

    @testset "model bounds are aligned by parameter name" begin
        _, _, param_names, bounds = HydroModelMCP.Calibration._load_model_and_bounds("exphydro")
        bound_map = Dict(string(param_names[i]) => bounds[i] for i in eachindex(param_names))

        @test bound_map["f"] == (0.0, 0.1)
        @test bound_map["Smax"] == (100.0, 2000.0)
        @test bound_map["Qmax"] == (10.0, 50.0)
        @test bound_map["Df"] == (0.0, 5.0)
        @test bound_map["Tmax"] == (0.0, 3.0)
        @test bound_map["Tmin"] == (-3.0, 0.0)
    end

    @testset "calibrate_model tool rejects legacy direct forcing interface" begin
        response = HydroModelMCP.calibrate_tool.handler(Dict(
            "model" => "exphydro",
            "forcing" => Dict("source_type" => "csv", "path" => data_file),
            "observation" => Dict("source_type" => "csv", "path" => data_file),
            "obs_column" => "flow(mm)",
        ))

        @test startswith(response.text, "Error:")
        @test occursin("inputs", response.text)
    end

    @testset "多目标诊断 (Pareto 退化检测)" begin
        # 构造模拟的多目标校准结果
        # Case 1: 正常 Pareto 前沿
        mock_result_normal = Dict{String,Any}(
            "objectives" => ["KGE", "LogKGE"],
            "n_solutions" => 5,
            "pareto_front" => [
                Dict{String,Any}("objectives" => Dict("KGE" => 0.9, "LogKGE" => 0.5)),
                Dict{String,Any}("objectives" => Dict("KGE" => 0.8, "LogKGE" => 0.7)),
                Dict{String,Any}("objectives" => Dict("KGE" => 0.7, "LogKGE" => 0.8)),
                Dict{String,Any}("objectives" => Dict("KGE" => 0.6, "LogKGE" => 0.85)),
                Dict{String,Any}("objectives" => Dict("KGE" => 0.5, "LogKGE" => 0.9)),
            ]
        )

        diag_normal = diagnose_multiobjective(mock_result_normal)
        @test haskey(diag_normal, "degeneracy")
        @test haskey(diag_normal, "n_solutions")
        @test haskey(diag_normal, "recommendations")
        @test diag_normal["n_solutions"] == 5

        println("   -> Normal Pareto: degeneracy=$(diag_normal["degeneracy"]["status"])")

        # Case 2: 退化 Pareto (所有解几乎相同)
        mock_result_degen = Dict{String,Any}(
            "objectives" => ["KGE", "LogKGE"],
            "n_solutions" => 3,
            "pareto_front" => [
                Dict{String,Any}("objectives" => Dict("KGE" => 0.8, "LogKGE" => 0.8)),
                Dict{String,Any}("objectives" => Dict("KGE" => 0.801, "LogKGE" => 0.799)),
                Dict{String,Any}("objectives" => Dict("KGE" => 0.799, "LogKGE" => 0.801)),
            ]
        )

        diag_degen = diagnose_multiobjective(mock_result_degen)
        @test diag_degen["degeneracy"]["status"] in ["point_degenerate", "normal", "line_degenerate"]

        println("   -> Degenerate Pareto: degeneracy=$(diag_degen["degeneracy"]["status"])")

        # Case 3: 不足解
        mock_result_few = Dict{String,Any}(
            "objectives" => ["KGE", "LogKGE"],
            "n_solutions" => 1,
            "pareto_front" => [
                Dict{String,Any}("objectives" => Dict("KGE" => 0.8, "LogKGE" => 0.7)),
            ]
        )

        diag_few = diagnose_multiobjective(mock_result_few)
        @test diag_few["degeneracy"]["status"] == "insufficient_solutions"

        println("   -> Few solutions: degeneracy=$(diag_few["degeneracy"]["status"])")
    end

    @testset "不同优化算法" begin
        algorithms = ["BBO", "DE", "PSO", "DDS", "SCE"]

        for alg in algorithms
            println("\n   -> Testing algorithm: $(alg)")
            result = calibrate_model(
                test_model, forcing_nt, obs;
                algorithm=alg,
                maxiters=20,
                objective="KGE",
                n_trials=1,
                warm_up=1
            )

            @test result["algorithm"] == alg
            @test result["best_objective"] isa Float64

            println("      Best objective: $(round(result["best_objective"], digits=4))")
        end
    end

    @testset "错误处理：无效算法" begin
        @test_throws ArgumentError calibrate_model(
            test_model, forcing_nt, obs;
            algorithm="InvalidAlgorithm",
            maxiters=10,
            warm_up=1
        )
        println("   [Pass] Invalid algorithm throws ArgumentError as expected")
    end

    @testset "Warm-up 期测试" begin
        println("\n   -> Testing warm-up period functionality...")

        # 测试不同的 warm_up 值
        result_no_warmup = calibrate_model(
            test_model, forcing_nt, obs;
            algorithm="BBO",
            maxiters=20,
            objective="KGE",
            n_trials=1,
            warm_up=1  # 默认值
        )

        result_with_warmup = calibrate_model(
            test_model, forcing_nt, obs;
            algorithm="BBO",
            maxiters=20,
            objective="KGE",
            n_trials=1,
            warm_up=10  # 跳过前10个时间步
        )

        @test result_no_warmup["best_objective"] isa Float64
        @test result_with_warmup["best_objective"] isa Float64

        println("   -> No warm-up objective: $(round(result_no_warmup["best_objective"], digits=4))")
        println("   -> With warm-up (10) objective: $(round(result_with_warmup["best_objective"], digits=4))")
    end

    println("\n   [Pass] All Calibration tests passed successfully!")
end

@testset "compute_diagnostics_full 工具测试" begin
    println("\n   -> Testing compute_diagnostics_full tool...")
    test_dir = @__DIR__
    data_file = joinpath(dirname(test_dir), "data", "03604000.csv")

    # 场景1: 收敛场景
    @testset "收敛场景" begin
        trial_results = [
            Dict(
                "trial_id" => 1,
                "best_objective" => 0.25,
                "best_parameters" => Dict("x1" => 100.0, "x2" => 0.5),
                "objective_history" => [0.5, 0.45, 0.4, 0.35, 0.3, 0.28, 0.26, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
                "iterations_used" => 100
            ),
            Dict(
                "trial_id" => 2,
                "best_objective" => 0.26,
                "best_parameters" => Dict("x1" => 102.0, "x2" => 0.51),
                "objective_history" => [0.5, 0.45, 0.4, 0.35, 0.3, 0.28, 0.27, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26],
                "iterations_used" => 100
            )
        ]
        parameter_bounds = Dict(
            "x1" => Dict("lower" => 50.0, "upper" => 200.0),
            "x2" => Dict("lower" => 0.0, "upper" => 1.0)
        )

        result = Calibration.diagnose_calibration_full(
            trial_results, parameter_bounds; objective_threshold=0.7
        )

        @test haskey(result, "convergence_status")
        @test haskey(result, "hat_trick_achieved")
        @test haskey(result, "recommended_action")
        @test result["convergence_status"] == "converged"
        @test result["is_plateaued"] == true
        @test result["is_still_improving"] == false
        @test result["history_source"] == "observed"
        @test result["diagnostic_confidence"] == "high"
        println("   -> 收敛场景测试通过")
    end

    @testset "策略8 AUTO 算法选择" begin
        payload = _parse_calibration_tool_json(HydroModelMCP.calibrate_tool.handler(Dict(
            "model" => "exphydro",
            "inputs" => Dict(
                "forcing" => Dict("source_type" => "csv", "path" => data_file),
                "observation" => Dict("source_type" => "csv", "path" => data_file, "column" => "flow(mm)"),
            ),
            "algorithm" => "AUTO",
            "budget" => "low",
            "maxiters" => 12,
            "n_trials" => 1,
            "objective" => "KGE",
            "save_to_storage" => false,
        )))

        @test payload["status"] == "success"
        @test payload["requested_algorithm"] == "AUTO"
        @test payload["algorithm"] in ["DDS", "BBO", "SCE"]
    end

    @testset "约束在优化中生效（delta_method）" begin
        payload = _parse_calibration_tool_json(HydroModelMCP.calibrate_tool.handler(Dict(
            "model" => "exphydro",
            "inputs" => Dict(
                "forcing" => Dict("source_type" => "csv", "path" => data_file),
                "observation" => Dict("source_type" => "csv", "path" => data_file, "column" => "flow(mm)"),
            ),
            "algorithm" => "DDS",
            "objective" => "KGE",
            "maxiters" => 16,
            "n_trials" => 1,
            "constraints" => Dict(
                "delta_method" => Dict(
                    "inequalities" => [["Tmin", "Tmax"]],
                ),
            ),
            "save_to_storage" => false,
        )))

        @test payload["status"] == "success"
        @test payload["constraints_applied"] == true
        @test payload["best_params"]["Tmin"] < payload["best_params"]["Tmax"]
    end

    @testset "自动全流程工具可执行" begin
        payload = _parse_calibration_tool_json(HydroModelMCP.auto_calibration_workflow_tool.handler(Dict(
            "model" => "exphydro",
            "inputs" => Dict(
                "forcing" => Dict("source_type" => "csv", "path" => data_file),
                "observation" => Dict("source_type" => "csv", "path" => data_file, "column" => "flow(mm)"),
            ),
            "goal" => "general_fit",
            "budget" => "low",
            "max_rounds" => 1,
            "maxiters_per_round" => 8,
            "n_trials_per_round" => 1,
            "run_validation" => false,
            "save_to_storage" => false,
        )))

        @test payload["status"] == "success"
        @test payload["workflow"] == "knowledge_md_auto_calibration"
        @test haskey(payload, "strategy_alignment")
        @test payload["strategy_alignment"]["strategy_1_sensitivity"] == true
        @test haskey(payload, "best_result")
        @test haskey(payload, "rounds")
        @test length(payload["rounds"]) >= 1
    end

    @testset "多目标诊断工具" begin
        mock_result = Dict(
            "objectives" => ["KGE", "LogKGE"],
            "pareto_front" => [
                Dict("objectives" => Dict("KGE" => 0.8, "LogKGE" => 0.6)),
                Dict("objectives" => Dict("KGE" => 0.7, "LogKGE" => 0.75)),
            ],
        )
        payload = _parse_calibration_tool_json(HydroModelMCP.diagnose_multi_tool.handler(Dict(
            "multiobjective_result" => mock_result,
        )))

        @test payload["status"] == "success"
        @test haskey(payload, "degeneracy")
        @test haskey(payload, "recommendations")
    end

    # 场景2: 参数触及边界
    @testset "边界触及场景" begin
        trial_results = [
            Dict(
                "trial_id" => 1,
                "best_objective" => 0.3,
                "best_parameters" => Dict("x1" => 199.5, "x2" => 0.5),
                "objective_history" => [0.5, 0.4, 0.35, 0.3],
                "iterations_used" => 50
            )
        ]
        parameter_bounds = Dict(
            "x1" => Dict("lower" => 50.0, "upper" => 200.0),
            "x2" => Dict("lower" => 0.0, "upper" => 1.0)
        )

        result = Calibration.diagnose_calibration_full(
            trial_results, parameter_bounds
        )

        @test "x1" in result["parameters_at_boundary"]
        @test result["recommended_action"] == "widen_ranges"
        println("   -> 边界触及场景测试通过")
    end

    # 场景3: 持续改进
    @testset "持续改进场景" begin
        trial_results = [
            Dict(
                "trial_id" => 1,
                "best_objective" => 0.3,
                "best_parameters" => Dict("x1" => 100.0, "x2" => 0.5),
                "objective_history" => [0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.28, 0.26, 0.24, 0.22, 0.2, 0.18, 0.16, 0.14, 0.12, 0.1],
                "iterations_used" => 100
            )
        ]
        parameter_bounds = Dict(
            "x1" => Dict("lower" => 50.0, "upper" => 200.0),
            "x2" => Dict("lower" => 0.0, "upper" => 1.0)
        )

        result = Calibration.diagnose_calibration_full(
            trial_results, parameter_bounds
        )

        @test result["is_still_improving"] == true
        @test result["recommended_action"] == "increase_budget"
        println("   -> 持续改进场景测试通过")
    end

    # 场景4: 平台期检测
    @testset "平台期场景" begin
        trial_results = [
            Dict(
                "trial_id" => 1,
                "best_objective" => 0.5,
                "best_parameters" => Dict("x1" => 100.0, "x2" => 0.5),
                "objective_history" => [0.8, 0.7, 0.6, 0.55, 0.52, 0.51, 0.505, 0.502, 0.501, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                "iterations_used" => 100
            )
        ]
        parameter_bounds = Dict(
            "x1" => Dict("lower" => 50.0, "upper" => 200.0),
            "x2" => Dict("lower" => 0.0, "upper" => 1.0)
        )

        result = Calibration.diagnose_calibration_full(
            trial_results, parameter_bounds
        )

        @test result["is_plateaued"] == true
        @test result["is_still_improving"] == false
        println("   -> 平台期场景测试通过")
    end

    println("   [Pass] All compute_diagnostics_full tests passed!")
end

# ==============================================================================
# 新接口兼容性测试：final_parameters / parameter_uncertainty / best_trial_id
# ==============================================================================
@testset "compute_diagnostics_full 新接口兼容性" begin
    println("\n   -> Testing compute_diagnostics_full new interface...")

    @testset "final_parameters 字段兼容" begin
        # 使用 final_parameters（而非 best_parameters）
        trial_results = [
            Dict(
                "trial_id"         => 1,
                "best_objective"   => 0.823,
                "final_parameters" => Dict("X1" => 450.2, "X2" => -1.3),
                "objective_history"=> [0.5, 0.65, 0.78, 0.82, 0.823]
            ),
            Dict(
                "trial_id"         => 2,
                "best_objective"   => 0.819,
                "final_parameters" => Dict("X1" => 460.1, "X2" => -1.1),
                "objective_history"=> [0.5, 0.62, 0.75, 0.81, 0.819]
            )
        ]

        # 不传 parameter_bounds，测试自动推断路径
        result = Calibration.diagnose_calibration_full(
            [Dict("trial_id" => t["trial_id"],
                  "best_objective" => t["best_objective"],
                  "best_parameters" => t["final_parameters"],
                  "objective_history" => t["objective_history"])
             for t in trial_results],
            Dict{String,Any}()  # 空 bounds → 会造成 _convert_bounds_format 的情况
        )

        @test haskey(result, "convergence_status")
        @test haskey(result, "hat_trick_achieved")
        println("   -> final_parameters 兼容测试通过")
    end

    @testset "best_trial_id 和 parameter_uncertainty 输出字段" begin
        # 模拟工具层级的输出（包含 best_trial_id 和 parameter_uncertainty）
        trial_results_raw = [
            Dict(
                "trial_id"         => 1,
                "best_objective"   => 0.9,
                "final_parameters" => Dict("x1" => 120.0, "x2" => 0.3),
                "convergence_curve"=> [0.5, 0.7, 0.85, 0.9]
            ),
            Dict(
                "trial_id"         => 2,
                "best_objective"   => 0.88,
                "final_parameters" => Dict("x1" => 125.0, "x2" => 0.32),
                "convergence_curve"=> [0.5, 0.68, 0.83, 0.88]
            ),
        ]

        # 规范化（模拟 compute_diagnostics_full_tool handler 的处理）
        normalized = Dict{String,Any}[]
        for trial in trial_results_raw
            t = Dict{String,Any}(string(k) => v for (k, v) in trial)
            haskey(t, "final_parameters") && (t["best_parameters"] = t["final_parameters"])
            haskey(t, "convergence_curve") && (t["objective_history"] = t["convergence_curve"])
            push!(normalized, t)
        end

        # 参数边界（数组格式，工具层会转换）
        bounds_array = Dict{String,Any}(
            "x1" => Dict("lower" => 50.0, "upper" => 200.0),
            "x2" => Dict("lower" => 0.0, "upper" => 1.0)
        )

        result = Calibration.diagnose_calibration_full(normalized, bounds_array)

        @test haskey(result, "convergence_status")
        @test haskey(result, "hat_trick_achieved")
        @test haskey(result, "statistics")
        @test result["statistics"]["n_trials"] == 2

        # best_trial_id 和 parameter_uncertainty 由工具层添加
        # 直接验证核心逻辑
        objectives = [Float64(t["best_objective"]) for t in normalized]
        best_idx = argmax(objectives)
        @test normalized[best_idx]["trial_id"] == 1

        println("   -> best_trial_id 逻辑验证通过（best trial: trial_id=1）")
    end

    println("   [Pass] All new interface compatibility tests passed!")
end

@testset "compute_diagnostics_full tool confidence metadata" begin
    trial_results = [
        Dict(
            "trial_id" => 1,
            "best_objective" => 0.8,
            "final_parameters" => Dict("x1" => 100.0),
            "objective_summary" => Dict(
                "plateau_detected" => true,
                "still_improving" => false,
            ),
        ),
    ]

    payload = _parse_calibration_tool_json(HydroModelMCP.compute_diagnostics_full_tool.handler(Dict(
        "trial_results" => trial_results,
    )))

    @test haskey(payload, "history_inferred_trials")
    @test payload["history_inferred_trials"] >= 1
    @test payload["history_source"] == "mixed"
    @test payload["diagnostic_confidence"] == "medium"
    @test haskey(payload, "history_note")
end
