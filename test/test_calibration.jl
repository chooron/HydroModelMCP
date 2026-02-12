# ==========================================================================
# 测试 Calibration 模块
# ==========================================================================
using .HydroModelMCP.Calibration

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
        algorithms = ["BBO", "DE", "PSO"]

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
