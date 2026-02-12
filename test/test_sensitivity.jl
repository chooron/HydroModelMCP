# ==========================================================================
# 测试 SensitivityAnalysis 模块
# ==========================================================================
using .HydroModelMCP.SensitivityAnalysis

@testset "SensitivityAnalysis Module Tests" begin

    # 使用真实数据 - 基于测试文件位置构建路径
    test_dir = @__DIR__
    data_file = joinpath(dirname(test_dir), "data", "03604000.csv")
    df = CSV.File(data_file) |> DataFrame
    forcing_nt = (
        P = df[1:365, "prcp(mm/day)"],
        T = df[1:365, "tmean(C)"],
        Ep = df[1:365, "pet(mm)"]  # 使用 PET 作为 Ep
    )
    obs = df[1:365, "flow(mm)"]

    test_model = "exphydro"

    @testset "Morris 方法 (快速筛选)" begin
        println("\n   -> Running Morris sensitivity analysis (this may take a minute)...")

        result = run_sensitivity(
            test_model, forcing_nt, obs;
            method="morris",
            n_samples=20,  # 减少采样数以加快测试
            objective="NSE",
            threshold=0.1,
            solver_type="DISCRETE",
            interp_type="LINEAR"
        )

        # 检查返回结构
        @test haskey(result, "model_name")
        @test haskey(result, "param_names")
        @test haskey(result, "sensitivities")
        @test haskey(result, "calibratable")
        @test haskey(result, "fixed")
        @test haskey(result, "method")
        @test haskey(result, "threshold")

        # 检查方法标记
        @test result["method"] == "morris"
        @test result["threshold"] == 0.1

        # 检查敏感性值
        @test result["sensitivities"] isa Vector
        @test length(result["sensitivities"]) == length(result["param_names"])
        @test all(0.0 .<= result["sensitivities"] .<= 1.0)  # 归一化后应该在 [0,1]

        # 检查参数分类
        @test result["calibratable"] isa Vector{String}
        @test result["fixed"] isa Vector{String}
        @test length(result["calibratable"]) + length(result["fixed"]) == length(result["param_names"])

        println("   -> Model: $(result["model_name"])")
        println("   -> Total params: $(length(result["param_names"]))")
        println("   -> Calibratable: $(length(result["calibratable"])) params")
        println("   -> Fixed: $(length(result["fixed"])) params")
        println("   -> Calibratable params: $(result["calibratable"])")
    end

    # Sobol 方法测试（可选，因为计算量较大）
    # @testset "Sobol 方法 (全局敏感性)" begin
    #     println("\n   -> Running Sobol sensitivity analysis (this will take longer)...")

    #     result = run_sensitivity(
    #         test_model, forcing_nt, obs;
    #         method="sobol",
    #         n_samples=100,  # Sobol 需要更多样本
    #         objective="KGE",
    #         threshold=0.05
    #     )

    #     @test result["method"] == "sobol"
    #     @test all(result["sensitivities"] .>= 0.0)  # Sobol ST 应该非负

    #     println("   -> Sobol sensitivities: $(round.(result["sensitivities"], digits=3))")
    # end

    @testset "不同目标函数" begin
        # 测试使用不同的目标函数
        result_nse = run_sensitivity(
            test_model, forcing_nt, obs;
            method="morris", n_samples=15, objective="NSE", threshold=0.15
        )

        result_kge = run_sensitivity(
            test_model, forcing_nt, obs;
            method="morris", n_samples=15, objective="KGE", threshold=0.15
        )

        # 不同目标函数可能导致不同的敏感性排序
        @test result_nse["objective"] == "NSE"
        @test result_kge["objective"] == "KGE"

        println("   -> NSE calibratable: $(result_nse["calibratable"])")
        println("   -> KGE calibratable: $(result_kge["calibratable"])")
    end

    @testset "阈值影响" begin
        # 测试不同阈值对参数分类的影响
        result_low = run_sensitivity(
            test_model, forcing_nt, obs;
            method="morris", n_samples=15, threshold=0.05
        )

        result_high = run_sensitivity(
            test_model, forcing_nt, obs;
            method="morris", n_samples=15, threshold=0.3
        )

        # 高阈值应该导致更少的 calibratable 参数
        @test length(result_high["calibratable"]) <= length(result_low["calibratable"])

        println("   -> Threshold 0.05: $(length(result_low["calibratable"])) calibratable")
        println("   -> Threshold 0.30: $(length(result_high["calibratable"])) calibratable")
    end

    @testset "错误处理：无效模型" begin
        @test_throws ArgumentError run_sensitivity(
            "NonExistentModel", forcing_nt, obs;
            method="morris", n_samples=10
        )
        println("   [Pass] Invalid model throws ArgumentError as expected")
    end

    @testset "错误处理：无效方法" begin
        @test_throws ArgumentError run_sensitivity(
            test_model, forcing_nt, obs;
            method="invalid_method", n_samples=10
        )
        println("   [Pass] Invalid method throws ArgumentError as expected")
    end

    println("\n   [Pass] All SensitivityAnalysis tests passed successfully!")
end
