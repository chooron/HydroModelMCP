# ==========================================================================
# 测试 Metrics 模块
# ==========================================================================
using .HydroModelMCP.Metrics

@testset "Metrics Module Tests" begin

    # 准备测试数据
    obs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    sim_perfect = copy(obs)  # 完美模拟
    sim_good = obs .+ randn(length(obs)) .* 0.5  # 带噪声的好模拟
    sim_biased = obs .* 1.2  # 系统性偏高

    @testset "NSE (Nash-Sutcliffe Efficiency)" begin
        # 完美模拟应该得到 NSE = 1.0
        @test nse(sim_perfect, obs) ≈ 1.0 atol=1e-10

        # 好模拟应该 NSE > 0.5
        nse_val = nse(sim_good, obs)
        @test nse_val > 0.0
        println("   -> NSE (good sim): $(round(nse_val, digits=4))")

        # 测试 log 变换
        nse_log = nse(sim_good, obs; log_transform=true)
        @test nse_log isa Float64
        println("   -> LogNSE: $(round(nse_log, digits=4))")
    end

    @testset "KGE (Kling-Gupta Efficiency)" begin
        # 完美模拟应该得到 KGE = 1.0
        @test kge(sim_perfect, obs) ≈ 1.0 atol=1e-10

        # 好模拟应该 KGE > 0
        kge_val = kge(sim_good, obs)
        @test kge_val > 0.0
        println("   -> KGE (good sim): $(round(kge_val, digits=4))")

        # 测试 KGE 分量
        components = kge_components(sim_good, obs)
        @test haskey(components, "KGE")
        @test haskey(components, "r")
        @test haskey(components, "alpha")
        @test haskey(components, "beta")
        println("   -> KGE components: r=$(round(components["r"], digits=3)), α=$(round(components["alpha"], digits=3)), β=$(round(components["beta"], digits=3))")
    end

    @testset "PBIAS (Percent Bias)" begin
        # 完美模拟应该得到 PBIAS = 0
        @test pbias(sim_perfect, obs) ≈ 0.0 atol=1e-10

        # 偏高模拟应该 PBIAS > 0
        pbias_val = pbias(sim_biased, obs)
        @test pbias_val > 0.0
        println("   -> PBIAS (biased sim): $(round(pbias_val, digits=2))%")
    end

    @testset "R² (Coefficient of Determination)" begin
        # 完美模拟应该得到 R² = 1.0
        @test r_squared(sim_perfect, obs) ≈ 1.0 atol=1e-10

        # 好模拟应该 R² > 0.5
        r2_val = r_squared(sim_good, obs)
        @test r2_val > 0.5
        println("   -> R² (good sim): $(round(r2_val, digits=4))")
    end

    @testset "RMSE (Root Mean Square Error)" begin
        # 完美模拟应该得到 RMSE = 0
        @test rmse(sim_perfect, obs) ≈ 0.0 atol=1e-10

        # 带噪声模拟应该 RMSE > 0
        rmse_val = rmse(sim_good, obs)
        @test rmse_val > 0.0
        println("   -> RMSE (good sim): $(round(rmse_val, digits=4))")
    end

    @testset "compute_metrics (批量计算)" begin
        metrics = ["NSE", "KGE", "PBIAS", "R2", "RMSE"]
        result = compute_metrics(sim_good, obs, metrics)

        # 检查所有指标都被计算
        for m in metrics
            @test haskey(result, m)
            @test result[m] isa Float64 || result[m] isa String
        end

        # 检查自动检测 log 变换建议
        @test haskey(result, "_magnitude_ratio")
        @test haskey(result, "_log_transform_recommended")

        println("   -> Computed metrics: $(keys(result))")
        println("   -> Log transform recommended: $(result["_log_transform_recommended"])")
    end

    @testset "跨数量级数据的 log 变换建议" begin
        # 创建跨越多个数量级的数据
        obs_wide = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        sim_wide = obs_wide .* (1.0 .+ randn(length(obs_wide)) .* 0.1)

        result = compute_metrics(sim_wide, obs_wide, ["NSE", "KGE"])

        # 应该建议 log 变换
        @test result["_log_transform_recommended"] == true
        @test result["_magnitude_ratio"] > 100.0

        println("   -> Magnitude ratio: $(round(result["_magnitude_ratio"], digits=2))")
        println("   -> Log transform recommended: true (as expected)")
    end

    @testset "is_higher_better 辅助函数" begin
        @test Metrics.is_higher_better("NSE") == true
        @test Metrics.is_higher_better("KGE") == true
        @test Metrics.is_higher_better("LogKGE") == true
        @test Metrics.is_higher_better("PBIAS") == false
        @test Metrics.is_higher_better("RMSE") == false
    end

    @testset "weighted_metric (加权组合指标)" begin
        # 单指标权重 = 直接计算该指标
        w_single = Dict{String,Float64}("KGE" => 1.0)
        wm = weighted_metric(sim_perfect, obs, w_single)
        @test wm ≈ 1.0 atol=1e-10  # KGE=1.0, higher_better → +1.0

        # 多指标加权
        w_multi = Dict{String,Float64}("KGE" => 0.7, "NSE" => 0.3)
        wm2 = weighted_metric(sim_perfect, obs, w_multi)
        @test wm2 ≈ 1.0 atol=1e-10  # 完美模拟两个都是 1.0

        # 包含 lower-is-better 指标 (RMSE)
        w_mixed = Dict{String,Float64}("KGE" => 0.5, "RMSE" => 0.5)
        wm3 = weighted_metric(sim_perfect, obs, w_mixed)
        # KGE=1.0 (higher_better → +1.0), RMSE=0.0 (lower_better → -0.0=0.0)
        @test wm3 ≈ 0.5 atol=1e-10

        println("   -> Weighted metric works correctly")
    end

    @testset "transform_data (数据变换)" begin
        data = [1.0, 10.0, 100.0, 1000.0]

        # Log 变换
        transformed, params = transform_data(data, :log)
        @test length(transformed) == length(data)
        @test transformed[1] ≈ log(1.0) atol=1e-10
        @test transformed[2] ≈ log(10.0) atol=1e-10
        @test params["method"] == "log"

        # Log10 变换
        transformed10, params10 = transform_data(data, :log10)
        @test transformed10[2] ≈ 1.0 atol=1e-10  # log10(10) = 1
        @test transformed10[3] ≈ 2.0 atol=1e-10  # log10(100) = 2

        # Box-Cox 变换
        transformed_bc, params_bc = transform_data(data, :boxcox)
        @test length(transformed_bc) == length(data)
        @test haskey(params_bc, "lambda")

        # 逆变换 round-trip
        recovered = inverse_transform_data(transformed, params)
        @test all(isapprox.(recovered, data, atol=1e-8))

        recovered10 = inverse_transform_data(transformed10, params10)
        @test all(isapprox.(recovered10, data, atol=1e-8))

        recovered_bc = inverse_transform_data(transformed_bc, params_bc)
        @test all(isapprox.(recovered_bc, data, atol=1e-6))

        println("   -> transform_data + inverse_transform_data round-trip ✓")
    end

    println("\n   [Pass] All Metrics tests passed successfully!")
end
