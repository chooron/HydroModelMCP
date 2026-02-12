# ==========================================================================
# 测试 DataSplitter 模块
# ==========================================================================
using .HydroModelMCP.DataSplitter

@testset "DataSplitter Module Tests" begin

    # 准备测试数据：模拟 3 年的日数据 (1095 天)
    n_days = 1095
    obs = rand(n_days) .* 10.0  # 随机观测值
    forcing_nt = (
        P = rand(n_days) .* 20.0,
        T = randn(n_days) .* 5.0 .+ 10.0,
        Ep = rand(n_days) .* 5.0
    )

    @testset "split_sample 方法 (经典前后分割)" begin
        result = split_data(obs, forcing_nt; method="split_sample", ratio=0.7, warmup=365)

        # 检查返回结构
        @test haskey(result, "cal_obs")
        @test haskey(result, "val_obs")
        @test haskey(result, "cal_forcing")
        @test haskey(result, "val_forcing")
        @test haskey(result, "method")
        @test haskey(result, "warmup")

        # 检查方法标记
        @test result["method"] == "split_sample"
        @test result["warmup"] == min(365, div(n_days, 4))  # warmup is capped at n/4

        # 检查长度
        @test result["cal_length"] + result["val_length"] + result["warmup"] == n_days
        @test result["cal_length"] > 0
        @test result["val_length"] > 0

        # 检查 forcing NamedTuple 结构
        @test result["cal_forcing"] isa NamedTuple
        @test haskey(result["cal_forcing"], :P)
        @test length(result["cal_forcing"].P) == result["cal_length"]

        println("   -> Split: cal=$(result["cal_length"]) days, val=$(result["val_length"]) days, warmup=$(result["warmup"]) days")
    end

    @testset "recent_first 方法 (最新数据优先)" begin
        result = split_data(obs, forcing_nt; method="recent_first", ratio=0.7, warmup=100)

        @test result["method"] == "recent_first"
        @test result["warmup"] == 100

        # 最新数据用于校准，所以 cal_indices 应该在后面
        @test result["cal_indices"][1] > result["val_indices"][1]

        println("   -> Recent-first split: cal indices=$(result["cal_indices"]), val indices=$(result["val_indices"])")
    end

    @testset "use_all 方法 (全部用于校准)" begin
        result = split_data(obs, forcing_nt; method="use_all", warmup=200)

        @test result["method"] == "use_all"
        @test result["val_length"] == 0
        @test isempty(result["val_obs"])
        @test isnothing(result["val_forcing"])

        println("   -> Use-all: cal=$(result["cal_length"]) days, no validation set")
    end

    @testset "自定义 ratio 和 warmup" begin
        # 测试不同的 ratio
        result_80 = split_data(obs, forcing_nt; method="split_sample", ratio=0.8, warmup=100)
        result_60 = split_data(obs, forcing_nt; method="split_sample", ratio=0.6, warmup=100)

        @test result_80["cal_length"] > result_60["cal_length"]
        @test result_80["val_length"] < result_60["val_length"]

        println("   -> Ratio 0.8: cal=$(result_80["cal_length"]), val=$(result_80["val_length"])")
        println("   -> Ratio 0.6: cal=$(result_60["cal_length"]), val=$(result_60["val_length"])")
    end

    @testset "边界情况：短数据" begin
        short_obs = rand(100)
        short_forcing = (P=rand(100), T=rand(100), Ep=rand(100))

        # warmup 会被自动限制为 n/4
        result = split_data(short_obs, short_forcing; method="split_sample", warmup=50)

        @test result["warmup"] <= 25  # 100/4 = 25
        @test result["cal_length"] + result["val_length"] + result["warmup"] == 100

        println("   -> Short data (100 days): warmup auto-capped to $(result["warmup"])")
    end

    @testset "错误处理：无效方法" begin
        @test_throws ArgumentError split_data(obs, forcing_nt; method="invalid_method")
        println("   [Pass] Invalid method throws ArgumentError as expected")
    end

    println("\n   [Pass] All DataSplitter tests passed successfully!")
end
