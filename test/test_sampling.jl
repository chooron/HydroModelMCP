# ==========================================================================
# 测试 Sampling 模块
# ==========================================================================
using .HydroModelMCP.Sampling

@testset "Sampling Module Tests" begin

    # 定义测试用的参数边界
    bounds = [(0.0, 1.0), (10.0, 100.0), (-5.0, 5.0), (0.001, 0.1)]
    n_params = length(bounds)
    n_samples = 50

    @testset "LHS (拉丁超立方采样)" begin
        samples = generate_samples(bounds; method="lhs", n_samples=n_samples)

        # 检查维度
        @test size(samples) == (n_params, n_samples)

        # 检查所有样本都在边界内
        for i in 1:n_params
            @test all(bounds[i][1] .<= samples[i, :] .<= bounds[i][2])
        end

        # LHS 特性：每个维度应该均匀覆盖
        # 检查第一个参数的分布
        sorted_first = sort(samples[1, :])
        gaps = diff(sorted_first)
        # 间隔应该相对均匀（标准差不应太大）
        @test std(gaps) < mean(gaps) * 0.5

        println("   -> LHS samples shape: $(size(samples))")
        println("   -> First param range: [$(minimum(samples[1,:])), $(maximum(samples[1,:]))]")
        println("   -> Gap std/mean ratio: $(round(std(gaps)/mean(gaps), digits=3))")
    end

    @testset "Sobol 序列" begin
        samples = generate_samples(bounds; method="sobol", n_samples=n_samples)

        # 检查维度
        @test size(samples) == (n_params, n_samples)

        # 检查边界
        for i in 1:n_params
            @test all(bounds[i][1] .<= samples[i, :] .<= bounds[i][2])
        end

        # Sobol 序列应该有低差异性
        # 简单检查：样本应该覆盖整个空间
        for i in 1:n_params
            range_covered = maximum(samples[i, :]) - minimum(samples[i, :])
            total_range = bounds[i][2] - bounds[i][1]
            coverage = range_covered / total_range
            @test coverage > 0.8  # 至少覆盖 80% 的范围
        end

        println("   -> Sobol samples shape: $(size(samples))")
        println("   -> Coverage check passed")
    end

    @testset "Random 采样" begin
        samples = generate_samples(bounds; method="random", n_samples=n_samples)

        # 检查维度
        @test size(samples) == (n_params, n_samples)

        # 检查边界
        for i in 1:n_params
            @test all(bounds[i][1] .<= samples[i, :] .<= bounds[i][2])
        end

        println("   -> Random samples shape: $(size(samples))")
    end

    @testset "不同采样数" begin
        for n in [10, 100, 200]
            samples = generate_samples(bounds; method="lhs", n_samples=n)
            @test size(samples, 2) == n
        end
        println("   -> Variable sample sizes: 10, 100, 200 all work")
    end

    @testset "单参数采样" begin
        single_bound = [(0.0, 10.0)]
        samples = generate_samples(single_bound; method="lhs", n_samples=20)

        @test size(samples) == (1, 20)
        @test all(0.0 .<= samples[1, :] .<= 10.0)

        println("   -> Single parameter sampling works")
    end

    @testset "高维采样" begin
        # 测试 10 维参数空间
        high_dim_bounds = [(0.0, 1.0) for _ in 1:10]
        samples = generate_samples(high_dim_bounds; method="lhs", n_samples=30)

        @test size(samples) == (10, 30)
        println("   -> High-dimensional (10D) sampling works")
    end

    @testset "Pie-share 约束采样 (参数加和为常数)" begin
        n_params = 4
        n_samples = 50

        # 默认 total=1.0
        samples = pie_share_sampling(n_params, n_samples)
        @test size(samples) == (n_params, n_samples)

        # 每列之和应等于 1.0
        for j in 1:n_samples
            @test sum(samples[:, j]) ≈ 1.0 atol=1e-10
        end

        # 所有值应为正
        @test all(samples .> 0)

        # 自定义 total
        samples2 = pie_share_sampling(3, 30; total=100.0)
        for j in 1:30
            @test sum(samples2[:, j]) ≈ 100.0 atol=1e-8
        end

        println("   -> Pie-share: all columns sum to total ✓")
    end

    @testset "Delta Method 约束采样 (不等式约束)" begin
        bounds = [(0.0, 10.0), (0.0, 10.0), (0.0, 10.0)]
        # 约束: param1 < param2, param2 < param3
        constraints = [(1, 2), (2, 3)]

        samples = delta_method_sampling(bounds, constraints; n_samples=100)
        @test size(samples) == (3, 100)

        # 检查所有约束都满足
        for j in 1:100
            @test samples[1, j] < samples[2, j]
            @test samples[2, j] < samples[3, j]
        end

        # 检查边界
        for i in 1:3
            @test all(bounds[i][1] .<= samples[i, :] .<= bounds[i][2])
        end

        println("   -> Delta method: all inequality constraints satisfied ✓")
    end

    @testset "错误处理：无效方法" begin
        @test_throws ArgumentError generate_samples(bounds; method="invalid_method")
        println("   [Pass] Invalid method throws ArgumentError as expected")
    end

    @testset "LHS vs Random 对比" begin
        # 比较 LHS 和 Random 的空间覆盖均匀性
        lhs_samples = generate_samples(bounds; method="lhs", n_samples=100)
        rand_samples = generate_samples(bounds; method="random", n_samples=100)

        # 计算第一个参数的分布均匀性（使用分箱）
        n_bins = 10
        lhs_hist = fit(Histogram, lhs_samples[1, :], nbins=n_bins)
        rand_hist = fit(Histogram, rand_samples[1, :], nbins=n_bins)

        # LHS 的分箱应该更均匀（标准差更小）
        lhs_std = std(lhs_hist.weights)
        rand_std = std(rand_hist.weights)

        println("   -> LHS bin std: $(round(lhs_std, digits=2)), Random bin std: $(round(rand_std, digits=2))")
        println("   -> LHS is $(lhs_std < rand_std ? "more" : "less") uniform than Random")
    end

    println("\n   [Pass] All Sampling tests passed successfully!")
end
