
using CSV
using DataFrames
using .HydroModelMCP.Ensemble
using .HydroModelMCP.DataLoader

# =========================================================================
# 辅助函数：生成测试数据
# =========================================================================
function generate_ensemble_test_data()
    df = CSV.File("../data/03604000.csv") |> DataFrame
    return (P=df[!, "prcp(mm/day)"], T=df[!, "tmean(C)"], Ep=df[!, "pet(mm)"])
end

# 测试模型
const TEST_MODEL_NAME = "exphydro"

# =========================================================================
# 集合模拟测试
# =========================================================================

@testset "Ensemble Simulation Tests" begin

    # 生成测试数据
    test_data = generate_ensemble_test_data()

    # 准备多个参数集
    param_sets = [
        Dict("f" => 0.001, "Smax" => 1000.0, "Qmax" => 20.0, "Df" => 2.5, "Tmax" => 1.5, "Tmin" => -1.0),
        Dict("f" => 0.002, "Smax" => 1200.0, "Qmax" => 22.0, "Df" => 2.8, "Tmax" => 1.8, "Tmin" => -0.8),
        Dict("f" => 0.0015, "Smax" => 1100.0, "Qmax" => 21.0, "Df" => 2.6, "Tmax" => 1.6, "Tmin" => -0.9)
    ]

    # ---------------------------------------------------------------------
    # Test 1: JSON 格式集合模拟
    # ---------------------------------------------------------------------
    @testset "Case 1: Ensemble with JSON Input" begin
        println("\n>>> Testing Ensemble with JSON Input...")

        payload = Dict(
            "model_name" => TEST_MODEL_NAME,
            "parameter_sets" => param_sets,
            "forcing" => Dict(
                "source_type" => "json",
                "data" => Dict(
                    "P" => test_data.P,
                    "T" => test_data.T,
                    "Ep" => test_data.Ep
                )
            ),
            "parallel" => false,  # 串行执行以便调试
            "save_to_storage" => false
        )

        # 调用集合模拟
        response = Ensemble.run_ensemble(payload)

        # 验证结果
        @test haskey(response, "ensemble_members")
        @test length(response["ensemble_members"]) == length(param_sets)
        @test haskey(response, "ensemble_mean")
        @test haskey(response, "ensemble_std")
        @test length(response["ensemble_mean"]) == length(test_data.P)
        @test length(response["ensemble_std"]) == length(test_data.P)

        println("   [Pass] Ensemble simulation successful.")
        println("   -> Number of members: $(length(response["ensemble_members"]))")
        println("   -> Result length: $(length(response["ensemble_mean"]))")
    end

    # ---------------------------------------------------------------------
    # Test 2: 并行执行测试
    # ---------------------------------------------------------------------
    @testset "Case 2: Parallel Ensemble Execution" begin
        println("\n>>> Testing Parallel Ensemble Execution...")

        payload = Dict(
            "model_name" => TEST_MODEL_NAME,
            "parameter_sets" => param_sets,
            "forcing" => Dict(
                "source_type" => "json",
                "data" => Dict(
                    "P" => test_data.P,
                    "T" => test_data.T,
                    "Ep" => test_data.Ep
                )
            ),
            "parallel" => true,
            "save_to_storage" => false
        )

        response = Ensemble.run_ensemble(payload)

        @test haskey(response, "ensemble_members")
        @test length(response["ensemble_members"]) == length(param_sets)
        @test haskey(response, "ensemble_mean")
        @test haskey(response, "ensemble_std")

        println("   [Pass] Parallel ensemble execution successful.")
    end

    # ---------------------------------------------------------------------
    # Test 3: CSV 格式集合模拟
    # ---------------------------------------------------------------------
    @testset "Case 3: Ensemble with CSV Input" begin
        println("\n>>> Testing Ensemble with CSV Input...")

        # 准备临时 CSV 文件
        temp_dir = mktempdir()
        csv_path = joinpath(temp_dir, "test_forcing.csv")

        df = DataFrame(P=test_data.P, Ep=test_data.Ep, T=test_data.T)
        CSV.write(csv_path, df)

        payload = Dict(
            "model_name" => TEST_MODEL_NAME,
            "parameter_sets" => param_sets,
            "forcing" => Dict(
                "source_type" => "csv",
                "path" => csv_path
            ),
            "parallel" => false,
            "save_to_storage" => false
        )

        response = Ensemble.run_ensemble(payload)

        @test haskey(response, "ensemble_members")
        @test length(response["ensemble_members"]) == length(param_sets)

        println("   [Pass] CSV ensemble simulation successful.")

        # 清理
        rm(temp_dir, recursive=true, force=true)
    end

    # ---------------------------------------------------------------------
    # Test 4: 单个参数集（边界情况）
    # ---------------------------------------------------------------------
    @testset "Case 4: Single Parameter Set" begin
        println("\n>>> Testing Single Parameter Set...")

        payload = Dict(
            "model_name" => TEST_MODEL_NAME,
            "parameter_sets" => [param_sets[1]],
            "forcing" => Dict(
                "source_type" => "json",
                "data" => Dict(
                    "P" => test_data.P,
                    "T" => test_data.T,
                    "Ep" => test_data.Ep
                )
            ),
            "save_to_storage" => false
        )

        response = Ensemble.run_ensemble(payload)

        @test length(response["ensemble_members"]) == 1
        @test haskey(response, "ensemble_mean")

        println("   [Pass] Single parameter set handled correctly.")
    end

end
