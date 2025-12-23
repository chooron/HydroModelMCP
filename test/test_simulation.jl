
using .HydroModelMCP.DataLoader
using .HydroModelMCP.Simulation

# =========================================================================
# 辅助函数：生成一些假的 Forcing 数据
# =========================================================================
function generate_dummy_forcing()
    # 简单的正弦波降雨和蒸发
    df = CSV.File("../data/03604000.csv") |> DataFrame
    # 注意：这里的键名 (prec, pet) 必须与你的水文模型 input names 对应
    # 假设模型是 HBV 或类似，通常需要 P (prec) 和 E (pet)
    return (P=df[!, "prcp(mm/day)"], T=df[!, "tmean(C)"], Ep=df[!, "pet(mm)"])
end

# 假设我们要测试的模型名称
const TEST_MODEL_NAME = "exphydro" # <--- 请确保你的 HydroModelLibrary 里有这个模型

# =========================================================================
# 测试套件开始
# =========================================================================

@testset "HydroMCP Core & Interface Tests" begin

    # 生成测试数据
    dummy_data = generate_dummy_forcing()
    # 构造一个符合模型参数数量的 dummy params (假设 params 长度为 10，请按需调整)
    dummy_params = Dict("f" => 0.001, "Smax" => 1000.0, "Qmax" => 20.0, "Df" => 2.5, "Tmax" => 1.5, "Tmin" => -1.0)

    # ---------------------------------------------------------------------
    # Test 1: JSON 格式 (最基础的内存数据测试)
    # 验证：核心计算是否成功，返回值是否直接包含结果
    # ---------------------------------------------------------------------
    @testset "Case 1: JSON Input (In-Memory)" begin
        println("\n>>> Testing JSON Input...")

        payload = Dict(
            "model_name" => TEST_MODEL_NAME,
            "params" => dummy_params,
            "forcing" => Dict(
                "source_type" => "json",
                "data" => Dict(
                    "P" => dummy_data.P,
                    "T" => dummy_data.T,
                    "Ep" => dummy_data.Ep
                )
            ),
            "config" => Dict("output_variable" => "Qt") # 假设想获取总径流
        )

        # 调用接口
        response = Simulation.run_simulation(payload)

        # 验证
        @test response["source_type"] == "json"
        @test haskey(response, "result")
        @test response["result"] isa AbstractVector
        @test length(response["result"]) == length(dummy_data.P)
        println("   [Pass] JSON simulation successful. Result length: $(length(response["result"]))")
    end

    # ---------------------------------------------------------------------
    # Test 2: CSV 格式 (文件 I/O 测试)
    # 验证：输入是 CSV 路径，输出是否生成了新的 CSV 文件
    # ---------------------------------------------------------------------
    @testset "Case 2: CSV Input (File I/O)" begin
        println("\n>>> Testing CSV Input...")

        # 1. 准备临时 CSV 文件
        temp_dir = mktempdir()
        csv_path = joinpath(temp_dir, "test_forcing.csv")

        df = DataFrame(P=dummy_data.P, Ep=dummy_data.Ep, T=dummy_data.T)
        CSV.write(csv_path, df)

        payload = Dict(
            "model_name" => TEST_MODEL_NAME,
            "params" => dummy_params,
            "forcing" => Dict(
                "source_type" => "csv",
                "path" => csv_path
            )
        )

        # 2. 调用接口
        response = Simulation.run_simulation(payload)

        # 3. 验证
        @test response["source_type"] == "csv"
        @test haskey(response, "path")
        output_path = response["path"]

        println("   -> Output generated at: $output_path")
        @test isfile(output_path)
        @test output_path != csv_path # 确保没有覆盖原文件

        # 检查输出文件内容
        out_df = CSV.read(output_path, DataFrame)
        @test nrow(out_df) == length(dummy_data.P)
        println("   [Pass] CSV simulation successful. Output file verifyied.")

        # 清理
        rm(temp_dir, recursive=true, force=true)
    end

    # # ---------------------------------------------------------------------
    # # Test 3: Redis 格式 (缓存 I/O 测试)
    # # 验证：输入 Key，输出是否写入了新 Key
    # # 注意：如果没有本地 Redis，此测试会被 Skip 或报错
    # # ---------------------------------------------------------------------
    # @testset "Case 3: Redis Input (Cache I/O)" begin
    #     println("\n>>> Testing Redis Input...")

    #     # 尝试连接 Redis，如果失败则跳过测试
    #     try
    #         conn = Redis.RedisConnection(host="127.0.0.1", port=6379)
    #         if !isopen(conn)
    #             error("Redis not connected")
    #         end

    #         # 1. 准备数据
    #         input_key = "test_hydro_forcing"
    #         input_json = JSON3.write(Dict(
    #             "P" => dummy_data.P,
    #             "T" => dummy_data.T,
    #             "Ep" => dummy_data.Ep
    #         ))
    #         Redis.set(conn, input_key, input_json)

    #         payload = Dict(
    #             "model_name" => TEST_MODEL_NAME,
    #             "params" => dummy_params,
    #             "forcing" => Dict(
    #                 "source_type" => "redis",
    #                 "key" => input_key,
    #                 "host" => "127.0.0.1",
    #                 "port" => 6379
    #             )
    #         )

    #         # 2. 调用接口
    #         response = Simulation.run_simulation(payload)

    #         # 3. 验证
    #         @test response["source_type"] == "redis"
    #         output_key = response["key"]
    #         println("   -> Output stored in Redis key: $output_key")

    #         # 读取结果验证
    #         res_str = Redis.get(conn, output_key)
    #         res_json = JSON3.read(res_str)
    #         @test haskey(res_json, "result")
    #         @test length(res_json["result"]) == length(dummy_data.P)

    #         println("   [Pass] Redis simulation successful.")

    #         # 清理
    #         Redis.del(conn, input_key)
    #         Redis.del(conn, output_key)
    #         close(conn)

    #     catch e
    #         @warn "Redis 测试跳过或失败: 无法连接到本地 Redis 服务 (127.0.0.1:6379)。" exception = e
    #     end
    # end

    # # ---------------------------------------------------------------------
    # # Test 4: 鲁棒性测试 (缺省参数)
    # # 验证：不传 params 是否会自动随机生成，配置是否生效
    # # ---------------------------------------------------------------------
    # @testset "Case 4: Robustness (Missing Params)" begin
    #     println("\n>>> Testing Robustness...")

    #     # 不传 "params"，测试自动随机初始化
    #     payload = Dict(
    #         "model_name" => TEST_MODEL_NAME,
    #         "forcing" => Dict(
    #             "source_type" => "json",
    #             "data" => Dict(
    #                 "P" => dummy_data.P,
    #                 "T" => dummy_data.T,
    #                 "Ep" => dummy_data.Ep
    #             )
    #         )
    #     )

    #     response = Simulation.run_simulation(payload)
    #     @test length(response["result"]) == length(dummy_data.P)
    #     println("   [Pass] Auto-random parameter generation worked.")
    # end

end