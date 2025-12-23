# ==========================================================================
# 1. 引入待测模块
# ==========================================================================
# 注意：这里我们使用 include 方式，或者确保 Discovery 在 LOAD_PATH 中
using .HydroModelMCP.Discovery

# ==========================================================================
# 2. 开始测试
# ==========================================================================

@testset "Discovery Module Tests" begin

    @testset "list_models" begin
        models = list_models()

        # 1. 测试返回类型
        @test models isa Vector{String}

        # 2. 打印一下实际列表，方便调试
        println("   -> Available models: $(first(models, 5))...")

        # 3. 修正：根据你的报错，库里确实有 "exphydro" 和 "gr4j"，且是小写
        @test "exphydro" in models
        @test "gr4j" in models

        # 确保列表不为空
        @test !isempty(models)
    end

    @testset "find_model (Case Insensitive)" begin
        # 1. 修正：期望值改为小写 "exphydro"，因为库里存的就是小写
        # 测试：输入 "exphydro" -> 返回 "exphydro"
        @test find_model("exphydro") == "exphydro"

        # 2. 测试大小写不敏感匹配
        # 输入混合大小写 "ExpHydro" -> 依然应该找到并返回库里的标准名 "exphydro"
        @test find_model("ExpHydro") == "exphydro"

        # 3. 测试 HBV (报错显示库里是 "hbv")
        # 输入 "HbV" -> 返回 "hbv"
        @test find_model("HbV") == "hbv"

        # 4. 测试不存在的模型
        @test isnothing(find_model("SuperModel_2077"))

        println("   [Pass] find_model matches regardless of case input.")
    end

    @testset "get_model_info" begin
        # 1. 测试存在的模型
        model_name = "hbv"
        info = get_model_info(model_name)

        # 检查返回结构
        @test info isa Dict
        @test info["model_name"] == "hbv" # 应该返回标准名

        # 检查元数据是否提取成功
        @test "P" in info["inputs"]
        @test "SM" in info["states"]
        @test "TT" in info["params"]
        @test "Qt" in info["outputs"]

        # 2. 测试不存在的模型 (应报错)
        @test_throws ArgumentError get_model_info("NonExistentModel")
        println("   [Pass] get_model_info threw error for invalid model as expected.")
    end

    @testset "Variable & Parameter Details" begin
        # 选择一个肯定存在的模型名称进行测试
        # 根据之前的报错信息，"exphydro" 或 "gr4j" 肯定是存在的
        # 如果你已经添加了 HBV，也可以改成 "hbv"
        test_model_name = "exphydro"

        # 1. 测试 get_variables_detail
        println("\n   -> Testing get_variables_detail for '$test_model_name'...")
        vars_info = get_variables_detail(test_model_name)

        @test vars_info isa Vector
        @test !isempty(vars_info)

        # 检查第一项的结构
        first_var = vars_info[1]
        @test first_var isa Dict
        @test haskey(first_var, "name")
        @test haskey(first_var, "description")
        @test haskey(first_var, "unit")
        @test first_var["type"] == "variable"

        # 打印出来看看效果
        println("      Found variable: $(first_var["name"]) -> $(first_var["description"]) [$(first_var["unit"])]")

        # 2. 测试 get_parameters_detail
        println("\n   -> Testing get_parameters_detail for '$test_model_name'...")
        params_info = get_parameters_detail(test_model_name)

        @test params_info isa Vector
        # 注意：有些模型可能没有参数（虽然少见），如果确定有参数，可以加 !isempty
        if !isempty(params_info)
            first_param = params_info[1]
            @test first_param isa Dict
            @test haskey(first_param, "name")
            @test haskey(first_param, "bounds")
            @test first_param["type"] == "parameter"

            # 关键测试：Bounds 必须是 JSON 友好的 Vector (即 [min, max])，不能是 Tuple
            if !isnothing(first_param["bounds"])
                @test first_param["bounds"] isa Vector
                @test length(first_param["bounds"]) == 2
                @test first_param["bounds"][1] <= first_param["bounds"][2] # min <= max
            end

            println("      Found parameter: $(first_param["name"]) -> Bounds: $(first_param["bounds"])")
        end
    end
end