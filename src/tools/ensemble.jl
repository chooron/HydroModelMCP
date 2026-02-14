"""
集合模拟MCP工具定义
"""

using ModelContextProtocol
using JSON3
using .Schemas

# ==============================================================================
# 参数集合模拟工具
# ==============================================================================
ensemble_parameter_tool = MCPTool(
    name = "run_ensemble_parameters",
    description = """
    对单个模型批量运行多个参数集的模拟，计算集合统计量。

    适用场景：
    - 参数不确定性分析
    - 参数敏感性可视化
    - 集合预报
    - 蒙特卡洛模拟

    支持并行执行以提升性能。返回每个成员的模拟结果及集合均值、标准差。
    """,
    input_schema = Dict{String,Any}(
        "type" => "object",
        "properties" => Dict{String,Any}(
            "model_name" => MODEL_NAME_SCHEMA,
            "parameter_sets" => PARAMETER_SETS_SCHEMA,
            "forcing" => FORCING_SCHEMA,
            "input_mapping" => INPUT_MAPPING_SCHEMA,
            "solver" => SOLVER_SCHEMA,
            "interpolator" => INTERPOLATOR_SCHEMA,
            "parallel" => PARALLEL_SCHEMA,
            "save_to_storage" => SAVE_TO_STORAGE_SCHEMA
        ),
        "required" => ["model_name", "parameter_sets", "forcing"]
    ),
    handler = function(args::AbstractDict)
        # 参数验证
        validation_error = validate_required_params(
            args,
            ["model_name", "parameter_sets", "forcing"]
        )
        if !isnothing(validation_error)
            return create_error_response(validation_error)
        end

        # 验证parameter_sets是数组
        if !isa(args["parameter_sets"], Vector)
            return create_error_response(
                "parameter_sets must be an array"
            )
        end

        if isempty(args["parameter_sets"])
            return create_error_response(
                "parameter_sets cannot be empty"
            )
        end

        # 验证枚举参数
        if haskey(args, "solver")
            enum_error = validate_enum_param(args, "solver",
                ["ODE", "DISCRETE", "MUTABLE", "IMMUTABLE"], "DISCRETE")
            if !isnothing(enum_error)
                return create_error_response(enum_error)
            end
        end

        if haskey(args, "interpolator")
            enum_error = validate_enum_param(args, "interpolator",
                ["LINEAR", "CONSTANT", "DIRECT"], "LINEAR")
            if !isnothing(enum_error)
                return create_error_response(enum_error)
            end
        end

        try
            # 执行集合模拟
            result = Ensemble.run_ensemble(args)

            # 保存到存储层（可选）
            if get(args, "save_to_storage", true)
                Storage.save_result(
                    STORAGE_BACKEND,
                    "ensemble",
                    result["result_id"],
                    result
                )
            end

            return TextContent(text = JSON3.write(result))
        catch e
            return create_error_response(e)
        end
    end
)

export ensemble_parameter_tool
