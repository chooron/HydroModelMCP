"""
集合模拟MCP工具定义
"""

using ..Ensemble
using ..Storage
using ..JSON3

"""
集合模拟工具
"""
function ensemble_tool()
    return MCPTool(
        name = "run_ensemble",
        description = """
        批量运行多个参数集的模拟，计算集合统计量。

        适用场景：
        - 不确定性分析
        - 参数敏感性可视化
        - 集合预报

        支持并行执行以提升性能。
        """,
        input_schema = Dict(
            "type" => "object",
            "properties" => Dict(
                "model_name" => Dict(
                    "type" => "string",
                    "description" => "模型名称（如exphydro, HBV）"
                ),
                "parameter_sets" => Dict(
                    "type" => "array",
                    "description" => "参数集列表，每个元素是一个参数字典",
                    "items" => Dict("type" => "object")
                ),
                "forcing" => Dict(
                    "type" => "object",
                    "description" => "驱动数据配置（支持csv/json/redis）"
                ),
                "input_mapping" => Dict(
                    "type" => "object",
                    "description" => "输入变量映射（可选）"
                ),
                "solver" => Dict(
                    "type" => "string",
                    "enum" => ["ODE", "DISCRETE", "MUTABLE", "IMMUTABLE"],
                    "default" => "DISCRETE"
                ),
                "interpolator" => Dict(
                    "type" => "string",
                    "enum" => ["LINEAR", "CONSTANT", "DIRECT"],
                    "default" => "LINEAR"
                ),
                "parallel" => Dict(
                    "type" => "boolean",
                    "description" => "是否并行执行（默认true）",
                    "default" => true
                ),
                "save_to_storage" => Dict(
                    "type" => "boolean",
                    "description" => "是否保存到存储层（默认true）",
                    "default" => true
                )
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
end
