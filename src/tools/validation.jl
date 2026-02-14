"""
验证期运行MCP工具定义
"""

using ..Validation
using ..JSON3

"""
验证期运行工具
"""
function validation_tool()
    return MCPTool(
        name = "run_validation",
        description = """
        在率定期和验证期分别运行模型，计算性能指标。

        用于评估模型的泛化能力和参数的稳定性。

        时间范围可以通过日期（start/end）或索引（start_index/end_index）指定。
        """,
        input_schema = Dict(
            "type" => "object",
            "properties" => Dict(
                "model_name" => Dict(
                    "type" => "string",
                    "description" => "模型名称"
                ),
                "parameters" => Dict(
                    "type" => "object",
                    "description" => "模型参数字典"
                ),
                "forcing" => Dict(
                    "type" => "object",
                    "description" => "驱动数据配置"
                ),
                "observation" => Dict(
                    "type" => "object",
                    "description" => "观测数据配置"
                ),
                "obs_column" => Dict(
                    "type" => "string",
                    "description" => "观测值列名"
                ),
                "calibration_period" => Dict(
                    "type" => "object",
                    "description" => "率定期时间范围（start/end或start_index/end_index）"
                ),
                "validation_period" => Dict(
                    "type" => "object",
                    "description" => "验证期时间范围（start/end或start_index/end_index）"
                ),
                "metrics" => Dict(
                    "type" => "array",
                    "description" => "评价指标列表",
                    "items" => Dict(
                        "type" => "string",
                        "enum" => ["NSE", "KGE", "LogNSE", "LogKGE", "RMSE", "R2", "PBIAS"]
                    ),
                    "default" => ["NSE", "KGE", "RMSE"]
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
                )
            ),
            "required" => [
                "model_name", "parameters", "forcing", "observation",
                "obs_column", "calibration_period", "validation_period"
            ]
        ),
        handler = function(args::AbstractDict)
            # 参数验证
            validation_error = validate_required_params(
                args,
                ["model_name", "parameters", "forcing", "observation",
                 "obs_column", "calibration_period", "validation_period"]
            )
            if !isnothing(validation_error)
                return create_error_response(validation_error)
            end

            try
                result = Validation.run_validation(args)
                return TextContent(text = JSON3.write(result))
            catch e
                return create_error_response(e)
            end
        end
    )
end
