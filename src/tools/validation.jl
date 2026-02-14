"""
验证期运行MCP工具定义
"""

using ModelContextProtocol
using JSON3
using .Schemas

"""
验证期运行工具
"""
validation_tool = MCPTool(
    name = "run_validation",
    description = """
    在率定期和验证期分别运行模型，计算性能指标。

    用于评估模型的泛化能力和参数的稳定性。

    时间范围可以通过日期（start/end）或索引（start_index/end_index）指定。
    """,
    input_schema = Dict{String,Any}(
        "type" => "object",
        "properties" => Dict{String,Any}(
            "model_name" => MODEL_NAME_SCHEMA,
            "parameters" => PARAMETERS_SCHEMA,
            "forcing" => FORCING_SCHEMA,
            "observation" => OBSERVATION_SCHEMA,
            "obs_column" => OBS_COLUMN_SCHEMA,
            "calibration_period" => CALIBRATION_PERIOD_SCHEMA,
            "validation_period" => VALIDATION_PERIOD_SCHEMA,
            "metrics" => METRICS_SCHEMA,
            "input_mapping" => INPUT_MAPPING_SCHEMA,
            "solver" => SOLVER_SCHEMA,
            "interpolator" => INTERPOLATOR_SCHEMA
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
            result = Validation.run_validation(args)
            return TextContent(text = JSON3.write(result))
        catch e
            return create_error_response(e)
        end
    end
)

export validation_tool
