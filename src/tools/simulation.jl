using ModelContextProtocol
using JSON3
using .Schemas

# 定义 run_simulation 工具
simulation_tool = MCPTool(
    name="run_simulation",
    description="执行水文模型模拟。支持动态加载模型、多种数据源(CSV/JSON/Redis)、自动参数补全及结果持久化。",

    input_schema=Dict{String,Any}(
        "type" => "object",
        "properties" => Dict{String,Any}(
            "model_name" => MODEL_NAME_SCHEMA,
            "forcing" => FORCING_SCHEMA,
            "params" => PARAMS_ARRAY_SCHEMA,
            "solver" => SOLVER_SCHEMA,
            "interpolator" => INTERPOLATOR_SIMPLE_SCHEMA,
            "config" => CONFIG_SCHEMA,
            "init_states" => INIT_STATES_SCHEMA
        ),
        "required" => ["model_name", "forcing"]
    ),

    handler=function (params)
        # 验证必需参数
        validation_error = validate_required_params(params, ["model_name", "forcing"])
        if !isnothing(validation_error)
            return create_error_response(validation_error)
        end

        # 验证枚举参数
        if haskey(params, "solver")
            enum_error = validate_enum_param(params, "solver",
                ["ODE", "DISCRETE", "MUTABLE", "IMMUTABLE"], "DISCRETE")
            if !isnothing(enum_error)
                return create_error_response(enum_error)
            end
        end

        if haskey(params, "interpolator")
            enum_error = validate_enum_param(params, "interpolator",
                ["LINEAR", "CONSTANT"], "LINEAR")
            if !isnothing(enum_error)
                return create_error_response(enum_error)
            end
        end

        try
            result = Simulation.run_simulation(params)
            return TextContent(text = JSON3.write(result))
        catch e
            err_msg = sprint(showerror, e)
            return create_error_response("Simulation Failed: $err_msg")
        end
    end
)

export simulation_tool