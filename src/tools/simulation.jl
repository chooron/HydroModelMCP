using ModelContextProtocol
using JSON3  # 必须引入，用于将 Dict 转为 String

# 定义 run_simulation 工具
simulation_tool = MCPTool(
    name="run_simulation",
    description="执行水文模型模拟。支持动态加载模型、多种数据源(CSV/JSON/Redis)、自动参数补全及结果持久化。",

    # Input Schema 保持不变
    input_schema=Dict{String,Any}(
        "type" => "object",
        "properties" => Dict{String,Any}(
            # 1. 模型名称
            "model_name" => Dict{String,Any}(
                "type" => "string",
                "description" => "水文模型的唯一标识符 (例如 'hbv', 'xaj', 'exphydro')。"
            ),

            # 2. 驱动数据配置 (嵌套对象)
            "forcing" => Dict{String,Any}(
                "type" => "object",
                "description" => "驱动数据配置。需指定 source_type 及对应的 path/key/data。",
                "properties" => Dict{String,Any}(
                    "source_type" => Dict{String,Any}(
                        "type" => "string",
                        "enum" => ["csv", "json", "redis"],
                        "description" => "数据源类型"
                    ),
                    "path" => Dict{String,Any}(
                        "type" => "string",
                        "description" => "文件路径 (仅适用于 csv/json)"
                    ),
                    "key" => Dict{String,Any}(
                        "type" => "string",
                        "description" => "Redis 键名 (仅适用于 redis)"
                    ),
                    "data" => Dict{String,Any}(
                        "type" => "object",
                        "description" => "直接传入的数据对象 (仅适用于 json)"
                    ),
                    "host" => Dict{String,Any}(
                        "type" => "string",
                        "default" => "127.0.0.1",
                        "description" => "Redis 服务器地址 (默认 127.0.0.1)"
                    ),
                    "port" => Dict{String,Any}(
                        "type" => "integer",
                        "default" => 6379,
                        "description" => "Redis 服务器端口 (默认 6379)"
                    )
                ),
                "required" => ["source_type"]
            ),

            # 3. 模型参数 (数组)
            "params" => Dict{String,Any}(
                "type" => "array",
                "items" => Dict{String,Any}("type" => "number"),
                "description" => "模型参数数组。如果省略，将自动生成随机参数。"
            ),

            # 4. 求解器 (枚举)
            "solver" => Dict{String,Any}(
                "type" => "string",
                "enum" => ["ODE", "DISCRETE", "MUTABLE", "IMMUTABLE"],
                "default" => "ODE",
                "description" => "微分方程求解器类型。"
            ),

            # 5. 插值器 (枚举)
            "interpolator" => Dict{String,Any}(
                "type" => "string",
                "enum" => ["LINEAR", "CONSTANT"],
                "default" => "LINEAR",
                "description" => "输入数据的插值方式。"
            ),

            # 6. 其他配置
            "config" => Dict{String,Any}(
                "type" => "object",
                "description" => "高级模拟配置 (例如指定输出变量)。",
                "properties" => Dict{String,Any}(
                    "output_variable" => Dict{String,Any}("type" => "string")
                )
            ),

            # 7. 初始状态
            "init_states" => Dict{String,Any}(
                "type" => "object",
                "description" => "自定义初始状态 (状态名 -> 值)。"
            )
        ),
        "required" => ["model_name", "forcing"]
    ),

    # Handler 函数 (修改点在这里)
    handler=function (params)
        try
            # 1. 调用业务逻辑，获取结果 (这是一个 Dict)
            result = Simulation.run_simulation(params)

            # 2. 【关键修改】显式转换为 JSON 字符串并包裹在 TextContent 中
            # MCP 不接受原生 Dict，必须给它 TextContent
            return TextContent(
                text = JSON3.write(result)
            )

        catch e
            # 捕获错误并返回标准错误格式
            err_msg = sprint(showerror, e)
            # 为了调试方便，可以缩短堆栈信息，或者只保留错误信息
            
            return CallToolResult(
                content=[
                    Dict("type" => "text", "text" => "Simulation Failed: $err_msg")
                ],
                is_error=true
            )
        end
    end
)

export simulation_tool