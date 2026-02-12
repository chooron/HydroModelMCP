using ModelContextProtocol
using JSON3 # 需要引入 JSON3 用于将数据转为字符串

# ==============================================================================
# 1. 列出所有模型工具
# ==============================================================================
list_models_tool = MCPTool(
    name = "list_models",
    description = "列出当前库中所有可用的水文模型名称。在不知道该用哪个模型时，首先调用此工具。",
    parameters = [], 
    handler = function(params)
        try
            models = Discovery.list_models()
            
            # 修改点：显式序列化为 JSON 字符串并封装在 TextContent 中
            return TextContent(
                text = JSON3.write(models)
            )
        catch e
            return CallToolResult(
                content = [TextContent(text = "Error listing models: $e")],
                is_error = true
            )
        end
    end
)

# ==============================================================================
# 2. 查找模型工具
# ==============================================================================
find_model_tool = MCPTool(
    name = "find_model",
    description = "检查某个特定的水文模型是否存在。支持模糊匹配（忽略大小写）。如果存在，返回其在库中的标准名称。",
    parameters = [
        ToolParameter(
            name = "name",
            type = "string",
            description = "想要查找的模型名称 (例如 'HBV', 'xinanjiang')",
            required = true
        )
    ],
    handler = function(params)
        name_query = params["name"]
        try
            canonical_name = Discovery.find_model(name_query)
            
            result_dict = Dict{String, Any}()
            
            if isnothing(canonical_name)
                result_dict["found"] = false
                result_dict["message"] = "Model '$name_query' not found in library."
                result_dict["query"] = name_query
            else
                result_dict["found"] = true
                result_dict["canonical_name"] = canonical_name
                result_dict["query"] = name_query
                result_dict["message"] = "Model found."
            end
            
            # 修改点：返回 JSON 字符串
            return TextContent(
                text = JSON3.write(result_dict)
            )

        catch e
            return CallToolResult(
                content = [TextContent(text = "Error finding model: $e")],
                is_error = true
            )
        end
    end
)

# ==============================================================================
# 3. 获取模型基本信息
# ==============================================================================
get_model_info_tool = MCPTool(
    name = "get_model_info",
    description = "获取指定水文模型的详细信息，包括必需的输入变量、参数列表、状态变量以及模型结构的文本描述。",
    parameters = [
        ToolParameter(
            name = "model_name",
            type = "string",
            description = "模型名称 (例如 'hbv', 'xinanjiang')",
            required = true
        )
    ],
    handler = function(params)
        name = params["model_name"]
        try
            info = Discovery.get_model_info(name)
            # 修改点：无论 info 是 Dict 还是 Struct，JSON3.write 都能处理
            return TextContent(
                text = JSON3.write(info)
            )
        catch e
            return CallToolResult(
                content = [TextContent(text = "Error fetching model info: $e")],
                is_error = true
            )
        end
    end
)

# ==============================================================================
# 4. 变量查询工具
# ==============================================================================
get_model_variables_tool = MCPTool(
    name = "get_model_variables",
    description = "获取指定水文模型的所有内部变量详情，包括输入变量、状态变量和输出通量的物理含义及单位。",
    parameters = [
        ToolParameter(
            name = "model_name",
            type = "string",
            description = "模型名称 (例如 'hbv', 'xinanjiang')",
            required = true
        )
    ],
    handler = function(params)
        name = params["model_name"]
        try
            data = Discovery.get_variables_detail(name)
            return TextContent(
                text = JSON3.write(data)
            )
        catch e
            return CallToolResult(
                content = [TextContent(text = "Error fetching variables: $e")],
                is_error = true
            )
        end
    end
)

# ==============================================================================
# 5. 参数查询工具
# ==============================================================================
get_model_parameters_tool = MCPTool(
    name = "get_model_parameters",
    description = "获取指定水文模型的参数详情。返回参数列表、物理含义、单位以及**推荐的取值范围(Bounds)**。在进行参数率定前必须调用此工具。",
    parameters = [
        ToolParameter(
            name = "model_name",
            type = "string",
            description = "模型名称",
            required = true
        )
    ],
    handler = function(params)
        name = params["model_name"]
        try
            data = Discovery.get_parameters_detail(name)
            return TextContent(
                text = JSON3.write(data)
            )
        catch e
            return CallToolResult(
                content = [TextContent(text = "Error fetching parameters: $e")],
                is_error = true
            )
        end
    end
)

export list_models_tool, find_model_tool, get_model_info_tool, get_model_variables_tool, get_model_parameters_tool