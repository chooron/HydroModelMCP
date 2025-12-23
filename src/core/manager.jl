module Manager

"""工具管理：注册并列出可用工具（用于 MCP 管理）。"""

export register_tool, list_tools, get_tool

const TOOLS = Dict{String,Any}()

function register_tool(name::String, tool)
    TOOLS[name] = tool
    return true
end

function list_tools()
    return collect(keys(TOOLS))
end

function get_tool(name::String)
    return get(TOOLS, name, nothing)
end

end # module
