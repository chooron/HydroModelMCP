using Pkg
Pkg.activate(@__DIR__) # 激活当前环境
using ModelContextProtocol

# Create a server with a simple echo tool
server = mcp_server(
    name = "echo-server",
    version = "1.0.0",
    tools = [
        MCPTool(
            name = "echo",
            description = "Echo back the input message",
            parameters = [
                ToolParameter(
                    name = "message",
                    type = "string",
                    description = "Message to echo",
                    required = true
                )
            ],
            handler = (params) -> TextContent(text = params["message"])
        )
    ]
)

# Start the server (stdio transport by default)
start!(server)