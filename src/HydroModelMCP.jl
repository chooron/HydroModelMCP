module HydroModelMCP

using CSV
using DataFrames
using JSON3
using Redis

using ComponentArrays
using DataInterpolations
using Statistics

using ModelContextProtocol

using HydroModels
using HydroModelLibrary

include("core/dataloader.jl")
include("core/simulation.jl")
include("core/discovery.jl")

include("tools/simulation.jl")
include("tools/discovery.jl")
include("prompts/experts.jl")

ALL_TOOLS = [
    simulation_tool,
    get_model_info_tool,
    list_models_tool,
    find_model_tool,
    get_model_variables_tool,
    get_model_parameters_tool
]

ALL_PROMPTS = [
    Experts.hydro_expert_prompt
]

function run_server()
    # 创建并运行 MCP 服务
    server = mcp_server(
        name="HydroModel-Agent-Interface",
        version="0.1.0",
        tools=ALL_TOOLS,
        # prompts=ALL_PROMPTS
    )
    start!(server)
end

using ModelContextProtocol: HttpTransport

function run_http_server()
    # 1. 定义传输层 (监听 3000 端口)
    transport = HttpTransport(
        host = "0.0.0.0", # 允许局域网访问，不仅是 localhost
        port = 3000,
        enable_sse = true # 开启流式推送
    )

    # 2. 创建服务 (加载你所有的 Tools)
    server = mcp_server(
        name = "Hydro-Web-Service",
        tools = ALL_TOOLS 
    )

    # 3. 绑定并启动
    # 注意：这会阻塞当前进程，就像 Web Server 一样
    server.transport = transport
    ModelContextProtocol.connect(transport)
    println("🌊 水文模型服务已启动: http://127.0.0.1:3000")
    start!(server)
end


end # module HydroModelMCP
