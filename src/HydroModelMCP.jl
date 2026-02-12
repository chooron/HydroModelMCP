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

# 校准相关依赖
using Optimization
using OptimizationBBO
using OptimizationMetaheuristics
using GlobalSensitivity

# Core 模块 (顺序重要：被依赖的在前)
include("core/dataloader.jl")
include("core/metrics.jl")
include("core/datasplitter.jl")
include("core/sampling.jl")
include("core/simulation.jl")
include("core/discovery.jl")
include("core/sensitivity.jl")
include("core/calibration.jl")
include("core/storage.jl")

# MCP 资源定义
include("resources/models.jl")
include("resources/calibration.jl")
include("resources/parameters.jl")
include("resources/templates.jl")

# MCP 工具封装
include("tools/simulation.jl")
include("tools/discovery.jl")
include("tools/calibration.jl")
include("prompts/experts.jl")

# 初始化存储后端 (可通过环境变量配置)
const STORAGE_BACKEND = if get(ENV, "STORAGE_BACKEND", "file") == "redis"
    Storage.RedisBackend(
        get(ENV, "REDIS_HOST", "127.0.0.1"),
        parse(Int, get(ENV, "REDIS_PORT", "6379")),
        "hydro";
        ttl = parse(Int, get(ENV, "HYDRO_STORAGE_TTL", "604800"))
    )
else
    Storage.FileBackend(
        get(ENV, "HYDRO_STORAGE_PATH", joinpath(homedir(), ".hydro_mcp", "storage"));
        ttl = parse(Int, get(ENV, "HYDRO_STORAGE_TTL", "604800"))
    )
end

# 构建资源列表
ALL_RESOURCES = [
    # 静态资源
    model_catalog_resource,
    algorithm_guide_resource,
    objective_guide_resource,
    # 动态资源 (存储相关)
    create_calibration_resources(STORAGE_BACKEND)...,
    # 动态资源 (URI模板)
    create_dynamic_resources(STORAGE_BACKEND)...
]

ALL_TOOLS = [
    # 模型发现
    list_models_tool,
    find_model_tool,
    get_model_info_tool,
    get_model_variables_tool,
    get_model_parameters_tool,
    # 模拟
    simulation_tool,
    # 校准工作流
    compute_metrics_tool,
    split_data_tool,
    sensitivity_tool,
    sampling_tool,
    calibrate_tool,
    calibrate_multi_tool,
    diagnose_tool,
    configure_objectives_tool,
    init_calibration_setup_tool,
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
        resources=ALL_RESOURCES,
        resource_templates=ALL_RESOURCE_TEMPLATES
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
        version = "0.1.0",
        tools = ALL_TOOLS,
        resources = ALL_RESOURCES,
        resource_templates = ALL_RESOURCE_TEMPLATES
    )

    # 3. 绑定并启动
    # 注意：这会阻塞当前进程，就像 Web Server 一样
    server.transport = transport
    ModelContextProtocol.connect(transport)
    println("🌊 水文模型服务已启动: http://127.0.0.1:3000")
    start!(server)
end


end # module HydroModelMCP
