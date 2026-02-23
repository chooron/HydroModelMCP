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
include("schemas/Schemas.jl")
using .Schemas

include("data_handles.jl")
include("core/dataloader.jl")
include("core/metrics.jl")
include("core/datasplitter.jl")
include("core/sampling.jl")
include("core/simulation.jl")
include("core/discovery.jl")
include("core/sensitivity.jl")
include("core/calibration.jl")
include("core/storage.jl")
include("core/ensemble.jl")
include("core/validation.jl")

# MCP 资源定义
include("resources/models.jl")
include("resources/calibration.jl")
include("resources/parameters.jl")
include("resources/templates.jl")

# MCP 工具辅助函数
include("tools/helpers.jl")

# MCP 工具封装
include("tools/data_loading.jl")
include("tools/simulation.jl")
include("tools/discovery.jl")
include("tools/calibration.jl")
include("tools/ensemble.jl")
include("tools/validation.jl")
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
    # 数据加载
    load_camels_data_tool,
    analyze_distribution_from_handle_tool,
    # 模型发现
    list_models_tool,
    find_model_tool,
    get_model_info_tool,
    get_model_variables_tool,
    get_model_parameters_tool,
    # 模拟
    simulation_tool,
    ensemble_parameter_tool,
    validation_tool,
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
    # 创建并运行 MCP 服务 (使用 stdio 传输)
    server = mcp_server(
        name="HydroModel-Agent-Interface",
        version="0.1.0",
        tools=ALL_TOOLS,
        resources=ALL_RESOURCES
        # prompts=ALL_PROMPTS
    )
    # 设置 stdio 传输层
    server.transport = ModelContextProtocol.StdioTransport()
    start!(server)
    println("🌊 HydroModelMCP 服务已启动 (stdio 传输)")
end

using ModelContextProtocol: HttpTransport

function run_http_server(;
    host::String = get(ENV, "MCP_HOST", "127.0.0.1"),
    port::Int = parse(Int, get(ENV, "MCP_PORT", "3000")),
    allowed_origins::Vector{String} = String[]
)
    # 1. 定义传输层 (监听指定端口)
    transport = HttpTransport(
        host = host,
        port = port,
        endpoint = "/",
        protocol_version = "2025-06-18",  # MCP 协议版本
        session_required = false,  # 开发环境可设为 false，生产环境建议 true
        allowed_origins = isempty(allowed_origins) ? ["*"] : allowed_origins  # CORS 配置
    )

    # 2. 创建服务 (加载所有 Tools 和 Resources)
    server = mcp_server(
        name = "HydroModel-Agent-Interface",
        version = "0.1.0",
        tools = ALL_TOOLS,
        resources = ALL_RESOURCES
    )

    # 3. 绑定并启动
    server.transport = transport
    ModelContextProtocol.connect(transport)

    println("🌊 HydroModelMCP HTTP 服务已启动")
    println("   地址: http://$host:$port")
    println("   协议版本: 2025-06-18")
    println("   工具数量: $(length(ALL_TOOLS))")
    println("   资源数量: $(length(ALL_RESOURCES))")
    println("\n📝 Python 客户端连接示例:")
    println("   from mcp import ClientSession, StdioServerParameters")
    println("   from mcp.client.stdio import stdio_client")
    println("   # 或使用 HTTP client 连接到 http://$host:$port")

    start!(server)
end


end # module HydroModelMCP
