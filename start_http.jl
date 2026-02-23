using Pkg
Pkg.activate(@__DIR__)

# 加载环境变量
using DotEnv
DotEnv.config(joinpath(@__DIR__, ".env"))

using HydroModelMCP

# HTTP 传输模式启动
# 适用于 Python MCP client 和 Web 应用
# 从 .env 文件读取配置

# 读取配置
host = get(ENV, "JULIA_HTTP_HOST", "127.0.0.1")
port = parse(Int, get(ENV, "JULIA_HTTP_PORT", "3000"))
allowed_origins = get(ENV, "JULIA_HTTP_ALLOWED_ORIGINS", "*")

# 解析 allowed_origins（支持逗号分隔）
origins = allowed_origins == "*" ? String[] : split(allowed_origins, ",") .|> strip

println("🚀 正在启动 HydroModelMCP HTTP 服务...")
println("📋 配置信息:")
println("   主机: $host")
println("   端口: $port")
println("   允许来源: $(isempty(origins) ? "所有来源 (*)" : join(origins, ", "))")
println()

# 启动服务
HydroModelMCP.run_http_server(
    host = host,
    port = port,
    allowed_origins = origins
)
