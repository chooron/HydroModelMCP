using Pkg
Pkg.activate(@__DIR__) # 激活当前环境
using HydroModelMCP

# stdio 传输模式启动
# 适用于命令行工具和 MCP Inspector
# 使用方式: npx @modelcontextprotocol/inspector julia --project=. start.jl
println("🚀 正在启动 HydroModelMCP 服务 (stdio 模式)...")
HydroModelMCP.run_server()

