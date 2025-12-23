using Pkg
Pkg.activate(@__DIR__) # 激活当前环境
using HydroModelMCP

# npx @modelcontextprotocol/inspector julia --project=. start.jl
# 启动服务
HydroModelMCP.run_server()

