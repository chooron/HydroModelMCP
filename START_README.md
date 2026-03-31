# 启动脚本说明

## 📁 文件说明

- **start.jl** - stdio 模式启动脚本（命令行工具）
- **start_http.jl** - HTTP 模式启动脚本（Web 服务）
- **.env** - HTTP 模式配置文件
- **.env.example** - 配置文件模板

## 🚀 快速启动

### stdio 模式
```bash
julia --project=. start.jl
```

### HTTP 模式
```bash
julia --project=. start_http.jl
```

## ⚙️ 配置 HTTP 模式

编辑 `.env` 文件：

```bash
# 主机地址（127.0.0.1=本地，0.0.0.0=允许外部）
JULIA_HTTP_HOST=127.0.0.1

# 端口号
JULIA_HTTP_PORT=3000

# CORS 来源（*=所有，或逗号分隔的域名列表）
JULIA_HTTP_ALLOWED_ORIGINS=*
```

## 📖 详细文档

查看 [USAGE.md](USAGE.md) 获取完整使用指南。

### MCP 工具请求协议说明

当前 HydroModelMCP 的模拟/验证/集合/率定工作流采用统一 v2 请求结构：

- 必填：`model`、`inputs`
- 可选：`output`、`options`
- `inputs` 下按角色组织：`forcing`、`observation`、`parameters`、`runtime`

完整示例请参考主文档 `README.md` 的 `Unified v2 Workflow Protocol` 章节。

## 🧪 测试连接

```bash
# Python 客户端测试
python test_python_client.py

# curl 测试
curl -X POST http://127.0.0.1:3000/ \
  -H 'Content-Type: application/json' \
  -H 'MCP-Protocol-Version: 2025-06-18' \
  -d '{"jsonrpc":"2.0","method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"test","version":"1.0.0"}},"id":1}' \
  -i
```
