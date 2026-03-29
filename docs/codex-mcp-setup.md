# HydroModelMCP 操作手册

本文档给出一个最小可用流程，包括：

- 本地安装 HydroModelMCP
- 启动并验证 MCP 服务
- 将这个 MCP 服务接入 Codex
- 常用环境变量
- 会话结束后的缓存清理

## 1. 前置条件

需要先准备：

- Julia 1.10 及以上
- Git
- 可选：Redis

推荐在 Windows PowerShell 中操作。

## 2. 获取项目并安装依赖

在本地执行：

```powershell
git clone https://github.com/chooron/HydroModelMCP
cd HydroModelMCP
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

如果你已经在项目目录里，只需要执行最后一条依赖安装命令即可。

## 3. 本地启动方式

### 3.1 推荐：stdio 模式

这是最适合接入 Codex 的方式。

```powershell
julia --project=. start.jl
```

说明：

- `start.jl` 会以 MCP `stdio` 服务启动
- 这个模式适合 Codex、Inspector 和其他本地 MCP 客户端

### 3.2 可选：HTTP 模式

如果你要给其他 HTTP 客户端接入，可以用：

```powershell
julia --project=. start_http.jl
```

默认地址：

```text
http://127.0.0.1:3000/
```

## 4. 启动前的常用环境变量

### 4.1 结果存储

```powershell
$env:STORAGE_BACKEND = "file"
$env:HYDRO_STORAGE_PATH = "C:\Users\OMEN\.hydro_mcp\storage"
$env:HYDRO_STORAGE_TTL = "604800"
```

说明：

- `STORAGE_BACKEND=file` 表示最终结果走文件存储
- 最终结果仍会写入工作区下的 `./result`

### 4.2 会话缓存

推荐优先使用 Redis 作为会话缓存，用于 `data_handle` 之类的中间态。

```powershell
$env:HYDRO_SESSION_CACHE_BACKEND = "redis"
$env:HYDRO_SESSION_CACHE_TTL = "86400"
$env:REDIS_HOST = "127.0.0.1"
$env:REDIS_PORT = "6379"
```

如果你本机没有 Redis，可以改为：

```powershell
$env:HYDRO_SESSION_CACHE_BACKEND = "memory"
```

说明：

- 中间态优先放 Redis，会话结束后可清理
- Redis 不可用时，服务会回退到内存缓存
- `./result` 下的最终产物不会因为清缓存而被删除

## 5. 先做一次本地验证

建议先用 Inspector 验证 MCP 服务能正常列出工具。

```powershell
npx @modelcontextprotocol/inspector --cli julia --project=. start.jl --method tools/list
```

如果 `julia` 在你的系统里不是直接可执行，改成 Julia 的绝对路径，例如：

```powershell
npx @modelcontextprotocol/inspector --cli "C:\Users\OMEN\.julia\juliaup\julia-1.12.1+0.x64.w64.mingw32\bin\julia.exe" --project=. start.jl --method tools/list
```

看到工具列表里包含这些名称，说明服务基本可用：

- `list_models`
- `find_model`
- `run_simulation`
- `compute_metrics`
- `list_workspace_files`
- `clear_session_cache`

## 6. 安装到 Codex

### 6.1 推荐方式：在 Codex 中新增一个 stdio MCP 服务

在 Codex 的 MCP 服务配置界面里，新建一个服务，填写下面这些信息。

服务名称示例：

```text
HydroModelMCP
```

命令：

```text
C:\Users\OMEN\.julia\juliaup\julia-1.12.1+0.x64.w64.mingw32\bin\julia.exe
```

参数：

```text
--project=E:\JlCode\HydroModelMCP
E:\JlCode\HydroModelMCP\start.jl
```

工作目录如果界面支持，填：

```text
E:\JlCode\HydroModelMCP
```

建议环境变量：

```text
STORAGE_BACKEND=file
HYDRO_SESSION_CACHE_BACKEND=redis
HYDRO_SESSION_CACHE_TTL=86400
REDIS_HOST=127.0.0.1
REDIS_PORT=6379
```

如果没有 Redis：

```text
STORAGE_BACKEND=file
HYDRO_SESSION_CACHE_BACKEND=memory
```

### 6.1.1 直接写入 `~/.codex/config.toml`

根据 OpenAI 当前 Codex 文档，Codex 支持在 `~/.codex/config.toml` 中配置 `mcp_servers`。你的本机可以直接加入下面这段：

```toml
[mcp_servers.HydroModelMCP]
command = "C:\\Users\\OMEN\\.julia\\juliaup\\julia-1.12.1+0.x64.w64.mingw32\\bin\\julia.exe"
args = ["--project=E:\\JlCode\\HydroModelMCP", "E:\\JlCode\\HydroModelMCP\\start.jl"]

[mcp_servers.HydroModelMCP.env]
STORAGE_BACKEND = "file"
HYDRO_SESSION_CACHE_BACKEND = "redis"
HYDRO_SESSION_CACHE_TTL = "86400"
REDIS_HOST = "127.0.0.1"
REDIS_PORT = "6379"
```

如果你暂时不用 Redis，把 `env` 段改成：

```toml
[mcp_servers.HydroModelMCP]
command = "C:\\Users\\OMEN\\.julia\\juliaup\\julia-1.12.1+0.x64.w64.mingw32\\bin\\julia.exe"
args = ["--project=E:\\JlCode\\HydroModelMCP", "E:\\JlCode\\HydroModelMCP\\start.jl"]

[mcp_servers.HydroModelMCP.env]
STORAGE_BACKEND = "file"
HYDRO_SESSION_CACHE_BACKEND = "memory"
```

写完后，重启 Codex。

### 6.2 参数填写原则

- `command` 填 Julia 可执行文件
- `args` 里指定 `--project=<项目根目录>` 和 `start.jl`
- 优先使用绝对路径，不要依赖当前 shell 的相对路径
- 如果 Codex 有单独的环境变量配置项，就把 Redis 和存储相关变量放进去

## 7. 在 Codex 里做最小验收

接入完成后，可以在 Codex 中直接让它调用这个 MCP。

建议按这个顺序验证：

1. 先让 Codex 调用 `list_models`
2. 再调用 `list_workspace_files` 查看 `./data`
3. 然后对 `./data/03604000.csv` 执行一次 `run_simulation`
4. 如果有观测数据，再调用 `compute_metrics`
5. 最后调用 `clear_session_cache`

一个典型验证请求可以是：

```text
列出可用模型，然后检查 ./data 下有哪些 csv 文件，再用 exphydro 对 ./data/03604000.csv 跑一次模拟。
```

如果你使用 Codex CLI，也可以先验证 MCP 是否已被 Codex 识别：

```powershell
codex mcp list
```

如果配置正确，你应该能看到名为 `HydroModelMCP` 的服务器。

如果执行成功，你应当能在项目目录下看到：

```text
./result/simulation/
./result/metrics/
./result/run_summary_*.md
```

## 8. 常用工具说明

### 8.1 `run_simulation`

用于单次模型运行，当前对 `runoff-forecast` skill 已对齐。

最小输入示例：

```json
{
  "model": "exphydro",
  "source_type": "csv",
  "path": "./data/03604000.csv"
}
```

### 8.2 `list_workspace_files`

用于让客户端只看文件清单，不直接把 CSV 内容读入对话。

示例：

```json
{
  "directory": "./data",
  "extensions": ["csv"]
}
```

### 8.3 `clear_session_cache`

用于清理会话中的中间缓存。

示例：

```json
{}
```

这个工具只清理：

- Redis 中的会话缓存
- 或当前进程内的内存缓存

不会清理：

- `./result` 下的模拟结果
- `./result/metrics` 下的指标文件
- `./result` 下的 summary 文件

## 9. 推荐运行习惯

- 最终结果保留在 `./result`
- 中间态使用 Redis 或内存缓存
- 每次会话结束前调用一次 `clear_session_cache`
- 如果是 `runoff-forecast` 这类 skill，建议把清缓存作为结束动作固定下来

## 10. 常见问题

### 10.1 Codex 无法启动服务

优先检查：

- Julia 路径是不是绝对路径
- `--project` 是否指向项目根目录
- `start.jl` 是否使用绝对路径
- 项目依赖是否已经 `Pkg.instantiate()`
- `stdio` 模式启动脚本是否向标准输出打印了普通文本；MCP `stdio` 服务不能在握手前向 stdout 输出日志

### 10.2 Codex 能连上，但看不到工具

优先用 Inspector 单独验证：

```powershell
npx @modelcontextprotocol/inspector --cli "C:\Users\OMEN\.julia\juliaup\julia-1.12.1+0.x64.w64.mingw32\bin\julia.exe" --project=. start.jl --method tools/list
```

如果 Inspector 也失败，先不要查 Codex，先修服务本身。

### 10.2.1 Codex 连上了，但模型不主动调用这个 MCP

这通常不是安装失败，而是模型没有足够强的选择信号。

建议在项目根目录的 `AGENTS.md` 里加入一条明确规则，例如：

```text
For hydrological modeling, runoff simulation, calibration, metrics, and workspace data inspection in this repository, always prefer the HydroModelMCP MCP server before trying ad hoc shell-based workflows.
```

这样 Codex 更容易稳定选中这个 MCP。

### 10.3 Redis 没启动怎么办

两种方式：

- 启动本机 Redis，然后继续使用 `HYDRO_SESSION_CACHE_BACKEND=redis`
- 或直接切换成 `HYDRO_SESSION_CACHE_BACKEND=memory`

### 10.4 清缓存后结果文件不见了吗

不会。

`clear_session_cache` 只删除中间态缓存，不删除 `./result` 下的最终产物。

## 11. 一组推荐配置

如果你当前项目路径就是 `E:\JlCode\HydroModelMCP`，推荐直接使用：

命令：

```text
C:\Users\OMEN\.julia\juliaup\julia-1.12.1+0.x64.w64.mingw32\bin\julia.exe
```

参数：

```text
--project=E:\JlCode\HydroModelMCP
E:\JlCode\HydroModelMCP\start.jl
```

环境变量：

```text
STORAGE_BACKEND=file
HYDRO_SESSION_CACHE_BACKEND=redis
HYDRO_SESSION_CACHE_TTL=86400
REDIS_HOST=127.0.0.1
REDIS_PORT=6379
```

如果你希望我顺手再补一份“Codex 配置截图式模板”或“Claude Desktop / Cherry Studio / Inspector 三端接入对照版”，可以继续加在 `docs/` 里。
