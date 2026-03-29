# `runoff-forecast` MCP 测试对话脚本

这个脚本用于验证 Codex 是否已经能够通过 `HydroModelMCP` 正常调用 `runoff-forecast` 相关能力。

测试目标：

- MCP 能被 Codex 正常选中
- `list_models`、`find_model`、`get_model_info` 可用
- `list_workspace_files` 可用
- `run_simulation` 可用
- `compute_metrics` 可用
- `clear_session_cache` 会在对话结束时被调用

建议在仓库根目录 `E:\JlCode\HydroModelMCP` 中执行这些对话。

## 测试前准备

确认工作区里至少有这些文件：

- `./data/03604000.csv`

如果你希望测试指标计算，还需要一个可识别的观测流量文件。当前仓库中 `03604000.csv` 自带常见观测列，可直接用于 `compute_metrics` 测试。

## 测试对话 1：模型发现

用户输入：

```text
请使用 HydroModelMCP，先帮我找一个适合当前项目的降雨径流水文模型，并说明有哪些候选模型。
```

期望 Codex 行为：

1. 调用 `find_model`
2. 必要时调用 `list_models`
3. 调用 `get_model_info`
4. 调用 `get_model_parameters`

期望结果：

- 能返回一个候选模型，通常会包含 `exphydro`
- 能说明模型输入变量与参数数量

## 测试对话 2：检查工作区数据

用户输入：

```text
请不要读取原始 csv 内容，直接检查当前工作区 ./data 和 ./result 下面有哪些 csv 文件。
```

期望 Codex 行为：

1. 调用 `list_workspace_files`
2. 分别检查 `./data` 和 `./result`

期望结果：

- 正常列出文件
- 不把整段 CSV 原文贴进对话

## 测试对话 3：执行一次单次模拟

用户输入：

```text
请按 runoff-forecast 的工作流，用 exphydro 对 ./data/03604000.csv 跑一次模拟。不要手工读取 csv，直接把路径传给 MCP。参数如果没有给出，就让 MCP 自动随机生成，并把结果写到 ./result。
```

期望 Codex 行为：

1. 调用 `find_model` 或直接使用 `exphydro`
2. 调用 `run_simulation`
3. 传入类似参数：

```json
{
  "model": "exphydro",
  "source_type": "csv",
  "path": "./data/03604000.csv",
  "output_dir": "./result"
}
```

期望结果：

- 返回 `run_id`
- 返回 `output_path`
- 返回 `metadata_path`
- 返回 `summary_path`
- 返回 `params_used`
- `params_source` 为 `random`

## 测试对话 4：计算评价指标

用户输入：

```text
如果 ./data/03604000.csv 里存在观测流量列，请继续基于刚才的模拟结果计算评价指标，并把指标结果写到 ./result/metrics。
```

期望 Codex 行为：

1. 使用上一步的模拟输出 `output_path`
2. 调用 `compute_metrics`
3. 传入类似参数：

```json
{
  "simulated": {
    "source_type": "csv",
    "path": "<上一步 output_path>"
  },
  "observed": {
    "source_type": "csv",
    "path": "./data/03604000.csv"
  },
  "output_dir": "./result"
}
```

期望结果：

- 返回 `metrics`
- 返回 `output_path`
- 结果中通常包含 `NSE`、`KGE`、`RMSE`、`PBIAS`、`R2`

## 测试对话 5：完整验收版

用户输入：

```text
请严格按照 runoff-forecast skill 执行一次完整流程：
1. 识别模型
2. 检查 ./data 下的输入文件
3. 用 exphydro 对 ./data/03604000.csv 运行一次模拟
4. 如果可以，计算评价指标
5. 输出结果摘要
6. 在结束前调用 clear_session_cache 清理会话缓存
```

期望 Codex 行为：

- 依次调用：
  - `find_model`
  - `get_model_info`
  - `get_model_parameters`
  - `list_workspace_files`
  - `run_simulation`
  - `compute_metrics`
  - `clear_session_cache`

期望结果结构：

- `Runoff run summary`
- `Model`
- `Workspace inputs`
- `Execution`
- `Results`
- `Evaluation`
- `Outputs written`
- `Next action`

## 失败判定

如果出现以下情况，说明 MCP 接入仍有问题：

- Codex 没有调用任何 MCP 工具，直接改用 shell 读 CSV
- Codex 调用 MCP 时提示 server startup failed
- `run_simulation` 调用后没有生成 `./result/simulation` 产物
- 对话结束时没有调用 `clear_session_cache`

## 一条最小测试输入

如果你只想快速测一次，直接给 Codex 这一句：

```text
请使用 HydroModelMCP，按 runoff-forecast skill 对 ./data/03604000.csv 做一次 exphydro 模拟，并在结束前清理 session cache。
```

如果这句能稳定触发 `run_simulation` 和 `clear_session_cache`，说明 MCP 已基本可用。
