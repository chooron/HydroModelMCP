# `baseline-workflow` MCP 测试对话脚本

> 如果你需要更全面、可回归、可门禁的自动化测试方案，请优先使用：
>
> - `docs/mcp-llm-harness.md`
> - skill: `hydromodel-mcp-harness`

这个脚本用于验证 Codex 是否已经能够通过 `HydroModelMCP` 正常执行基线测试流程。

测试目标：

- MCP 能被 Codex 正常选中
- `list_models`、`find_model`、`get_model_info` 可用
- `list_workspace_files` 可用
- `inspect_hydro_data` 可用
- `run_simulation` 可用
- `load_hydro_csv` 可用
- `calibrate_model` 可用
- `compute_metrics` 可用
- 对 calibration / validation 路径，若无实测径流会立即停止并反馈
- `clear_session_cache` 会在正常结束时被调用

建议在仓库根目录 `E:\JlCode\HydroModelMCP` 中执行这些对话。

## 测试前准备

确认工作区里至少有这些文件：

- `./data/03604000.csv`

当前仓库中的 `03604000.csv` 可用于验证：

- forcing 要素识别
- observed runoff 检测
- simulation 路径
- metrics 路径

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
5. 若 `find_model` 返回空候选，必须回退到 `list_models` 并优先选择通用降雨径流模型（如 `exphydro`）继续

期望结果：

- 能返回一个候选模型，通常会包含 `exphydro`
- 能说明模型输入变量与参数数量
- 不因自然语言查询（含中文描述）返回空候选后直接中断

## 测试对话 2：检查工作区文件，不读取原始内容

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
- 如果 `./result` 尚不存在，`list_workspace_files` 会自动创建目录并返回空列表，而不是报错

## 测试对话 3：检查输入数据要素是否充足

用户输入：

```text
请使用 HydroModelMCP 检查 ./data/03604000.csv 是否具备 exphydro 运行所需的输入要素，并判断它是否包含实测径流。不要读取原始 csv 内容。
```

期望 Codex 行为：

1. 可先调用 `find_model`
2. 调用 `inspect_hydro_data`
3. 使用类似参数：

```json
{
  "source": {
    "source_type": "csv",
    "path": "./data/03604000.csv"
  },
  "model": "exphydro",
  "intended_use": "simulation"
}
```

期望结果：

- 返回 `forcing_elements`
- 返回 `observed_runoff`
- 返回 `model_check`
- 返回 `readiness`
- 不贴出原始 CSV 内容

## 测试对话 4：执行一次单次模拟

用户输入：

```text
请按 baseline-workflow 的工作流，用 exphydro 对 ./data/03604000.csv 跑一次模拟。不要手工读取 csv，先检查输入数据，再把路径传给 MCP。参数如果没有给出，就让 MCP 自动随机生成，并把结果写到 ./result。
```

期望 Codex 行为：

1. 调用 `inspect_hydro_data`
2. 调用 `run_simulation`
3. 推荐传入类似参数：

```json
{
  "model": "exphydro",
  "inputs": {
    "forcing": {
      "source_type": "csv",
      "path": "./data/03604000.csv"
    }
  },
  "output": {
    "result_source_type": "csv",
    "output_dir": "./result"
  }
}
```

期望结果：

- 返回 `run_id`
- 返回 `output_path`
- 返回 `metadata_path`
- 返回 `summary_path`
- 返回 `params_used`
- `params_source` 为 `random`
- 若用户给出部分参数（而非全量参数），simulation 路径应补全缺失参数并在 `warnings` 中说明补全来源

## 测试对话 5：校准前检查，如果无实测径流应立即停止

用户输入：

```text
请按 baseline-workflow 的风格，先检查 ./data/03604000.csv 是否适合做模型参数率定。如果缺少实测径流，不要继续调用校准工具，直接停止并告诉我缺什么。
```

期望 Codex 行为：

1. 调用 `inspect_hydro_data`
2. 使用 `intended_use="calibration"`
3. 根据检查结果决定是否继续

期望结果：

- 如果检测到实测径流，则说明可以继续率定
- 如果没有检测到实测径流，则明确停止，不调用 `calibrate_model`
- 如果模型真实输入需求不满足，也必须立即停止
- 返回失败或停止原因，而不是强行给出率定结果

## 测试对话 6：CSV 输入 + Redis cache-only 输出

用户输入：

```text
请按 baseline-workflow 的工作流，用 exphydro 对 ./data/03604000.csv 跑一次模拟。不要手工读取 csv，先检查输入数据，结果不要落盘，只写入 redis。
```

期望 Codex 行为：

1. 调用 `inspect_hydro_data`
2. 调用 `run_simulation`
3. 推荐传入类似参数：

```json
{
  "model": "exphydro",
  "inputs": {
    "forcing": {
      "source_type": "csv",
      "path": "./data/03604000.csv"
    }
  },
  "output": {
    "result_source_type": "redis",
    "result_host": "127.0.0.1",
    "result_port": 6379
  }
}
```

期望结果：

- 返回 `run_id`
- 返回 `source_type=redis`
- 返回 `key=<redis_result_key>`
- 不返回本地 `output_path`

## 测试对话 7：低成本参数优化

用户输入：

```text
请使用 HydroModelMCP，对 ./data/03604000.csv 的 exphydro 做一次低成本参数优化。不要读取原始 csv，先检查输入是否满足率定要求，再用默认的小预算进行优化。
```

期望 Codex 行为：

1. 调用 `find_model`
2. 调用 `get_model_info`
3. 调用 `get_model_parameters`
4. 调用 `inspect_hydro_data`，并设置 `intended_use="calibration"`
5. 调用 `load_hydro_csv`
6. 可调用 `analyze_distribution_from_handle`
7. 调用 `configure_objectives`
8. 调用 `calibrate_model`
9. 调用 `diagnose_calibration`

`diagnose_calibration` 推荐调用方式：

- 首选显式传入 `calibration_result=<calibrate_model 返回体>`
- 如在同一会话中刚执行过 `calibrate_model`，可省略 `calibration_result`，由 MCP 自动使用最近一次校准结果

期望结果：

- `calibrate_model` 使用统一 v2 协议（`model + inputs`）
- 默认预算保持较小，例如 `maxiters` 在 `80` 左右
- 返回 `best_params`
- 返回 `best_objective`
- 返回诊断结果

## 测试对话 8：优化接口使用统一 v2 协议

用户输入：

```text
请按新的 optimization workflow 做一次参数优化。请使用统一 v2 请求结构调用 calibrate_model（model + inputs）。
```

期望 Codex 行为：

1. `inspect_hydro_data`
2. `load_hydro_csv`
3. `calibrate_model`

期望结果：

- `calibrate_model` 请求体中应包含 `model` 和 `inputs`
- `inputs` 中应包含 `forcing` 与 `observation` 源描述

## 测试对话 9：计算评价指标

用户输入：

```text
如果 ./data/03604000.csv 里存在观测流量列，请继续基于刚才的模拟结果计算评价指标，并把指标结果写到 ./result/metrics。
```

期望 Codex 行为：

1. 可先使用 `inspect_hydro_data` 确认 observed runoff 可用
2. 使用上一步的模拟输出 `output_path`
3. 调用 `compute_metrics`
4. 传入类似参数：

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

若仅给出输出目录（例如 `{"output_dir":"./result/metrics"}`），MCP 应尝试自动补全：

- `simulated`：优先使用同会话最近一次 `run_simulation` 的输出
- `observed`：优先按模拟结果文件前缀回推 `./data/<gauge_id>.csv`

期望结果：

- 返回 `metrics`
- 返回 `output_path`
- 结果中通常包含 `NSE`、`KGE`、`RMSE`、`PBIAS`、`R2`
- 自动补全成功时，应在 `warnings` 中说明补全来源

## 测试对话 10：完整验收版

用户输入：

```text
请严格按照 baseline-workflow skill 执行一次完整测试流程：
1. 识别模型
2. 检查 ./data 下的输入文件
3. 用 inspect_hydro_data 检查 ./data/03604000.csv 的输入要素和实测径流可用性
4. 用 exphydro 对 ./data/03604000.csv 运行一次模拟
5. 如果可以，计算评价指标
6. 输出完整执行 trace
7. 在结束前调用 clear_session_cache 清理会话缓存
```

期望 Codex 行为：

- 依次调用：
  - `find_model`
  - `get_model_info`
  - `get_model_parameters`
  - `list_workspace_files`
  - `inspect_hydro_data`
  - `run_simulation`
  - `compute_metrics`
  - `clear_session_cache`

期望结果结构：

- `MCP workflow test summary`
- `Test target`
- `Execution trace`
- `Status`
- `Outputs observed`
- `Failure`
- `Next action`

## 失败判定

如果出现以下情况，说明 MCP 接入仍有问题：

- Codex 没有调用任何 MCP 工具，直接改用 shell 读 CSV
- Codex 没有先做 `inspect_hydro_data` 就直接进入 simulation / calibration / validation
- `inspect_hydro_data` 已经指出输入不足，但 Codex 仍继续调用下游工具
- Codex 调用 `calibrate_model` 时仍使用旧参数结构（非 `model + inputs`）
- Codex 调用 MCP 时提示 server startup failed
- `run_simulation` 调用后没有生成预期产物，或 Redis 输出模式下没有返回缓存 key
- 对话结束时没有在合适的成功路径调用 `clear_session_cache`

## 一条最小测试输入

如果你只想快速测一次，直接给 Codex 这一句：

```text
请使用 HydroModelMCP，按 baseline-workflow skill 先检查 ./data/03604000.csv 的输入要素，再做一次 exphydro 模拟，并在结束前清理 session cache。
```

如果这句能稳定触发 `inspect_hydro_data`、`run_simulation` 和 `clear_session_cache`，说明新版基线流程已基本可用。
