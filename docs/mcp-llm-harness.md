# HydroModelMCP LLM 自动化测试 Harness 方案

本方案基于 `docs/test-dialogue.md` 扩展，目标是把“对话式手工验证”升级为“可重复、可分级、可追踪”的大模型自动化 MCP 测试工程。

## 1. 测试目标

该 Harness 重点验证四类能力：

- 协议正确性：工具请求是否遵循统一 v2 协议（`model + inputs (+ output/options)`）。
- 编排正确性：大模型是否按预期顺序调用工具，并在阻塞条件下立即停止。
- 结果完整性：关键输出字段是否完整、语义是否合理、缓存清理是否执行。
- 健壮性与漏洞暴露：错误参数、旧协议、缺失数据、歧义输入时是否正确报错与反馈。

## 2. Harness 架构

建议采用三层结构：

- Harness Orchestrator（大模型 + Skill）：负责读取用例、执行工具、判定通过/失败。
- MCP Under Test（HydroModelMCP）：被测对象。
- Fixture + Artifact 层：`./data` 输入、`./result` 输出、可选 Redis 缓存。

推荐报告输出目录：`./result/harness/`。

## 3. 运行模式

定义三个自动化等级：

- `smoke`：快速冒烟，验证核心可用性（5-6 个用例，2-5 分钟）。
- `standard`：日常回归，覆盖主流程与关键负向场景（10-12 个用例）。
- `full`：全面审计，包含鲁棒性/偏离/高风险负向场景（15+ 用例）。

默认策略：`fail_fast=true`（发现阻塞错误立即反馈并停止整套）。

## 4. 测试前准备

必需输入：

- `./data/03604000.csv`

可选负向样例（不存在则标记 `skipped_missing_fixture`）：

- `./data/forcing_only.csv`
- `./data/partial_forcing.csv`
- `./data/ambiguous_columns.csv`

环境建议：

- 默认文件输出：`./result`
- 可选 Redis：用于验证 `output.result_source_type="redis"`

## 5. 用例矩阵（核心）

下面每个用例均需记录：期望调用序列、实际调用序列、偏离点、是否阻塞。

### H-01 模型发现能力

- 目标：验证 `find_model/list_models/get_model_info/get_model_parameters` 可用。
- 关键断言：返回模型候选（通常含 `exphydro`），且能给出输入/参数信息。

### H-02 工作区轻量文件检查

- 目标：验证 `list_workspace_files` 能检查 `./data` 和 `./result`。
- 关键断言：不读取原始 CSV 内容；`./result` 不存在时不应导致崩溃。

### H-03 输入就绪检查（simulation）

- 目标：验证 `inspect_hydro_data` 的 simulation 就绪判定。
- 关键断言：返回 `forcing_elements`、`observed_runoff`、`model_check`、`readiness`。

### H-04 单次模拟（CSV 输出）

- 目标：验证 `run_simulation` v2 主路径。
- 关键断言：请求体使用 `model + inputs (+ output)`；返回 `run_id`、路径类产物、`params_source`。

### H-05 指标计算

- 目标：验证 `compute_metrics` 能基于模拟结果与观测计算指标。
- 关键断言：返回 `metrics`、`output_path`，指标中常见 `NSE/KGE/RMSE`。

### H-06 模拟 Redis cache-only 输出

- 目标：验证 `output.result_source_type="redis"`。
- 关键断言：返回 redis key；不强制本地 `output_path`。

### H-07 校准前阻塞判定（缺观测必须停止）

- 目标：验证 calibration 场景下的 fail-fast。
- 关键断言：`inspect_hydro_data` 若提示不可率定，后续不得调用 `calibrate_model`。

### H-08 低成本参数优化（v2 协议）

- 目标：验证 `calibrate_model` v2 输入契约与默认小预算路径。
- 关键断言：`calibrate_model` 请求体必须含 `model` 与 `inputs`，返回 `best_params`、`best_objective`。

### H-09 验证流程

- 目标：验证 `run_validation` 可执行并返回评价指标。
- 关键断言：请求体遵循 v2；包含 `calibration_period` 与 `validation_period`。
- 参数断言：
  - 优先显式提供 `inputs.parameters`（支持 inline object、json/csv/data_handle/calibration_result）
  - 若省略 `inputs.parameters`，仅在同会话存在最近一次校准/模拟参数时允许回退
  - 参数为部分字段时，应先尝试会话补全；补全失败必须 fail-fast
- period 断言：
  - 支持 `start_index/end_index`
  - 支持 `start/end` 日期格式
  - 若 forcing 无日期列且使用 `start/end`，允许降级到 synthetic timeline，并在 `warnings` 中明确标注

### H-10 集合参数模拟

- 目标：验证 `run_ensemble_parameters` 主路径。
- 关键断言：`parameter_sets` 非空；返回集合运行结果与基础统计。

### H-11 资源/模板/Prompt 面检查

- 目标：验证 `resources/templates/list`、`resources/read`、`prompts/list` 面可用。
- 关键断言：模板 URI 可发现；读具体模板可成功；Prompt 元数据可访问。

### H-12 负向：旧版 simulation 协议应被拒绝

- 目标：确认破坏式升级已经生效。
- 示例错误请求：顶层 `source_type/path`。
- 关键断言：应返回契约错误，不得静默接受旧字段。

### H-13 负向：旧版 calibration 协议应被拒绝

- 目标：确认 `calibrate_model` 不再接受旧接口。
- 示例错误请求：`model_name + data_handle` 直接率定（旧路径）。
- 关键断言：应返回契约错误或结构错误。

### H-14 负向：严格推断模式

- 目标：验证 `options.strict_infer=true` 的保护行为。
- 关键断言：当列名歧义明显时，应低置信度报错而非盲目运行。

### H-15 会话缓存清理

- 目标：验证 `clear_session_cache` 结束动作。
- 关键断言：清理中间态，不影响 `./result` 最终文件。

### H-16 稳定性回归（重复调用）

- 目标：检测同一会话重复调用下的稳定性。
- 关键断言：关键工具重复 3 次不应崩溃；错误应可解释。

## 6. 偏离检测规则（必须执行）

任一条命中即记为偏离，按严重级别判定：

- 未调用 MCP 工具，转而用 shell 直接读取原始 CSV。
- 未先 `inspect_hydro_data` 就直接进入 simulation/calibration/validation。
- 已出现 blocking issue 但仍继续下游调用。
- `run_simulation`/`calibrate_model` 请求未使用 v2 结构。
- 缺失关键输出字段却被错误判定为通过。
- 成功路径未执行 `clear_session_cache`。

## 7. 严重级别与闸门策略

- `P0` 阻塞：协议错误、服务不可用、阻塞后仍继续、核心工具不可调用。
- `P1` 高危：输出字段缺失、流程顺序偏离、缓存清理遗漏。
- `P2` 提示：非阻塞告警、性能退化、可恢复小问题。

建议门禁：

- 任意 `P0` => 整体 `failed`
- `P1 > 0` => 整体至少 `partial`
- 无 `P0/P1` => `passed`

## 8. 报告格式（人类 + 机器可读）

每次运行输出两部分：

- 人类可读摘要（用于快速定位）
- JSON 报告（用于自动汇总与趋势分析）

建议 JSON 结构：

```json
{
  "harness": "hydromodel-mcp",
  "mode": "standard",
  "fail_fast": true,
  "overall_status": "failed",
  "started_at": "2026-03-31T09:30:00Z",
  "finished_at": "2026-03-31T09:34:10Z",
  "cases": [
    {
      "id": "H-04",
      "status": "passed",
      "severity": "none",
      "expected_calls": ["inspect_hydro_data", "run_simulation"],
      "actual_calls": ["inspect_hydro_data", "run_simulation"],
      "mismatches": [],
      "artifacts": ["./result/simulation/xxx.csv"],
      "warnings": []
    },
    {
      "id": "H-12",
      "status": "failed",
      "severity": "P0",
      "mismatches": ["legacy simulation payload was accepted unexpectedly"],
      "failure_label": "contract_mismatch",
      "failed_step": "run_simulation"
    }
  ],
  "summary": {
    "passed": 7,
    "failed": 1,
    "skipped": 2,
    "p0": 1,
    "p1": 0,
    "p2": 1
  },
  "next_action": "Fix run_simulation contract guard for legacy flat payload"
}
```

## 9. 自动化执行建议

建议把该 Harness 作为日常工程质量闸门：

1. PR 前跑 `smoke`
2. 每日/每次发布跑 `standard`
3. 版本节点跑 `full`

出现失败时执行策略：

- 立即输出失败报告
- 指明失败用例、失败步骤、偏离原因、未执行的后续步骤
- 仅建议一个最小修复动作

## 10. 一键主提示词（给大模型）

```text
请使用 $hydromodel-mcp-harness skill 对 HydroModelMCP 执行一轮标准自动化回归（mode=standard, fail_fast=true）。
严格按照 docs/mcp-llm-harness.md 的用例矩阵执行，优先排查 MCP 协议和编排问题。
要求：
1) 每个用例记录 expected_calls 与 actual_calls；
2) 一旦出现错误或偏离，立即反馈失败标签、失败步骤和最小修复建议；
3) 输出人类可读摘要 + JSON 报告；
4) 成功路径末尾调用 clear_session_cache。
```

## 11. 与现有基线脚本关系

- `docs/test-dialogue.md` 仍适合人工逐条验证。
- 本文档用于高自动化、可回归、可门禁的 Harness 执行。
- 两者可以并存：先 `smoke` Harness，再用对话脚本定位复杂问题。
