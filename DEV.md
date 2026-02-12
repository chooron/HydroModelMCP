**项目：Hydromodels Calibration MCP Service (HMCS)**
**基于文档：** *Ten strategies towards successful calibration of environmental models (J. Mai, 2023)*
**目标：** 构建基于 Julia 的 MCP (Model Context Protocol) 服务，封装 `hydromodels.jl`，实现环境模型自动化率定全生命周期管理。

---

### 1. 总体架构设计

系统分为三层：
1.  **Core Logic Layer (`CalibrationCore.jl`)**: 实现论文中的十大策略算法逻辑。
2.  **Integration Layer (`ModelInterface.jl`)**: 负责与 `hydromodels.jl` 的交互适配。
3.  **Service Layer (`MCPServer.jl`)**: 通过 MCP 协议暴露功能，支持 Agent 调用。

---

### 2. 核心功能模块开发规范 (Core Logic Layer)

你需要开发以下具体模块及其函数接口。所有算法必须严格对应论文 Appendices 中的数学定义。

#### 模块 A: 预处理与敏感性分析 (Preparation)
**对应策略：** 1 (Sensitivity), 2 (Constraints), 3 (Data Transform), 4 (Data Splitting), 5 (Sampling)

*   **`ParameterSampling.jl`**
    *   **`latin_hypercube_sampling(bounds, n_samples)`**: 实现拉丁超立方采样 (LHS)。
    *   **`pie_share_sampling(n_params, threshold)`**: **必须实现**。对应 Appendix D (Eq. D.1)。用于处理参数加和为定值（如 1 或 $c$）的约束采样。
    *   **`delta_method_sampling(bounds, constraints)`**: 实现论文 Table 1 中的 Delta Method，处理 $x_1 < x_2$ 类型的约束。

*   **`SensitivityAnalysis.jl`**
    *   **`efficient_elementary_effects(model_func, bounds, r_trajectories)`**: 实现 Morris 方法/EEE (Efficient Elementary Effects)。
    *   **`screen_parameters(sensitivity_indices, threshold)`**: 返回如果是 "non-informative" 的参数列表，用于固定参数值。

*   **`DataOps.jl`**
    *   **`transform_data(data::Vector, method::Symbol)`**:
        *   支持 `:log` ($y = \ln(x)$), `:log10`, `:boxcox`。
        *   对应论文 2.3 节，处理数量级差异大的数据。
    *   **`split_data(timeseries, ratio, method)`**:
        *   `:sequential`: 传统前后切分。
        *   `:independent`: 对应论文 2.4 节，支持非连续时间段切分以进行更稳健的验证。

#### 模块 B: 目标函数库 (Objective Functions)
**对应策略：** 7 (Objective Functions)
**对应文档：** Appendix B

必须实现以下指标计算函数（输入均为 `sim`, `obs`）：
*   **`calc_rmse()`**: Root Mean Square Error (Eq. B.1).
*   **`calc_nse()`**: Nash-Sutcliffe Efficiency (Eq. B.2).
*   **`calc_nse_log()`**: Log-transformed NSE.
*   **`calc_kge()`**: Kling-Gupta Efficiency (Eq. B.3).
*   **`calc_kge_components()`**: 返回 `(r, alpha, beta)` 元组 (Eq. B.4 - B.6).
*   **`calc_r2()`**: Squared Pearson Correlation (Eq. B.9).
*   **`calc_pbias()`**: Percent Bias (Eq. B.10).
*   **`calc_weighted_metric(metrics::Vector, weights::Vector)`**: 支持多目标加权单目标化。

#### 模块 C: 优化算法引擎 (Execution)
**对应策略：** 6 (Ranges), 8 (Algorithms), 9 (Multi-objective)
**对应文档：** Appendix C

需实现或封装现有库以支持以下算法，并暴露论文 Table C.3 中的超参数：

*   **`OptimizationAlgorithms.jl`**
    *   **`dds(objective_func, bounds, max_iter, perturbation=0.2)`**: Dynamically Dimensioned Search (Appendix C.1). 必须支持单目标快速收敛。
    *   **`sce_ua(objective_func, bounds, complex_params...)`**: Shuffled Complex Evolution (Appendix C.2). 必须暴露 `NumComplexes`, `NumPointsPerComplex` 参数。
    *   **`pso(objective_func, bounds, swarm_size, ...)`**: Particle Swarm Optimization (Appendix C.3).
    *   **`padds(objective_funcs, bounds, max_iter)`**: Pareto Archived DDS (Appendix C.4). **核心需求**：用于多目标率定，需返回 Pareto 前沿集合。

#### 模块 D: 诊断与后处理 (Checking)
**对应策略：** 10 (Diagnostics)

*   **`Diagnostics.jl`**
    *   **`check_convergence(history::Vector)`**: 分析目标函数轨迹，检测是否达到 Plateau。
    *   **`check_parameter_spread(trials::Vector{OptimizationResult})`**: 分析多次独立试验（Trials）的参数分布。若 Spread 过大，标记该参数为 "Unidentifiable"。
    *   **`check_pareto_degeneracy(pareto_front)`**: 对应论文 2.9 节，检测多目标前沿是否退化（Degenerated）成单点或直线。

---

### 3. 数据结构定义 (Structs)

在 `hydromodels.jl` 上层定义以下 Structs 以标准化数据流：

```julia
# 标定配置对象
struct CalibrationConfig
    model_id::String
    parameters::Vector{ParameterDef} # 包含 bounds, constraints
    objective_metrics::Vector{Symbol} # [:kge, :rmse]
    algorithm::Symbol # :dds, :sce, :padds
    budget::Int # Max evaluations
    n_trials::Int # 独立重复试验次数 (Strategy 10 要求)
    data_split_config::DataSplitConfig
end

# 诊断报告对象
struct DiagnosticReport
    convergence_status::Symbol # :converged, :not_converged, :unknown
    parameter_identifiability::Dict{Symbol, Float64} # 参数收敛程度
    pareto_shape::Symbol # :normal, :degenerated (仅多目标)
    recommendations::Vector{String} # 基于 Check 阶段生成的建议 (e.g., "Increase budget", "Narrow range for param X")
end
```

---

### 4. MCP Server 接口规范

基于 Model Context Protocol，你需要暴露以下 Tool 给 LLM Agent 使用：

#### Tool 1: `init_calibration_setup`
*   **输入**: 模型元数据，参数先验范围。
*   **功能**: 执行策略 1 & 2。运行敏感性分析，剔除不敏感参数，应用 Delta/Pie-share 变换处理参数约束。
*   **返回**: 优化后的参数列表建议，建议的采样策略。

#### Tool 2: `configure_objectives`
*   **输入**: 业务目标描述 (e.g., "关注洪水峰值" 或 "关注枯水期")。
*   **功能**: 映射到具体的 Metric (策略 7)。
    *   High-flows -> `calc_kge_high` (Eq. B.8).
    *   Low-flows -> `calc_kge_log`.
*   **返回**: 选定的目标函数组合。

#### Tool 3: `run_calibration_experiment`
*   **输入**: `CalibrationConfig` JSON。
*   **功能**:
    1.  依据策略 4 切分数据。
    2.  依据策略 8 选择算法（DDS/SCE/PADDS）。
    3.  并行执行 N 次 Trials (策略 10 核心要求)。
    4.  调用 `hydromodels.jl` 进行模拟。
*   **返回**: 试验结果 ID，最优参数集，Pareto 前沿数据。

#### Tool 4: `diagnose_results`
*   **输入**: 试验结果 ID。
*   **功能**: 执行策略 10 的 Checklist。
    *   检查参数是否顶到了边界（Boundary hit）。
    *   检查多次试验的一致性（Consistency）。
    *   检查拟合残差分布。
*   **返回**: `DiagnosticReport` 文本描述，包含改进建议（如 "参数 x2 范围过窄，建议放宽"）。

---

### 5. 开发注意事项

1.  **并行计算**: 论文强调多次 Trials (e.g., 10次) 来评估随机性。Julia 的 `Distributed` 或 `Threads` 必须用于 `run_calibration_experiment`。
2.  **Hydromodels 抽象**: `hydromodels.jl` 的调用应包装在 `function evaluate_model(params)` 中，确保优化器不直接依赖具体模型结构。
3.  **约束处理**: 必须严格实现 Pie-share method，这是论文作者自引用的关键技术点，用于处理复杂的 sum-to-one 约束。
4.  **Budget 管理**: 算法必须严格遵守 Function Evaluations 的次数限制，不可超时。

请按此规格书开始构建 `CalibrationCore` 和 `MCPServer` 模块。