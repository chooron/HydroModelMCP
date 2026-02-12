# Mai (2023) Environmental Calibration Strategies: Implementation Guide for HydroAgent

**Context:** You are building an autonomous hydrological modeling agent. The following 10 strategies are the "Laws of Physics" for this agent's decision-making process. They are grouped by the workflow stages: **PREPARE**, **EXECUTE**, and **CHECK**.

---

## Phase I: PREPARE (准备阶段 - The Strategist)

### Strategy 1: Sensitivity Analysis (参数筛选)
*   **Core Logic:** 不要校准所有参数。识别出哪些参数对输出真正有影响。
*   **Agent Action (Tool):** 调用敏感性分析工具（如 Morris Method 或 Efficient Elementary Effects - EEE）。
*   **Decision Rule (DSpy):**
    *   IF `parameter_sensitivity` < `threshold`: Set parameter to `FIXED` (default value).
    *   IF `parameter_sensitivity` >= `threshold`: Set parameter to `CALIBRATABLE`.
*   **Value:** 减少搜索空间，提高收敛速度。

### Strategy 2: Handling Parameter Constraints (参数约束处理)
*   **Core Logic:** 参数往往不是独立的（例如：`soil_depth_layer1` 必须小于 `soil_depth_total`，或者某些比例之和必须为1）。直接采样会导致模型崩溃。
*   **Implementation (Pydantic/Python):**
    *   **Rule A (Summation):** 如果参数之和需为常数 $\rightarrow$ 使用 **Pie-share sampling** 方法重构输入。
    *   **Rule B (Inequality):** 如果 $x_1 < x_2$ $\rightarrow$ 使用 **Delta method**。即采样 $x_1$ 和 $\delta$，定义 $x_2 = x_1 + \delta$。
*   **Constraint:** 在 Pydantic Model 中使用 `root_validator` 拦截无效参数组合。

### Strategy 3: Data Magnitude Transformation (数据量级变换)
*   **Core Logic:** 如果观测数据跨越多个数量级（如流量从 $10^{-1}$ 到 $10^3$），标准误差指标会偏向大值。
*   **Decision Rule (DSpy):**
    *   IF `data_max / data_min` > $10^2$ (or distinct orders of magnitude): Apply **Log-transformation** to both simulation and observation data before calculating metrics.
    *   ELSE: Use Native space.

### Strategy 4: Data Splitting Strategy (数据划分)
*   **Core Logic:** 如何划分校准（Calibration）和验证（Validation）数据取决于目标。
*   **Decision Rule (DSpy):**
    *   **Scenario A (Operational Deployment):** 目标是预测未来 $\rightarrow$ 校准集应包含由近及远的最新数据（Most recent data）。
    *   **Scenario B (Process Understanding):** 目标是稳健性 $\rightarrow$ Split-sample test (e.g., Klemest test).
    *   **Scenario C (Maximum Robustness):** 在部署前 $\rightarrow$ 使用**所有可用数据**进行最终校准（不保留验证集）。

---

## Phase II: EXECUTE (执行阶段 - The Operator)

### Strategy 5: Parameter Sampling (初始采样)
*   **Core Logic:** 避免纯随机采样（Monte Carlo）。
*   **Agent Action:** 在算法初始阶段，必须使用 **Latin Hypercube Sampling (LHS)** 或 **Sobol Sequences** 来保证参数空间的均匀覆盖。
*   **Value:** 确保不会因为初始采样不均而陷入局部最优。

### Strategy 6: Parameter Ranges (范围界定)
*   **Core Logic:** 范围过宽 $\rightarrow$ 效率低；范围过窄 $\rightarrow$ 错过最优解。
*   **Agent Loop (LangGraph State):**
    *   这是一个迭代过程。初始设置较宽的物理合理范围。
    *   如果在 Phase III 发现最优值卡在边界上，必须回到此步骤扩大范围。

### Strategy 7: Objective Function Selection (目标函数选择)
*   **Core Logic:** 目标函数必须匹配用户意图。
*   **Mapping Table (DSpy Skill):**
    *   **Goal: General Fit** $\rightarrow$ KGE (Kling-Gupta Efficiency).
    *   **Goal: Peak Flows/Floods** $\rightarrow$ NSE (Nash-Sutcliffe) or KGE.
    *   **Goal: Low Flows/Droughts** $\rightarrow$ LogNSE or LogKGE.
    *   **Goal: Water Balance/Volume** $\rightarrow$ PBIAS (Percent Bias).
    *   **Goal: Dynamics/Pattern** $\rightarrow$ $r^2$ (Squared Correlation).

### Strategy 8: Calibration Algorithm Selection (算法选择)
*   **Core Logic:** 根据计算预算（Budget）和问题维度选择算法。
*   **Decision Rule (DSpy):**
    *   IF `budget` is LOW (limited model runs) OR `parameters` are HIGH dimensional $\rightarrow$ Select **DDS (Dynamically Dimensioned Search)**. (DDS 非常适合预算有限的情况).
    *   IF `budget` is HIGH/Unlimited $\rightarrow$ Select **SCE (Shuffled Complex Evolution)** or **PSO (Particle Swarm)**.

### Strategy 9: Multi-Objective Calibration (多目标优化)
*   **Core Logic:** 如果需要同时满足两个目标（如高流和低流）。
*   **Agent Action:** 使用 **PA-DDS** (Pareto Archived DDS) 算法。
*   **Check:** 检查生成的 Pareto Front。如果 Front 是直角/简并的（Degenerated），说明目标可能不冲突，或者设置有问题。

---

## Phase III: CHECK (检查阶段 - The Critic)

### Strategy 10: Diagnosing Calibration Performance (诊断与反馈)
这是智能体实现闭环（Self-Improvement）的关键。智能体必须执行 "Checklist" (Refer to Fig. 11 in paper)。

*   **Check 1: Convergence (收敛性)**
    *   *Detection:* 运行多次独立试验（Independent Trials）。
    *   *Logic:* 如果多次试验的结果（目标函数值）差异巨大（Large spread） $\rightarrow$ **Action:** `INCREASE_BUDGET` (Loop back to Execute).
*   **Check 2: Parameter Boundaries (边界触达)**
    *   *Detection:* 最优参数值是否等于 Upper Bound 或 Lower Bound？
    *   *Logic:* IF `param_value` == `bound` $\rightarrow$ **Action:** `WIDEN_RANGE` (Loop back to Prepare/Execute).
*   **Check 3: Objective Plateau (平台期)**
    *   *Detection:* 目标函数曲线在结束前是否已经拉平（Plateau）？
    *   *Logic:* 如果曲线在结束时仍在下降 $\rightarrow$ **Action:** `INCREASE_BUDGET`.
*   **Check 4: The "Hat-Trick"**
    *   只有当 (1) 模型拟合良好，(2) 参数收敛一致，(3) 目标函数收敛一致，才算成功。

---

### Summary for the Coding Agent:
*   **Pydantic** should enforce **Strategy 2 & 6**.
*   **DSpy** should handle the heuristic decisions in **Strategy 1, 3, 4, 7, 8**.
*   **LangGraph** should manage the iterative loops defined in **Strategy 10** (The logic of "If Diagnosis Fails -> Modify State -> Rerun").
*   **FastMCP** connects to RavenPy/Ostrich to physically execute **Strategy 5 & 8**.