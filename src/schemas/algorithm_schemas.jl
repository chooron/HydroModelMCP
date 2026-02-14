"""
优化算法相关 Schema 定义
"""

# ==============================================================================
# 单目标优化算法 Schema
# ==============================================================================

"""单目标优化算法"""
const ALGORITHM_SCHEMA = Dict{String,Any}(
    "type" => "string",
    "enum" => ["BBO", "DE", "PSO", "CMAES", "ECA"],
    "description" => "优化算法。默认 BBO"
)

"""最大迭代次数"""
const MAXITERS_SCHEMA = Dict{String,Any}(
    "type" => "integer",
    "description" => "最大迭代次数。默认 1000"
)

"""独立试验次数"""
const N_TRIALS_SCHEMA = Dict{String,Any}(
    "type" => "integer",
    "description" => "独立试验次数(用于收敛性检查)。默认 1"
)

# ==============================================================================
# 多目标优化算法 Schema
# ==============================================================================

"""多目标优化算法"""
const MULTI_ALGORITHM_SCHEMA = Dict{String,Any}(
    "type" => "string",
    "enum" => ["NSGA2", "NSGA3"],
    "description" => "多目标算法。默认 NSGA2"
)

"""种群大小"""
const POPULATION_SIZE_SCHEMA = Dict{String,Any}(
    "type" => "integer",
    "description" => "种群大小。默认 50"
)

# ==============================================================================
# 敏感性分析方法 Schema
# ==============================================================================

"""敏感性分析方法"""
const SENSITIVITY_METHOD_SCHEMA = Dict{String,Any}(
    "type" => "string",
    "enum" => ["morris", "sobol"],
    "description" => "敏感性分析方法。默认 morris"
)

"""敏感性分析采样数"""
const N_SAMPLES_SCHEMA = Dict{String,Any}(
    "type" => "integer",
    "description" => "采样数。Morris 默认 100，Sobol 默认 1000"
)

"""敏感性阈值"""
const THRESHOLD_SCHEMA = Dict{String,Any}(
    "type" => "number",
    "description" => "敏感性阈值，低于此值的参数建议固定。默认 0.1"
)

# ==============================================================================
# 参数采样方法 Schema
# ==============================================================================

"""参数采样方法"""
const SAMPLING_METHOD_SCHEMA = Dict{String,Any}(
    "type" => "string",
    "enum" => ["lhs", "sobol", "random"],
    "description" => "采样方法，默认 lhs"
)

"""采样数量"""
const SAMPLING_N_SAMPLES_SCHEMA = Dict{String,Any}(
    "type" => "integer",
    "description" => "采样数量，默认 100"
)

# ==============================================================================
# 数据划分方法 Schema
# ==============================================================================

"""数据划分方法"""
const SPLIT_METHOD_SCHEMA = Dict{String,Any}(
    "type" => "string",
    "enum" => ["recent_first", "split_sample", "use_all"],
    "description" => "划分方法。默认 split_sample"
)

"""划分比例"""
const RATIO_SCHEMA = Dict{String,Any}(
    "type" => "number",
    "description" => "校准集占比 (0-1)，默认 0.7"
)

"""预热期"""
const WARMUP_SCHEMA = Dict{String,Any}(
    "type" => "integer",
    "description" => "预热期天数，默认 365"
)

# ==============================================================================
# 诊断相关 Schema
# ==============================================================================

"""校准结果（用于诊断）"""
const CALIBRATION_RESULT_SCHEMA = Dict{String,Any}(
    "type" => "object",
    "description" => "calibrate_model 工具的返回结果（直接传入）"
)

"""边界容差"""
const BOUNDARY_TOLERANCE_SCHEMA = Dict{String,Any}(
    "type" => "number",
    "description" => "边界容差比例。默认 0.01"
)

"""收敛阈值"""
const CONVERGENCE_THRESHOLD_SCHEMA = Dict{String,Any}(
    "type" => "number",
    "description" => "收敛判定阈值(多次试验结果的变异系数)。默认 0.05"
)

"""平台期窗口"""
const PLATEAU_WINDOW_SCHEMA = Dict{String,Any}(
    "type" => "integer",
    "description" => "平台期检测窗口大小。默认 50"
)

# ==============================================================================
# 业务目标相关 Schema
# ==============================================================================

"""业务目标"""
const GOAL_SCHEMA = Dict{String,Any}(
    "type" => "string",
    "enum" => ["general_fit", "peak_flows", "low_flows", "water_balance", "dynamics"],
    "description" => "业务目标。默认 general_fit"
)

"""计算预算"""
const BUDGET_SCHEMA = Dict{String,Any}(
    "type" => "string",
    "enum" => ["low", "medium", "high"],
    "description" => "计算预算。low(<500次), medium(500-5000次), high(>5000次)。默认 medium"
)

"""敏感性分析采样数（用于初始化设置）"""
const SENSITIVITY_SAMPLES_SCHEMA = Dict{String,Any}(
    "type" => "integer",
    "description" => "敏感性分析采样数。默认 50"
)

export ALGORITHM_SCHEMA, MAXITERS_SCHEMA, N_TRIALS_SCHEMA,
       MULTI_ALGORITHM_SCHEMA, POPULATION_SIZE_SCHEMA,
       SENSITIVITY_METHOD_SCHEMA, N_SAMPLES_SCHEMA, THRESHOLD_SCHEMA,
       SAMPLING_METHOD_SCHEMA, SAMPLING_N_SAMPLES_SCHEMA,
       SPLIT_METHOD_SCHEMA, RATIO_SCHEMA, WARMUP_SCHEMA,
       CALIBRATION_RESULT_SCHEMA, BOUNDARY_TOLERANCE_SCHEMA,
       CONVERGENCE_THRESHOLD_SCHEMA, PLATEAU_WINDOW_SCHEMA,
       GOAL_SCHEMA, BUDGET_SCHEMA, SENSITIVITY_SAMPLES_SCHEMA
