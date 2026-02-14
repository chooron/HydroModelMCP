"""
Schema 定义模块

统一管理所有 MCP 工具的 input_schema 定义，避免重复代码。

# 使用方法

```julia
using .Schemas

# 构建工具的 input_schema
input_schema = Dict{String,Any}(
    "type" => "object",
    "properties" => Dict{String,Any}(
        "model_name" => MODEL_NAME_SCHEMA,
        "forcing" => FORCING_SCHEMA,
        "solver" => SOLVER_SCHEMA,
        "interpolator" => INTERPOLATOR_SCHEMA
    ),
    "required" => ["model_name", "forcing"]
)
```

# Schema 分类

- `common/base_schemas.jl`: 基础通用字段（model_name, forcing, observation 等）
- `config/solver_schemas.jl`: 求解器和插值器配置
- `metrics/metric_schemas.jl`: 评价指标相关
- `algorithms/algorithm_schemas.jl`: 优化算法和方法相关
"""
module Schemas

# 导入所有 schema 定义
include("base_schemas.jl")
include("solver_schemas.jl")
include("metric_schemas.jl")
include("algorithm_schemas.jl")

# 重新导出所有 schema
# 基础 Schema
export MODEL_NAME_SCHEMA, PARAMETERS_SCHEMA, PARAMS_ARRAY_SCHEMA,
       PARAMETER_SETS_SCHEMA, FIXED_PARAMS_SCHEMA, PARAM_BOUNDS_SCHEMA,
       INIT_STATES_SCHEMA, FORCING_SCHEMA, OBSERVATION_SCHEMA,
       OBS_COLUMN_SCHEMA, SIM_COLUMN_SCHEMA, DATA_SOURCE_SCHEMA,
       CALIBRATION_PERIOD_SCHEMA, VALIDATION_PERIOD_SCHEMA,
       INPUT_MAPPING_SCHEMA

# 配置 Schema
export SOLVER_SCHEMA, SOLVER_SIMPLE_SCHEMA,
       INTERPOLATOR_SCHEMA, INTERPOLATOR_SIMPLE_SCHEMA,
       PARALLEL_SCHEMA, SAVE_TO_STORAGE_SCHEMA, CONFIG_SCHEMA

# 指标 Schema
export METRICS_SCHEMA, OBJECTIVE_SCHEMA, OBJECTIVES_SCHEMA, LOG_TRANSFORM_SCHEMA

# 算法 Schema
export ALGORITHM_SCHEMA, MAXITERS_SCHEMA, N_TRIALS_SCHEMA,
       MULTI_ALGORITHM_SCHEMA, POPULATION_SIZE_SCHEMA,
       SENSITIVITY_METHOD_SCHEMA, N_SAMPLES_SCHEMA, THRESHOLD_SCHEMA,
       SAMPLING_METHOD_SCHEMA, SAMPLING_N_SAMPLES_SCHEMA,
       SPLIT_METHOD_SCHEMA, RATIO_SCHEMA, WARMUP_SCHEMA,
       CALIBRATION_RESULT_SCHEMA, BOUNDARY_TOLERANCE_SCHEMA,
       CONVERGENCE_THRESHOLD_SCHEMA, PLATEAU_WINDOW_SCHEMA,
       GOAL_SCHEMA, BUDGET_SCHEMA, SENSITIVITY_SAMPLES_SCHEMA

end # module Schemas
