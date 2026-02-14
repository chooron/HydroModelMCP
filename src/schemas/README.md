# Schema 定义说明

本文档说明 HydroModelMCP 项目中统一的 schema 定义结构。

## 目录结构

```
src/schemas/
├── Schemas.jl                    # 主模块文件，统一导出所有 schema
├── common/
│   └── base_schemas.jl          # 基础通用字段定义
├── config/
│   └── solver_schemas.jl        # 求解器和插值器配置
├── metrics/
│   └── metric_schemas.jl        # 评价指标相关
└── algorithms/
    └── algorithm_schemas.jl     # 优化算法和方法相关
```

## Schema 分类

### 1. 基础通用 Schema (`common/base_schemas.jl`)

包含所有工具共享的基础字段定义：

**模型相关**
- `MODEL_NAME_SCHEMA`: 模型名称
- `PARAMETERS_SCHEMA`: 模型参数（对象格式）
- `PARAMS_ARRAY_SCHEMA`: 模型参数（数组格式）
- `PARAMETER_SETS_SCHEMA`: 参数集合列表
- `FIXED_PARAMS_SCHEMA`: 固定参数
- `PARAM_BOUNDS_SCHEMA`: 自定义参数范围
- `INIT_STATES_SCHEMA`: 初始状态

**数据源相关**
- `FORCING_SCHEMA`: 驱动数据配置
- `OBSERVATION_SCHEMA`: 观测数据配置
- `OBS_COLUMN_SCHEMA`: 观测值列名
- `SIM_COLUMN_SCHEMA`: 模拟值列名
- `DATA_SOURCE_SCHEMA`: 数据源配置（通用）

**时间范围相关**
- `CALIBRATION_PERIOD_SCHEMA`: 率定期时间范围
- `VALIDATION_PERIOD_SCHEMA`: 验证期时间范围

**其他**
- `INPUT_MAPPING_SCHEMA`: 输入变量映射

### 2. 配置 Schema (`config/solver_schemas.jl`)

求解器和插值器相关配置：

- `SOLVER_SCHEMA`: 求解器类型（完整版本：ODE, DISCRETE, MUTABLE, IMMUTABLE）
- `SOLVER_SIMPLE_SCHEMA`: 求解器类型（简化版本：ODE, DISCRETE）
- `INTERPOLATOR_SCHEMA`: 插值方式（完整版本：LINEAR, CONSTANT, DIRECT）
- `INTERPOLATOR_SIMPLE_SCHEMA`: 插值方式（简化版本：LINEAR, CONSTANT）
- `PARALLEL_SCHEMA`: 并行执行开关
- `SAVE_TO_STORAGE_SCHEMA`: 存储开关
- `CONFIG_SCHEMA`: 模拟配置

### 3. 评价指标 Schema (`metrics/metric_schemas.jl`)

评价指标相关定义：

- `METRICS_SCHEMA`: 评价指标列表（NSE, KGE, LogNSE, LogKGE, RMSE, R2, PBIAS）
- `OBJECTIVE_SCHEMA`: 单个目标函数（用于单目标校准）
- `OBJECTIVES_SCHEMA`: 多个目标函数（用于多目标校准）
- `LOG_TRANSFORM_SCHEMA`: 对数变换开关

### 4. 算法 Schema (`algorithms/algorithm_schemas.jl`)

优化算法和方法相关定义：

**单目标优化**
- `ALGORITHM_SCHEMA`: 单目标优化算法（BBO, DE, PSO, CMAES, ECA）
- `MAXITERS_SCHEMA`: 最大迭代次数
- `N_TRIALS_SCHEMA`: 独立试验次数

**多目标优化**
- `MULTI_ALGORITHM_SCHEMA`: 多目标优化算法（NSGA2, NSGA3）
- `POPULATION_SIZE_SCHEMA`: 种群大小

**敏感性分析**
- `SENSITIVITY_METHOD_SCHEMA`: 敏感性分析方法（morris, sobol）
- `N_SAMPLES_SCHEMA`: 采样数
- `THRESHOLD_SCHEMA`: 敏感性阈值

**参数采样**
- `SAMPLING_METHOD_SCHEMA`: 参数采样方法（lhs, sobol, random）
- `SAMPLING_N_SAMPLES_SCHEMA`: 采样数量

**数据划分**
- `SPLIT_METHOD_SCHEMA`: 数据划分方法（recent_first, split_sample, use_all）
- `RATIO_SCHEMA`: 划分比例
- `WARMUP_SCHEMA`: 预热期

**诊断相关**
- `CALIBRATION_RESULT_SCHEMA`: 校准结果（用于诊断）
- `BOUNDARY_TOLERANCE_SCHEMA`: 边界容差
- `CONVERGENCE_THRESHOLD_SCHEMA`: 收敛阈值
- `PLATEAU_WINDOW_SCHEMA`: 平台期窗口

**业务目标**
- `GOAL_SCHEMA`: 业务目标（general_fit, peak_flows, low_flows, water_balance, dynamics）
- `BUDGET_SCHEMA`: 计算预算（low, medium, high）
- `SENSITIVITY_SAMPLES_SCHEMA`: 敏感性分析采样数

## 使用方法

### 在工具定义中使用

```julia
using .Schemas

# 构建工具的 input_schema
my_tool = MCPTool(
    name = "my_tool",
    description = "...",
    input_schema = Dict{String,Any}(
        "type" => "object",
        "properties" => Dict{String,Any}(
            "model_name" => MODEL_NAME_SCHEMA,
            "forcing" => FORCING_SCHEMA,
            "solver" => SOLVER_SCHEMA,
            "interpolator" => INTERPOLATOR_SCHEMA,
            "objective" => OBJECTIVE_SCHEMA
        ),
        "required" => ["model_name", "forcing"]
    ),
    handler = function(args)
        # ...
    end
)
```

### 优势

1. **代码复用**: 避免在多个工具中重复定义相同的 schema
2. **统一管理**: 所有 schema 定义集中在一个地方，便于维护和更新
3. **类型安全**: 确保所有工具使用一致的字段定义
4. **易于扩展**: 添加新的 schema 只需在对应的文件中定义即可

## 示例

### 模拟工具 (simulation.jl)

```julia
using .Schemas

simulation_tool = MCPTool(
    name = "run_simulation",
    input_schema = Dict{String,Any}(
        "type" => "object",
        "properties" => Dict{String,Any}(
            "model_name" => MODEL_NAME_SCHEMA,
            "forcing" => FORCING_SCHEMA,
            "params" => PARAMS_ARRAY_SCHEMA,
            "solver" => SOLVER_SCHEMA,
            "interpolator" => INTERPOLATOR_SIMPLE_SCHEMA
        ),
        "required" => ["model_name", "forcing"]
    ),
    # ...
)
```

### 校准工具 (calibration.jl)

```julia
using .Schemas

calibrate_tool = MCPTool(
    name = "calibrate_model",
    input_schema = Dict{String,Any}(
        "type" => "object",
        "properties" => Dict{String,Any}(
            "model_name" => MODEL_NAME_SCHEMA,
            "forcing" => FORCING_SCHEMA,
            "observation" => OBSERVATION_SCHEMA,
            "obs_column" => OBS_COLUMN_SCHEMA,
            "objective" => OBJECTIVE_SCHEMA,
            "algorithm" => ALGORITHM_SCHEMA,
            "maxiters" => MAXITERS_SCHEMA,
            "fixed_params" => FIXED_PARAMS_SCHEMA,
            "param_bounds" => PARAM_BOUNDS_SCHEMA
        ),
        "required" => ["model_name", "forcing", "observation", "obs_column"]
    ),
    # ...
)
```

## 维护指南

### 添加新的 Schema

1. 确定 schema 的类别（common/config/metrics/algorithms）
2. 在对应的文件中添加 schema 定义
3. 在 `Schemas.jl` 中添加导出语句
4. 更新本 README 文档

### 修改现有 Schema

1. 在对应的文件中修改 schema 定义
2. 检查所有使用该 schema 的工具是否需要更新
3. 运行测试确保没有破坏现有功能

### 删除 Schema

1. 确认没有工具使用该 schema
2. 从对应的文件中删除定义
3. 从 `Schemas.jl` 中删除导出语句
4. 更新本 README 文档
