"""
求解器和插值器相关 Schema 定义
"""

# ==============================================================================
# 求解器 Schema
# ==============================================================================

"""求解器类型（完整版本）"""
const SOLVER_SCHEMA = Dict{String,Any}(
    "type" => "string",
    "enum" => ["ODE", "DISCRETE", "MUTABLE", "IMMUTABLE"],
    "default" => "DISCRETE",
    "description" => "求解器类型"
)

"""求解器类型（简化版本，仅 ODE 和 DISCRETE）"""
const SOLVER_SIMPLE_SCHEMA = Dict{String,Any}(
    "type" => "string",
    "enum" => ["ODE", "DISCRETE"],
    "description" => "求解器类型"
)

# ==============================================================================
# 插值器 Schema
# ==============================================================================

"""插值方式（完整版本）"""
const INTERPOLATOR_SCHEMA = Dict{String,Any}(
    "type" => "string",
    "enum" => ["LINEAR", "CONSTANT", "DIRECT"],
    "default" => "LINEAR",
    "description" => "插值方式"
)

"""插值方式（简化版本，仅 LINEAR 和 CONSTANT）"""
const INTERPOLATOR_SIMPLE_SCHEMA = Dict{String,Any}(
    "type" => "string",
    "enum" => ["LINEAR", "CONSTANT"],
    "description" => "插值方式"
)

# ==============================================================================
# 其他配置 Schema
# ==============================================================================

"""并行执行开关"""
const PARALLEL_SCHEMA = Dict{String,Any}(
    "type" => "boolean",
    "description" => "是否并行执行（默认true）",
    "default" => true
)

"""存储开关"""
const SAVE_TO_STORAGE_SCHEMA = Dict{String,Any}(
    "type" => "boolean",
    "description" => "是否保存到存储层（默认true）",
    "default" => true
)

"""模拟配置"""
const CONFIG_SCHEMA = Dict{String,Any}(
    "type" => "object",
    "description" => "高级模拟配置 (例如指定输出变量)",
    "properties" => Dict{String,Any}(
        "output_variable" => Dict{String,Any}("type" => "string")
    )
)

export SOLVER_SCHEMA, SOLVER_SIMPLE_SCHEMA,
       INTERPOLATOR_SCHEMA, INTERPOLATOR_SIMPLE_SCHEMA,
       PARALLEL_SCHEMA, SAVE_TO_STORAGE_SCHEMA, CONFIG_SCHEMA
