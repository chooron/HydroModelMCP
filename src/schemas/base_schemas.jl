"""
基础通用 Schema 定义

包含所有工具共享的基础字段定义，如 model_name, forcing, observation 等。
"""

# ==============================================================================
# 模型相关 Schema
# ==============================================================================

"""模型名称字段"""
const MODEL_NAME_SCHEMA = Dict{String,Any}(
    "type" => "string",
    "description" => "模型名称"
)

"""模型参数字段（对象格式）"""
const PARAMETERS_SCHEMA = Dict{String,Any}(
    "type" => "object",
    "description" => "模型参数字典"
)

"""模型参数字段（数组格式）"""
const PARAMS_ARRAY_SCHEMA = Dict{String,Any}(
    "type" => "array",
    "items" => Dict{String,Any}("type" => "number"),
    "description" => "模型参数数组。如果省略，将自动生成随机参数。"
)

"""参数集合列表"""
const PARAMETER_SETS_SCHEMA = Dict{String,Any}(
    "type" => "array",
    "description" => "参数集列表，每个元素是一个参数字典",
    "items" => Dict{String,Any}("type" => "object")
)

"""固定参数字段"""
const FIXED_PARAMS_SCHEMA = Dict{String,Any}(
    "type" => "object",
    "description" => "固定参数 (参数名 -> 值)，不参与优化"
)

"""自定义参数范围"""
const PARAM_BOUNDS_SCHEMA = Dict{String,Any}(
    "type" => "object",
    "description" => "自定义参数范围 (参数名 -> [min, max])"
)

"""初始状态字段"""
const INIT_STATES_SCHEMA = Dict{String,Any}(
    "type" => "object",
    "description" => "自定义初始状态 (状态名 -> 值)"
)

# ==============================================================================
# 数据源相关 Schema
# ==============================================================================

"""驱动数据配置"""
const FORCING_SCHEMA = Dict{String,Any}(
    "type" => "object",
    "description" => "驱动数据配置（支持csv/json/redis）",
    "properties" => Dict{String,Any}(
        "source_type" => Dict{String,Any}(
            "type" => "string",
            "enum" => ["csv", "json", "redis"],
            "description" => "数据源类型"
        ),
        "path" => Dict{String,Any}(
            "type" => "string",
            "description" => "文件路径 (仅适用于 csv/json)"
        ),
        "key" => Dict{String,Any}(
            "type" => "string",
            "description" => "Redis 键名 (仅适用于 redis)"
        ),
        "data" => Dict{String,Any}(
            "type" => "object",
            "description" => "直接传入的数据对象 (仅适用于 json)"
        ),
        "host" => Dict{String,Any}(
            "type" => "string",
            "default" => "127.0.0.1",
            "description" => "Redis 服务器地址 (默认 127.0.0.1)"
        ),
        "port" => Dict{String,Any}(
            "type" => "integer",
            "default" => 6379,
            "description" => "Redis 服务器端口 (默认 6379)"
        )
    ),
    "required" => ["source_type"]
)

"""观测数据配置"""
const OBSERVATION_SCHEMA = Dict{String,Any}(
    "type" => "object",
    "description" => "观测数据配置"
)

"""观测值列名"""
const OBS_COLUMN_SCHEMA = Dict{String,Any}(
    "type" => "string",
    "description" => "观测值列名"
)

"""模拟值列名"""
const SIM_COLUMN_SCHEMA = Dict{String,Any}(
    "type" => "string",
    "description" => "模拟数据中的列名/键名"
)

"""数据源配置（通用）"""
const DATA_SOURCE_SCHEMA = Dict{String,Any}(
    "type" => "object",
    "description" => "数据源配置 (source_type + path/data)"
)

# ==============================================================================
# 时间范围相关 Schema
# ==============================================================================

"""率定期时间范围"""
const CALIBRATION_PERIOD_SCHEMA = Dict{String,Any}(
    "type" => "object",
    "description" => "率定期时间范围（start/end或start_index/end_index）"
)

"""验证期时间范围"""
const VALIDATION_PERIOD_SCHEMA = Dict{String,Any}(
    "type" => "object",
    "description" => "验证期时间范围（start/end或start_index/end_index）"
)

# ==============================================================================
# 输入映射 Schema
# ==============================================================================

"""输入变量映射"""
const INPUT_MAPPING_SCHEMA = Dict{String,Any}(
    "type" => "object",
    "description" => "输入变量映射（可选）"
)

export MODEL_NAME_SCHEMA, PARAMETERS_SCHEMA, PARAMS_ARRAY_SCHEMA,
       PARAMETER_SETS_SCHEMA, FIXED_PARAMS_SCHEMA, PARAM_BOUNDS_SCHEMA,
       INIT_STATES_SCHEMA, FORCING_SCHEMA, OBSERVATION_SCHEMA,
       OBS_COLUMN_SCHEMA, SIM_COLUMN_SCHEMA, DATA_SOURCE_SCHEMA,
       CALIBRATION_PERIOD_SCHEMA, VALIDATION_PERIOD_SCHEMA,
       INPUT_MAPPING_SCHEMA
