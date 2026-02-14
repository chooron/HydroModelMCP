"""
评价指标相关 Schema 定义
"""

# ==============================================================================
# 评价指标 Schema
# ==============================================================================

"""评价指标列表（完整版本）"""
const METRICS_SCHEMA = Dict{String,Any}(
    "type" => "array",
    "description" => "评价指标列表",
    "items" => Dict{String,Any}(
        "type" => "string",
        "enum" => ["NSE", "KGE", "LogNSE", "LogKGE", "RMSE", "R2", "PBIAS"]
    ),
    "default" => ["NSE", "KGE", "RMSE"]
)

"""单个目标函数（用于单目标校准）"""
const OBJECTIVE_SCHEMA = Dict{String,Any}(
    "type" => "string",
    "enum" => ["KGE", "NSE", "LogNSE", "LogKGE", "PBIAS", "R2", "RMSE"],
    "description" => "目标函数。默认 KGE"
)

"""多个目标函数（用于多目标校准）"""
const OBJECTIVES_SCHEMA = Dict{String,Any}(
    "type" => "array",
    "items" => Dict{String,Any}("type" => "string"),
    "description" => "目标函数列表，如 [\"KGE\", \"LogKGE\"]"
)

"""对数变换开关"""
const LOG_TRANSFORM_SCHEMA = Dict{String,Any}(
    "type" => "boolean",
    "description" => "是否对数据做对数变换。默认 false"
)

export METRICS_SCHEMA, OBJECTIVE_SCHEMA, OBJECTIVES_SCHEMA, LOG_TRANSFORM_SCHEMA
