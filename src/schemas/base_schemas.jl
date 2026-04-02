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

"""计算预算字段（策略8算法选择）"""
const CALIBRATION_BUDGET_SCHEMA = Dict{String,Any}(
    "type" => "string",
    "enum" => ["low", "medium", "high"],
    "description" => "计算预算。用于策略8算法推荐：low/middle/high"
)

"""随机种子字段"""
const SEED_SCHEMA = Dict{String,Any}(
    "type" => "integer",
    "description" => "随机种子（可选，用于提高复现实验稳定性）"
)

"""对数变换模式"""
const LOG_TRANSFORM_MODE_SCHEMA = Dict{String,Any}(
    "type" => "string",
    "enum" => ["auto", "manual"],
    "description" => "对数变换模式：auto 按量级自动判定，manual 按 log_transform 显式控制"
)

"""参数约束（Strategy 2）"""
const PARAMETER_CONSTRAINTS_SCHEMA = Dict{String,Any}(
    "type" => "object",
    "description" => "参数约束配置。支持 pie_share（参数和约束）与 delta_method（不等式约束）",
    "properties" => Dict{String,Any}(
        "pie_share" => Dict{String,Any}(
            "type" => "object",
            "description" => "Pie-share 约束：指定参数和固定总和（例如参数占比和为 1.0）",
            "properties" => Dict{String,Any}(
                "parameters" => Dict{String,Any}(
                    "type" => "array",
                    "items" => Dict{String,Any}("type" => "string"),
                    "description" => "参与和约束的参数名列表"
                ),
                "total" => Dict{String,Any}(
                    "type" => "number",
                    "description" => "参数总和目标值，默认 1.0",
                    "default" => 1.0
                )
            ),
            "required" => ["parameters"]
        ),
        "delta_method" => Dict{String,Any}(
            "type" => "object",
            "description" => "Delta-method 不等式约束：lower 参数必须小于 upper 参数",
            "properties" => Dict{String,Any}(
                "inequalities" => Dict{String,Any}(
                    "type" => "array",
                    "description" => "不等式约束列表，每个元素为 [lower_param, upper_param]",
                    "items" => Dict{String,Any}(
                        "type" => "array",
                        "items" => Dict{String,Any}("type" => "string"),
                        "minItems" => 2,
                        "maxItems" => 2
                    )
                )
            ),
            "required" => ["inequalities"]
        )
    )
)

"""初始状态字段"""
const INIT_STATES_SCHEMA = Dict{String,Any}(
    "type" => "object",
    "description" => "自定义初始状态 (状态名 -> 值)"
)

# ==============================================================================
# 数据源相关 Schema
# ==============================================================================

const BASE_DATASET_GAUGE_ID_SCHEMA = Dict{String,Any}(
    "oneOf" => [
        Dict{String,Any}("type" => "integer"),
        Dict{String,Any}("type" => "string"),
    ],
    "description" => "数据集站点 ID；支持像 01013500 这样的字符串形式，避免丢失前导 0"
)

const BASE_CARAVAN_GAUGE_ID_SCHEMA = Dict{String,Any}(
    "oneOf" => [
        Dict{String,Any}("type" => "integer"),
        Dict{String,Any}("type" => "string"),
    ],
    "description" => "Caravan 流域站点 ID；优先使用全局 ID（如 camels_01013500），也支持 dataset_name/source_dataset + 本地站点 ID"
)

const BASE_CARAVAN_DATASET_NAME_SCHEMA = Dict{String,Any}(
    "type" => "string",
    "enum" => ["camels", "camelsaus", "camelsbr", "camelscl", "camelsgb", "hysets", "lamah"],
    "description" => "Caravan 子数据集名称"
)

"""驱动数据配置"""
const FORCING_SCHEMA = Dict{String,Any}(
    "type" => "object",
    "description" => "驱动数据配置（支持csv/json/redis/caravan）",
    "properties" => Dict{String,Any}(
        "source_type" => Dict{String,Any}(
            "type" => "string",
            "enum" => ["csv", "json", "redis", "caravan"],
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
        ),
        "dataset_path" => Dict{String,Any}(
            "type" => "string",
            "description" => "Caravan 数据集根目录别名（可选）"
        ),
        "dataset_root" => Dict{String,Any}(
            "type" => "string",
            "description" => "Caravan 数据集根目录（包含 attributes/ 和 timeseries/）"
        ),
        "timeseries_root" => Dict{String,Any}(
            "type" => "string",
            "description" => "Caravan timeseries 根目录"
        ),
        "netcdf_root" => Dict{String,Any}(
            "type" => "string",
            "description" => "Caravan netCDF 根目录（未提供时优先使用环境变量）"
        ),
        "dataset_name" => BASE_CARAVAN_DATASET_NAME_SCHEMA,
        "source_dataset" => BASE_CARAVAN_DATASET_NAME_SCHEMA,
        "gage_id" => BASE_DATASET_GAUGE_ID_SCHEMA,
        "gauge_id" => merge(copy(BASE_DATASET_GAUGE_ID_SCHEMA), Dict{String,Any}(
            "description" => "站点 ID 别名（等价于 gage_id）；支持像 01013500 这样的字符串形式"
        )),
        "basin_id" => BASE_CARAVAN_GAUGE_ID_SCHEMA,
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

# ==============================================================================
# 存储结果相关 Schema
# ==============================================================================

"""存储分类"""
const STORAGE_CATEGORY_SCHEMA = Dict{String,Any}(
    "type" => "string",
    "enum" => ["calibration", "sensitivity", "ensemble"],
    "description" => "存储结果分类"
)

"""结果ID"""
const RESULT_ID_SCHEMA = Dict{String,Any}(
    "type" => "string",
    "description" => "存储结果的唯一标识"
)

export MODEL_NAME_SCHEMA, PARAMETERS_SCHEMA, PARAMS_ARRAY_SCHEMA,
       PARAMETER_SETS_SCHEMA, FIXED_PARAMS_SCHEMA, PARAM_BOUNDS_SCHEMA,
       CALIBRATION_BUDGET_SCHEMA, SEED_SCHEMA, LOG_TRANSFORM_MODE_SCHEMA,
       PARAMETER_CONSTRAINTS_SCHEMA,
       INIT_STATES_SCHEMA, FORCING_SCHEMA, OBSERVATION_SCHEMA,
       OBS_COLUMN_SCHEMA, SIM_COLUMN_SCHEMA, DATA_SOURCE_SCHEMA,
       CALIBRATION_PERIOD_SCHEMA, VALIDATION_PERIOD_SCHEMA,
       INPUT_MAPPING_SCHEMA, STORAGE_CATEGORY_SCHEMA, RESULT_ID_SCHEMA
