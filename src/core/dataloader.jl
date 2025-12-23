module DataLoader

using CSV
using DataFrames
using JSON3
using Redis
using Dates

# ==============================================================================
# 1. 对外统一接口
# ==============================================================================

"""
    process_io(f::Function, input_config::AbstractDict, args...)

I/O 处理的高阶函数。
它负责：
1. 解析输入配置 (`input_config`) 加载数据。
2. 执行核心计算函数 `f(data, args...)`。
3. 根据输入源类型，将计算结果保存为相同的格式。
"""
function process_io(f::Function, input_config::AbstractDict, args...)
    # 1. 识别源类型
    source_type_str = get(input_config, "source_type", "json")
    val_type = Val(Symbol(lowercase(source_type_str)))

    # 2. 加载数据 (Dispatch)
    # data 是 NamedTuple, metadata 是用于后续保存上下文的信息（如路径、Redis Key）
    data, metadata = load_data(val_type, input_config)

    # 3. 执行核心计算 (执行传入的 Simulation 逻辑)
    # result 通常是 Vector{Float64} 或 DataFrame
    result = f(data, args...)

    # 4. 保存结果 (Dispatch)
    # 保持输入输出形式一致
    return save_data(val_type, result, metadata)
end

# ==============================================================================
# 2. 数据加载 (Load Dispatch)
# ==============================================================================

# --- CSV ---
function load_data(::Val{:csv}, config::AbstractDict)
    path = config["path"]
    df = CSV.read(path, DataFrame)
    
    # 自动识别数值列并转换为 NamedTuple
    # 过滤掉非数值列（如日期）
    num_cols = names(df, Number)
    data_pairs = [Symbol(c) => Float64.(df[!, c]) for c in num_cols]
    
    return (; data_pairs...), (; path=path)
end

# --- JSON ---
function load_data(::Val{:json}, config::AbstractDict)
    # 支持两种模式：直接内嵌 data，或者指向本地 json 文件
    if haskey(config, "data")
        raw_data = config["data"]
    elseif haskey(config, "path")
        raw_data = JSON3.read(read(config["path"], String))
    else
        throw(ArgumentError("JSON 源必须包含 'data' 或 'path'"))
    end

    # 转换逻辑
    data_pairs = Pair{Symbol, Vector{Float64}}[]
    for (k, v) in raw_data
        push!(data_pairs, Symbol(k) => Float64.(collect(v)))
    end

    return (; data_pairs...), (; is_file=haskey(config, "path"))
end

# --- Redis ---
function load_data(::Val{:redis}, config::AbstractDict)
    host = get(config, "host", "127.0.0.1")
    port = get(config, "port", 6379)
    key  = config["key"]
    
    conn = Redis.RedisConnection(host=host, port=port)
    try
        val_str = Redis.get(conn, key)
        json_obj = JSON3.read(val_str)
        data_pairs = [Symbol(k) => Float64.(collect(v)) for (k, v) in json_obj]
        
        # 将连接配置传递给 metadata，以便保存时复用
        return (; data_pairs...), (; conn_conf=(host=host, port=port), input_key=key)
    finally
        if isopen(conn) close(conn) end
    end
end

# Fallback
load_data(::Val{T}, config) where T = throw(ArgumentError("不支持的数据源类型: $T"))

# ==============================================================================
# 3. 数据保存 (Save Dispatch)
# ==============================================================================

# --- CSV 输出 (输入是 CSV -> 输出也是 CSV) ---
function save_data(::Val{:csv}, result::AbstractVector, metadata)
    input_path = metadata.path
    # 生成输出路径： data.csv -> data_result_Timestamp.csv
    dir, fname = splitdir(input_path)
    base, ext  = splitext(fname)
    out_path = joinpath(dir, "$(base)_result_$(Dates.format(now(), "MMddHHmmss"))$ext")
    
    # 将结果写入 CSV
    # 假设结果是单列流量，如果有列名可以改进
    CSV.write(out_path, DataFrame(Result=result))
    
    return Dict("source_type" => "csv", "path" => out_path, "message" => "结果已保存至本地 CSV")
end

# --- JSON 输出 (直接返回数据) ---
function save_data(::Val{:json}, result::AbstractVector, metadata)
    return Dict("source_type" => "json", "result" => result)
end

# --- Redis 输出 (写入新 Key) ---
function save_data(::Val{:redis}, result::AbstractVector, metadata)
    conf = metadata.conn_conf
    input_key = metadata.input_key
    output_key = "$(input_key)_result"
    
    conn = Redis.RedisConnection(host=conf.host, port=conf.port)
    try
        # 序列化为 JSON 字符串存储
        Redis.set(conn, output_key, JSON3.write(Dict("result" => result)))
        return Dict("source_type" => "redis", "key" => output_key, "message" => "结果已写入 Redis")
    finally
        if isopen(conn) close(conn) end
    end
end

end