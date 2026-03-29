module DataLoader

using CSV
using DataFrames
using Dates
using JSON3
using Redis

function process_io(f::Function, input_config::AbstractDict, args...; save_context = nothing)
    source_type_str = get(input_config, "source_type", "json")
    val_type = Val(Symbol(lowercase(source_type_str)))

    data, metadata = load_data(val_type, input_config)
    result = f(data, args...)
    merged_metadata = _merge_metadata(metadata, save_context)
    return save_data(val_type, result, merged_metadata)
end

function load_data(::Val{:csv}, config::AbstractDict)
    path = config["path"]
    df = CSV.read(path, DataFrame)

    num_cols = names(df, Number)
    data_pairs = [Symbol(c) => Float64.(df[!, c]) for c in num_cols]

    return (; data_pairs...), (; path = path, column_names = String.(names(df)))
end

function load_data(::Val{:json}, config::AbstractDict)
    raw_data = if haskey(config, "data")
        config["data"]
    elseif haskey(config, "path")
        JSON3.read(read(config["path"], String))
    else
        throw(ArgumentError("JSON source must include 'data' or 'path'"))
    end

    data_pairs = Pair{Symbol,Vector{Float64}}[]
    for (k, v) in raw_data
        push!(data_pairs, Symbol(k) => Float64.(collect(v)))
    end

    return (; data_pairs...), (; path = get(config, "path", nothing), is_file = haskey(config, "path"))
end

function load_data(::Val{:redis}, config::AbstractDict)
    host = get(config, "host", "127.0.0.1")
    port = get(config, "port", 6379)
    key = config["key"]

    conn = Redis.RedisConnection(host = host, port = port)
    try
        val_str = Redis.get(conn, key)
        json_obj = JSON3.read(val_str)
        data_pairs = [Symbol(k) => Float64.(collect(v)) for (k, v) in json_obj]
        return (; data_pairs...), (; conn_conf = (host = host, port = port), input_key = key)
    finally
        if isopen(conn)
            close(conn)
        end
    end
end

load_data(::Val{T}, _config) where {T} = throw(ArgumentError("Unsupported source_type: $T"))

function save_data(::Val{:csv}, result::AbstractVector, metadata)
    return _write_result_artifacts(result, metadata; source_type = "csv")
end

function save_data(::Val{:json}, result::AbstractVector, metadata)
    output_dir = _metadata_get(metadata, :output_dir, nothing)
    if !isnothing(output_dir)
        return _write_result_artifacts(result, metadata; source_type = "json")
    end

    payload = Dict(
        "source_type" => "json",
        "result" => collect(result),
    )
    return _merge_payload(payload, metadata)
end

function save_data(::Val{:redis}, result::AbstractVector, metadata)
    output_dir = _metadata_get(metadata, :output_dir, nothing)
    if !isnothing(output_dir)
        return _write_result_artifacts(result, metadata; source_type = "redis")
    end

    conf = _metadata_get(metadata, :conn_conf)
    input_key = _metadata_get(metadata, :input_key)
    output_key = "$(input_key)_result"

    conn = Redis.RedisConnection(host = conf.host, port = conf.port)
    try
        Redis.set(conn, output_key, JSON3.write(Dict("result" => collect(result))))
        return _merge_payload(
            Dict(
                "source_type" => "redis",
                "key" => output_key,
                "message" => "Result written to Redis",
            ),
            metadata,
        )
    finally
        if isopen(conn)
            close(conn)
        end
    end
end

function _merge_metadata(metadata, save_context)
    if isnothing(save_context)
        return metadata
    elseif metadata isa NamedTuple && save_context isa NamedTuple
        return merge(metadata, save_context)
    elseif metadata isa NamedTuple && save_context isa AbstractDict
        return merge(metadata, (; (Symbol(k) => v for (k, v) in pairs(save_context))...))
    elseif metadata isa AbstractDict && save_context isa NamedTuple
        merged = Dict{String,Any}(string(k) => v for (k, v) in pairs(metadata))
        merge!(merged, Dict{String,Any}(string(k) => v for (k, v) in pairs(save_context)))
        return merged
    elseif metadata isa AbstractDict && save_context isa AbstractDict
        merged = Dict{String,Any}(string(k) => v for (k, v) in pairs(metadata))
        merge!(merged, Dict{String,Any}(string(k) => v for (k, v) in pairs(save_context)))
        return merged
    end

    return metadata
end

function _metadata_get(metadata, key::Symbol, default = nothing)
    if metadata isa NamedTuple
        return get(metadata, key, default)
    elseif metadata isa AbstractDict
        return get(metadata, string(key), get(metadata, key, default))
    end
    return default
end

function _write_result_artifacts(result::AbstractVector, metadata; source_type::String)
    input_path = _metadata_get(metadata, :path, nothing)
    output_dir = _metadata_get(metadata, :output_dir, nothing)
    timestamp = _metadata_get(metadata, :timestamp, Dates.format(now(), "yyyymmddHHMMSS"))
    base_name = _metadata_get(metadata, :base_name, nothing)
    base_name = isnothing(base_name) ?
        (isnothing(input_path) ? "simulation" : splitext(basename(input_path))[1]) :
        String(base_name)

    result_prefix = "$(base_name)_result_$(timestamp)"
    simulation_dir = if isnothing(output_dir)
        isnothing(input_path) ? pwd() : dirname(input_path)
    else
        joinpath(String(output_dir), "simulation")
    end
    mkpath(simulation_dir)

    output_path = joinpath(simulation_dir, "$(result_prefix).csv")
    CSV.write(output_path, DataFrame(Result = collect(result)))

    metadata_payload = Dict{String,Any}(
        "status" => _metadata_get(metadata, :status, "success"),
        "source_type" => source_type,
        "run_id" => _metadata_get(metadata, :run_id, nothing),
        "run_info" => _metadata_get(metadata, :run_info, Dict{String,Any}()),
        "warnings" => collect(String.(get(_metadata_get(metadata, :artifact_payload, Dict{String,Any}()), "warnings", Any[]))),
    )

    artifact_payload = _metadata_get(metadata, :artifact_payload, nothing)
    if artifact_payload isa AbstractDict
        merge!(metadata_payload, Dict{String,Any}(string(k) => v for (k, v) in pairs(artifact_payload)))
    end

    metadata_path = joinpath(simulation_dir, "$(result_prefix).metadata.json")
    open(metadata_path, "w") do io
        JSON3.write(io, metadata_payload)
    end

    summary_path = if isnothing(output_dir)
        joinpath(simulation_dir, "$(result_prefix).summary.md")
    else
        mkpath(String(output_dir))
        joinpath(String(output_dir), "run_summary_$(timestamp).md")
    end

    _write_summary_markdown(summary_path, metadata_payload, output_path)

    response = Dict(
        "status" => get(metadata_payload, "status", "success"),
        "source_type" => source_type,
        "path" => output_path,
        "output_path" => output_path,
        "metadata_path" => metadata_path,
        "summary_path" => summary_path,
    )

    return _merge_payload(response, metadata)
end

function _write_summary_markdown(summary_path::AbstractString, payload::Dict{String,Any}, output_path::AbstractString)
    warnings = get(payload, "warnings", Any[])
    run_info = get(payload, "run_info", Dict{String,Any}())
    params_source = get(payload, "params_source", "provided")
    params_seed = get(payload, "params_seed", nothing)

    lines = String[
        "# Run Summary",
        "",
        "- status: $(get(payload, "status", "success"))",
        "- model: $(get(run_info, "model", "unknown"))",
        "- output_path: $output_path",
        "- params_source: $params_source",
    ]

    if !isnothing(params_seed)
        push!(lines, "- params_seed: $(params_seed)")
    end

    if !isempty(warnings)
        push!(lines, "- warnings: $(join(String.(warnings), "; "))")
    end

    open(summary_path, "w") do io
        write(io, join(lines, "\n") * "\n")
    end
end

function _merge_payload(payload::AbstractDict, metadata)
    merged = Dict{String,Any}(string(k) => v for (k, v) in pairs(payload))
    for key in (:run_id, :run_info, :warnings, :params_used, :params_source, :params_seed)
        value = _metadata_get(metadata, key, nothing)
        !isnothing(value) && (merged[string(key)] = value)
    end
    return merged
end

end
