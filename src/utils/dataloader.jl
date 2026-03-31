module DataLoader

using CSV
using DataFrames
using Dates
using JSON3
using NPZ
using Redis
using UUIDs

function process_io(f::Function, input_config::AbstractDict, args...; save_context = nothing)
    source_type_str = get(input_config, "source_type", "json")
    val_type = Val(Symbol(lowercase(source_type_str)))

    data, metadata = load_data(val_type, input_config)
    result = f(data, args...)
    merged_metadata = _merge_metadata(metadata, save_context)
    return save_data(val_type, result, merged_metadata)
end

const CAMELS_ENV_KEYS = ("CAMESL_DATASET_PATH", "CAMELS_DATASET_PATH")
const DATE_COLUMN_HINTS = ("date", "datetime", "timestamp", "time")
const DATE_PARSE_FORMATS = (
    dateformat"yyyy-mm-dd",
    dateformat"yyyy/mm/dd",
    dateformat"yyyymmdd",
    dateformat"yyyy-mm-dd HH:MM:SS",
    dateformat"yyyy/mm/dd HH:MM:SS",
    dateformat"yyyy-mm-ddTHH:MM:SS",
    dateformat"yyyy-mm-ddTHH:MM:SS.s",
    dateformat"yyyy-mm-ddTHH:MM:SS.sss",
)

function _try_parse_date(value)
    if value isa Date
        return value
    elseif value isa DateTime
        return Date(value)
    elseif value isa Missing || value === nothing
        return nothing
    end

    text = strip(string(value))
    isempty(text) && return nothing

    for fmt in DATE_PARSE_FORMATS
        try
            return Date(text, fmt)
        catch
        end

        try
            return Date(DateTime(text, fmt))
        catch
        end
    end

    try
        return Date(text)
    catch
    end

    try
        return Date(DateTime(text))
    catch
    end

    return nothing
end

function _extract_dates_from_dataframe(df::DataFrame)
    column_names = String.(names(df))
    lookup = Dict(lowercase(name) => name for name in column_names)

    candidate_columns = String[]
    for alias in DATE_COLUMN_HINTS
        resolved = get(lookup, alias, nothing)
        isnothing(resolved) || push!(candidate_columns, resolved)
    end

    for name in names(df)
        column = df[!, name]
        if eltype(column) <: Union{Date,DateTime,Missing}
            push!(candidate_columns, string(name))
        end
    end
    unique!(candidate_columns)

    for column_name in candidate_columns
        raw_values = df[!, Symbol(column_name)]
        parsed_values = Date[]
        parse_ok = true

        for raw in raw_values
            parsed = _try_parse_date(raw)
            if isnothing(parsed)
                parse_ok = false
                break
            end
            push!(parsed_values, parsed)
        end

        parse_ok && !isempty(parsed_values) && return parsed_values, column_name
    end

    return nothing, nothing
end

function _parse_int_like(value, field_name::String)
    if value isa Integer
        return Int(value)
    elseif value isa AbstractFloat
        isfinite(value) || throw(ArgumentError("$field_name must be finite"))
        rounded = round(Int, value)
        rounded == value || throw(ArgumentError("$field_name must be an integer-compatible value"))
        return rounded
    end

    text = strip(string(value))
    isempty(text) && throw(ArgumentError("$field_name cannot be empty"))
    try
        return parse(Int, text)
    catch
        throw(ArgumentError("$field_name must be an integer-compatible value"))
    end
end

function _resolve_camels_dataset_path(config::AbstractDict)
    if haskey(config, "dataset_path")
        raw = strip(string(config["dataset_path"]))
        isempty(raw) && throw(ArgumentError("dataset_path cannot be empty"))
        return raw
    end

    for env_key in CAMELS_ENV_KEYS
        env_value = get(ENV, env_key, nothing)
        if !(env_value === nothing)
            candidate = strip(String(env_value))
            isempty(candidate) || return candidate
        end
    end

    throw(ArgumentError(
        "CAMELS source requires dataset_path, or env var CAMESL_DATASET_PATH/CAMELS_DATASET_PATH",
    ))
end

function _resolve_camels_gage_id(config::AbstractDict)
    if haskey(config, "gage_id")
        return _parse_int_like(config["gage_id"], "gage_id")
    elseif haskey(config, "gauge_id")
        return _parse_int_like(config["gauge_id"], "gauge_id")
    end

    throw(ArgumentError("CAMELS source must include 'gage_id' (or alias 'gauge_id')"))
end

function _camels_find_gage_index(gage_ids, gage_id::Int)
    for (idx, raw_value) in enumerate(gage_ids)
        parsed = try
            _parse_int_like(raw_value, "gage_ids[$idx]")
        catch
            continue
        end

        parsed == gage_id && return idx
    end
    return nothing
end

function _camels_extract_forcings(raw_forcings, catchment_idx::Int)
    ndims(raw_forcings) == 3 || throw(ArgumentError("CAMELS forcings must be a 3D array"))

    if size(raw_forcings, 3) == 3
        p = Float64.(vec(raw_forcings[catchment_idx, :, 1]))
        t = Float64.(vec(raw_forcings[catchment_idx, :, 2]))
        ep = Float64.(vec(raw_forcings[catchment_idx, :, 3]))
        return p, t, ep
    elseif size(raw_forcings, 2) == 3
        p = Float64.(vec(raw_forcings[catchment_idx, 1, :]))
        t = Float64.(vec(raw_forcings[catchment_idx, 2, :]))
        ep = Float64.(vec(raw_forcings[catchment_idx, 3, :]))
        return p, t, ep
    elseif size(raw_forcings, 3) >= 3 && size(raw_forcings, 2) < 3
        p = Float64.(vec(raw_forcings[catchment_idx, :, 1]))
        t = Float64.(vec(raw_forcings[catchment_idx, :, 2]))
        ep = Float64.(vec(raw_forcings[catchment_idx, :, 3]))
        return p, t, ep
    elseif size(raw_forcings, 2) >= 3 && size(raw_forcings, 3) < 3
        p = Float64.(vec(raw_forcings[catchment_idx, 1, :]))
        t = Float64.(vec(raw_forcings[catchment_idx, 2, :]))
        ep = Float64.(vec(raw_forcings[catchment_idx, 3, :]))
        return p, t, ep
    end

    throw(ArgumentError("CAMELS forcings must provide at least 3 variables (P/T/Ep)"))
end

function _camels_extract_target(raw_target, catchment_idx::Int)
    if ndims(raw_target) == 3
        size(raw_target, 3) >= 1 || throw(ArgumentError("CAMELS target array has no variable dimension"))
        return Float64.(vec(raw_target[catchment_idx, :, 1]))
    elseif ndims(raw_target) == 2
        return Float64.(vec(raw_target[catchment_idx, :]))
    end

    throw(ArgumentError("CAMELS target must be a 2D or 3D array"))
end

function _camels_resolve_area_km2(raw_attributes, catchment_idx::Int)
    ndims(raw_attributes) == 2 || throw(ArgumentError("CAMELS attributes must be a 2D array"))

    n_cols = size(raw_attributes, 2)
    preferred_cols = Int[]
    n_cols >= 12 && push!(preferred_cols, 12)
    n_cols >= 13 && push!(preferred_cols, 13)

    for col in preferred_cols
        area_val = try
            Float64(raw_attributes[catchment_idx, col])
        catch
            NaN
        end

        if isfinite(area_val) && area_val > 0
            return area_val, col
        end
    end

    for col in 1:n_cols
        area_val = try
            Float64(raw_attributes[catchment_idx, col])
        catch
            NaN
        end

        if isfinite(area_val) && area_val > 0
            return area_val, col
        end
    end

    throw(ArgumentError("Could not resolve a valid positive catchment area from CAMELS attributes"))
end

function load_data(::Val{:csv}, config::AbstractDict)
    path = config["path"]
    df = CSV.read(path, DataFrame)

    num_cols = names(df, Number)
    data_pairs = [Symbol(c) => Float64.(df[!, c]) for c in num_cols]

    dates, date_column = _extract_dates_from_dataframe(df)

    return (; data_pairs...), (
        ;
        path = path,
        column_names = String.(names(df)),
        dates = dates,
        date_column = date_column,
    )
end

function load_data(::Val{:camels}, config::AbstractDict)
    gage_id = _resolve_camels_gage_id(config)
    raw_dataset_path = _resolve_camels_dataset_path(config)
    dataset_path = normpath(abspath(raw_dataset_path))
    isfile(dataset_path) || throw(ArgumentError("CAMELS dataset not found: $dataset_path"))

    data = npzread(dataset_path)
    haskey(data, "gage_ids") || throw(ArgumentError("CAMELS dataset is missing key 'gage_ids'"))
    haskey(data, "forcings") || throw(ArgumentError("CAMELS dataset is missing key 'forcings'"))
    haskey(data, "target") || throw(ArgumentError("CAMELS dataset is missing key 'target'"))
    haskey(data, "attributes") || throw(ArgumentError("CAMELS dataset is missing key 'attributes'"))

    gage_ids = data["gage_ids"]
    catchment_idx = _camels_find_gage_index(gage_ids, gage_id)
    isnothing(catchment_idx) && throw(ArgumentError("Gage $gage_id not found in CAMELS dataset"))

    p, t, ep = _camels_extract_forcings(data["forcings"], catchment_idx)
    target_cfs = _camels_extract_target(data["target"], catchment_idx)
    area_km2, area_column = _camels_resolve_area_km2(data["attributes"], catchment_idx)

    flow_mm = target_cfs .* (1e3) .* 0.0283168 .* 3600 .* 24 ./ (area_km2 .* 1e6)
    tidx = Float64.(rem.(0:length(p)-1, 365) .+ 1)

    n_samples = minimum((length(p), length(t), length(ep), length(flow_mm), length(tidx)))
    n_samples > 0 || throw(ArgumentError("CAMELS series is empty for gage $gage_id"))

    start_date = Date(1980, 10, 1)
    dates = [start_date + Day(i - 1) for i in 1:n_samples]

    p = p[1:n_samples]
    t = t[1:n_samples]
    ep = ep[1:n_samples]
    flow_mm = flow_mm[1:n_samples]
    tidx = tidx[1:n_samples]

    valid_mask = .!(isnan.(p) .| isnan.(t) .| isnan.(ep) .| isnan.(flow_mm) .| isnan.(tidx))
    n_invalid = count(.!valid_mask)

    p = p[valid_mask]
    t = t[valid_mask]
    ep = ep[valid_mask]
    flow_mm = flow_mm[valid_mask]
    tidx = tidx[valid_mask]
    dates = dates[valid_mask]

    isempty(p) && throw(ArgumentError("CAMELS series contains no valid rows after NaN filtering for gage $gage_id"))

    data_pairs = Pair{Symbol,Vector{Float64}}[
        :P => p,
        :T => t,
        :Ep => ep,
        :Tidx => tidx,
        Symbol("flow(mm)") => flow_mm,
        :obs => flow_mm,
    ]

    metadata = (
        path = dataset_path,
        source_type = "camels",
        gage_id = gage_id,
        catchment_index = catchment_idx,
        area_km2 = area_km2,
        area_column = area_column,
        dropped_nan_rows = n_invalid,
        column_names = ["P", "T", "Ep", "Tidx", "flow(mm)", "obs"],
        dates = dates,
    )

    return (; data_pairs...), metadata
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
        isnothing(val_str) && throw(ArgumentError("Redis key not found: $key"))
        json_obj = JSON3.read(val_str)
        data_pairs = [Symbol(k) => Float64.(collect(v)) for (k, v) in json_obj]
        return (; data_pairs...), (; conn_conf = (host = host, port = port), input_key = key)
    finally
        try
            close(conn)
        catch
            # RedisConnection does not implement Base.isopen; ignore close-time state errors.
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

    conf = _metadata_get(metadata, :conn_conf, (host = "127.0.0.1", port = 6379))
    host = if conf isa NamedTuple
        string(get(conf, :host, "127.0.0.1"))
    elseif conf isa AbstractDict
        string(get(conf, "host", get(conf, :host, "127.0.0.1")))
    else
        "127.0.0.1"
    end
    port = if conf isa NamedTuple
        Int(get(conf, :port, 6379))
    elseif conf isa AbstractDict
        Int(get(conf, "port", get(conf, :port, 6379)))
    else
        6379
    end

    input_key = _metadata_get(metadata, :input_key, nothing)
    output_key = _metadata_get(metadata, :output_key, nothing)
    if isnothing(output_key)
        output_key = isnothing(input_key) ?
                     "simulation_result_" * string(_metadata_get(metadata, :run_id, UUIDs.uuid4())) :
                     "$(input_key)_result"
    end

    conn = Redis.RedisConnection(host = host, port = port)
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
        try
            close(conn)
        catch
            # RedisConnection does not implement Base.isopen; ignore close-time state errors.
        end
    end
end

_merge_metadata(metadata, ::Nothing) = metadata

function _merge_metadata(metadata::NamedTuple, save_context::NamedTuple)
    return merge(metadata, save_context)
end

function _merge_metadata(metadata::NamedTuple, save_context::AbstractDict)
    return merge(metadata, (; (Symbol(k) => v for (k, v) in pairs(save_context))...))
end

function _merge_metadata(metadata::AbstractDict, save_context::NamedTuple)
    merged = Dict{String,Any}(string(k) => v for (k, v) in pairs(metadata))
    merge!(merged, Dict{String,Any}(string(k) => v for (k, v) in pairs(save_context)))
    return merged
end

function _merge_metadata(metadata::AbstractDict, save_context::AbstractDict)
    merged = Dict{String,Any}(string(k) => v for (k, v) in pairs(metadata))
    merge!(merged, Dict{String,Any}(string(k) => v for (k, v) in pairs(save_context)))
    return merged
end

_merge_metadata(metadata, _save_context) = metadata

_metadata_get(metadata::NamedTuple, key::Symbol, default = nothing) = get(metadata, key, default)

function _metadata_get(metadata::AbstractDict, key::Symbol, default = nothing)
    return get(metadata, string(key), get(metadata, key, default))
end

_metadata_get(_metadata, _key::Symbol, default = nothing) = default

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
