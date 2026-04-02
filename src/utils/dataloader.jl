module DataLoader

using CSV
using DataFrames
using Dates
using JSON3
using NCDatasets
using Redis
using UUIDs

import ..get_config_env

function process_io(f::Function, input_config::AbstractDict, args...; save_context = nothing)
    source_type_str = get(input_config, "source_type", "json")
    val_type = Val(Symbol(lowercase(source_type_str)))

    data, metadata = load_data(val_type, input_config)
    result = f(data, args...)
    merged_metadata = _merge_metadata(metadata, save_context)
    return save_data(val_type, result, merged_metadata)
end

const CARAVAN_ROOT_ENV_KEYS = ("CARAVAN_DATASET_ROOT", "CARAVAN_DATASET_PATH")
const CARAVAN_TIMESERIES_ENV_KEYS = ("CARAVAN_TIMESERIES_ROOT", "CARAVAN_TIMESERIES_PATH")
const CARAVAN_NETCDF_ENV_KEYS = ("CARAVAN_NETCDF_ROOT", "CARAVAN_NETCDF_PATH")
const CARAVAN_DATASET_NAMES = Set([
    "camels",
    "camelsaus",
    "camelsbr",
    "camelscl",
    "camelsgb",
    "hysets",
    "lamah",
])
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

function _has_nonempty_string(config::AbstractDict, key::String)
    haskey(config, key) || return false
    return !isempty(strip(string(config[key])))
end

function _first_nonempty_env(keys)
    for env_key in keys
        env_value = get_config_env(env_key, nothing)
        env_value === nothing && continue
        candidate = strip(String(env_value))
        isempty(candidate) || return candidate
    end
    return nothing
end

function _normalize_caravan_dataset_name(value, field_name::String = "dataset_name")
    name = lowercase(strip(string(value)))
    isempty(name) && throw(ArgumentError("$field_name cannot be empty"))
    name in CARAVAN_DATASET_NAMES || throw(ArgumentError(
        "$field_name must be one of: $(join(sort!(collect(CARAVAN_DATASET_NAMES)), ", "))",
    ))
    return name
end

function _resolve_caravan_roots(config::AbstractDict)
    direct_file_path = nothing
    if _has_nonempty_string(config, "path")
        direct_file_path = normpath(abspath(strip(string(config["path"]))))
        isfile(direct_file_path) || throw(ArgumentError("Caravan file not found: $direct_file_path"))
    end

    dataset_root = nothing
    for key in ("dataset_root", "dataset_path")
        _has_nonempty_string(config, key) || continue
        candidate = normpath(abspath(strip(string(config[key]))))
        if isdir(candidate)
            dataset_root = candidate
            break
        elseif key == "dataset_path" && isfile(candidate)
            direct_file_path = candidate
        end
    end

    if isnothing(dataset_root)
        raw_root = _first_nonempty_env(CARAVAN_ROOT_ENV_KEYS)
        if !isnothing(raw_root)
            candidate = normpath(abspath(raw_root))
            isdir(candidate) || throw(ArgumentError("CARAVAN dataset root not found: $candidate"))
            dataset_root = candidate
        end
    end

    timeseries_root = nothing
    if _has_nonempty_string(config, "timeseries_root")
        candidate = normpath(abspath(strip(string(config["timeseries_root"]))))
        isdir(candidate) || throw(ArgumentError("Caravan timeseries_root not found: $candidate"))
        timeseries_root = candidate
    elseif isnothing(dataset_root)
        raw_timeseries = _first_nonempty_env(CARAVAN_TIMESERIES_ENV_KEYS)
        if !isnothing(raw_timeseries)
            candidate = normpath(abspath(raw_timeseries))
            isdir(candidate) || throw(ArgumentError("CARAVAN timeseries root not found: $candidate"))
            timeseries_root = candidate
        end
    end

    netcdf_root = nothing
    if _has_nonempty_string(config, "netcdf_root")
        candidate = normpath(abspath(strip(string(config["netcdf_root"]))))
        isdir(candidate) || throw(ArgumentError("Caravan netcdf_root not found: $candidate"))
        netcdf_root = candidate
    else
        raw_netcdf = _first_nonempty_env(CARAVAN_NETCDF_ENV_KEYS)
        if !isnothing(raw_netcdf)
            candidate = normpath(abspath(raw_netcdf))
            isdir(candidate) || throw(ArgumentError("CARAVAN netcdf root not found: $candidate"))
            netcdf_root = candidate
        end
    end

    if isnothing(netcdf_root)
        if !isnothing(timeseries_root)
            candidate = joinpath(timeseries_root, "netcdf")
            isdir(candidate) || throw(ArgumentError("Caravan netcdf directory not found: $candidate"))
            netcdf_root = candidate
        elseif !isnothing(dataset_root)
            candidate = joinpath(dataset_root, "timeseries", "netcdf")
            isdir(candidate) || throw(ArgumentError("Caravan netcdf directory not found: $candidate"))
            netcdf_root = candidate
        end
    end

    isnothing(netcdf_root) && isnothing(direct_file_path) && throw(ArgumentError(
        "Caravan source requires path, dataset_root/dataset_path/netcdf_root, or env var CARAVAN_DATASET_ROOT/CARAVAN_NETCDF_ROOT",
    ))

    if isnothing(dataset_root) && !isnothing(netcdf_root)
        candidate = normpath(joinpath(netcdf_root, "..", ".."))
        isdir(joinpath(candidate, "attributes")) && (dataset_root = candidate)
    end

    attributes_root = isnothing(dataset_root) ? nothing : joinpath(dataset_root, "attributes")
    return dataset_root, netcdf_root, attributes_root, direct_file_path
end

function _resolve_caravan_gauge_token(config::AbstractDict)
    for key in ("gauge_id", "gage_id", "basin_id")
        _has_nonempty_string(config, key) || continue
        return strip(string(config[key]))
    end
    throw(ArgumentError("Caravan source must include 'gauge_id', 'gage_id', or 'basin_id'"))
end

function _split_caravan_gauge_token(gauge_token::AbstractString)
    normalized_token = String(gauge_token)
    parts = split(normalized_token, '_'; limit = 2)
    if length(parts) == 2
        dataset_name = lowercase(strip(parts[1]))
        local_id = strip(parts[2])
        if dataset_name in CARAVAN_DATASET_NAMES && !isempty(local_id)
            return dataset_name, String(local_id)
        end
    end
    return nothing, normalized_token
end

function _caravan_local_id_candidates(local_id::String, dataset_name::Union{Nothing,String})
    candidates = String[local_id]
    if !isnothing(dataset_name) && dataset_name == "camels" && occursin(r"^\d+$", local_id)
        padded = lpad(local_id, 8, '0')
        padded in candidates || push!(candidates, padded)
    end
    return candidates
end

function _infer_caravan_identity_from_path(path::String)
    stem = splitext(basename(path))[1]
    dataset_name, local_id = _split_caravan_gauge_token(stem)
    isnothing(dataset_name) && throw(ArgumentError(
        "Could not infer Caravan dataset/source from file name '$stem'. Expected '<dataset>_<gauge_id>.nc'.",
    ))
    return dataset_name, local_id, stem
end

function _resolve_caravan_timeseries_file(config::AbstractDict, netcdf_root, direct_file_path)
    if !isnothing(direct_file_path)
        dataset_name, local_id, canonical_gauge_id = _infer_caravan_identity_from_path(direct_file_path)
        return direct_file_path, dataset_name, local_id, canonical_gauge_id
    end

    gauge_token = _resolve_caravan_gauge_token(config)
    dataset_name = if _has_nonempty_string(config, "dataset_name")
        _normalize_caravan_dataset_name(config["dataset_name"], "dataset_name")
    elseif _has_nonempty_string(config, "source_dataset")
        _normalize_caravan_dataset_name(config["source_dataset"], "source_dataset")
    else
        throw(ArgumentError("Caravan source requires dataset_name/source_dataset together with gauge_id/gage_id/basin_id"))
    end

    prefixed_dataset, local_id = _split_caravan_gauge_token(gauge_token)
    if !isnothing(prefixed_dataset)
        if !isnothing(dataset_name) && dataset_name != prefixed_dataset
            throw(ArgumentError("Caravan dataset_name '$dataset_name' conflicts with gauge_id prefix '$prefixed_dataset'"))
        end
    end

    source_dir = joinpath(netcdf_root, dataset_name)
    isdir(source_dir) || throw(ArgumentError("Caravan dataset '$dataset_name' was not found under netcdf root '$netcdf_root'"))

    for candidate_local_id in _caravan_local_id_candidates(local_id, dataset_name)
        canonical_gauge_id = "$(dataset_name)_$(candidate_local_id)"
        candidate_path = joinpath(source_dir, canonical_gauge_id * ".nc")
        isfile(candidate_path) && return candidate_path, dataset_name, candidate_local_id, canonical_gauge_id
    end

    throw(ArgumentError("Caravan gauge '$gauge_token' was not found under dataset '$dataset_name'"))
end

function _numeric_vector_from_caravan_var(values)
    if values isa AbstractVector{<:Union{Missing,Number}}
        return Float64.(coalesce.(values, NaN))
    elseif values isa AbstractVector{<:Number}
        return Float64.(values)
    end

    flattened = vec(values)
    return [value isa Missing ? NaN : Float64(value) for value in flattened]
end

function _parse_cf_time_units(units::AbstractString)
    matched = match(r"^(seconds|hours|days)\s+since\s+(\d{4}-\d{2}-\d{2})(?:[ T](\d{2}:\d{2}:\d{2}))?", lowercase(strip(units)))
    isnothing(matched) && return nothing
    unit_name = matched.captures[1]
    date_text = matched.captures[2]
    time_text = something(matched.captures[3], "00:00:00")
    base_dt = DateTime("$(date_text)T$(time_text)")
    return unit_name, base_dt
end

function _decode_caravan_dates(date_var)
    raw_values = date_var[:]
    parsed_dates = Date[]
    parse_ok = true
    for raw in raw_values
        parsed = _try_parse_date(raw)
        if isnothing(parsed)
            parse_ok = false
            break
        end
        push!(parsed_dates, parsed)
    end
    parse_ok && !isempty(parsed_dates) && return parsed_dates

    units = get(date_var.attrib, "units", nothing)
    if raw_values isa AbstractVector{<:Number} && units isa AbstractString
        decoded = _parse_cf_time_units(units)
        if !isnothing(decoded)
            unit_name, base_dt = decoded
            for raw in raw_values
                dt = if unit_name == "days"
                    base_dt + Day(round(Int, raw))
                elseif unit_name == "hours"
                    base_dt + Hour(round(Int, raw))
                else
                    base_dt + Dates.Second(round(Int, raw))
                end
                push!(parsed_dates, Date(dt))
            end
            return parsed_dates
        end
    end

    throw(ArgumentError("Could not decode Caravan 'date' coordinate into calendar dates"))
end

function _read_caravan_series(ds, candidate_names::Vector{String}, field_name::String)
    available = Set(String.(collect(keys(ds))))
    for name in candidate_names
        name in available || continue
        return _numeric_vector_from_caravan_var(ds[name][:]), name
    end
    throw(ArgumentError(
        "Caravan source is missing required $(field_name) variable. Tried: $(join(candidate_names, ", ")). Available: $(join(sort!(collect(available)), ", "))",
    ))
end

function _extract_caravan_streamflow_units(ds, flow_variable_name::String)
    if haskey(ds, flow_variable_name)
        flow_var = ds[flow_variable_name]
        for key in ("units", "unit", "Units")
            value = get(flow_var.attrib, key, nothing)
            value isa AbstractString && !isempty(strip(value)) && return strip(String(value))
        end
    end

    global_units = get(ds.attrib, "Units", nothing)
    if global_units isa AbstractString
        for line in split(global_units, '\n')
            stripped = strip(line)
            startswith(lowercase(stripped), "streamflow:") || continue
            parts = split(stripped, ':'; limit = 2)
            length(parts) == 2 || continue
            candidate = strip(parts[2])
            isempty(candidate) || return candidate
        end
    end

    return nothing
end

function _load_caravan_metadata(attributes_root, dataset_name::String, canonical_gauge_id::String)
    isnothing(attributes_root) && return Dict{String,Any}()
    attr_path = joinpath(attributes_root, dataset_name, "attributes_other_$(dataset_name).csv")
    isfile(attr_path) || return Dict{String,Any}()

    df = CSV.read(attr_path, DataFrame; types = Dict(:gauge_id => String))
    :gauge_id in propertynames(df) || return Dict{String,Any}()

    matches = findall(==(canonical_gauge_id), df[!, :gauge_id])
    isempty(matches) && return Dict{String,Any}()
    row = df[first(matches), :]

    metadata = Dict{String,Any}("attributes_path" => attr_path)
    for column_name in names(df)
        column_name == :gauge_id && continue
        value = row[column_name]
        value isa Missing && continue
        metadata[string(column_name)] = value
    end
    return metadata
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
    throw(ArgumentError("CAMELS routing has been retired. Use Caravan with source_type=caravan, dataset_name=camels, and gauge_id/gage_id instead."))
end

function load_data(::Val{:caravan}, config::AbstractDict)
    dataset_root, netcdf_root, attributes_root, direct_file_path = _resolve_caravan_roots(config)
    dataset_path, dataset_name, local_gauge_id, canonical_gauge_id = _resolve_caravan_timeseries_file(config, netcdf_root, direct_file_path)

    ds = NCDataset(dataset_path)
    try
        dates = _decode_caravan_dates(ds["date"])
        p, p_column = _read_caravan_series(ds, ["total_precipitation_sum", "total_precipitation"], "precipitation")
        t, t_column = _read_caravan_series(ds, ["temperature_2m_mean", "temperature_2m"], "temperature")
        ep, ep_column = _read_caravan_series(ds, ["potential_evaporation_sum", "potential_evaporation"], "potential evaporation")
        flow_mm, flow_column = _read_caravan_series(ds, ["streamflow"], "streamflow")
        flow_units = _extract_caravan_streamflow_units(ds, flow_column)

        tidx = Float64.(dayofyear.(dates))
        n_samples = minimum((length(dates), length(p), length(t), length(ep), length(flow_mm), length(tidx)))
        n_samples > 0 || throw(ArgumentError("Caravan series is empty for gauge $canonical_gauge_id"))

        dates = dates[1:n_samples]
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

        isempty(p) && throw(ArgumentError("Caravan series contains no valid rows after NaN filtering for gauge $canonical_gauge_id"))

        metadata_row = _load_caravan_metadata(attributes_root, dataset_name, canonical_gauge_id)
        area_km2 = get(metadata_row, "area", nothing)

        data_pairs = Pair{Symbol,Vector{Float64}}[
            :P => p,
            :T => t,
            :Ep => ep,
            :Tidx => tidx,
            Symbol("flow(mm)") => flow_mm,
        ]

        metadata = (
            path = dataset_path,
            dataset_root = dataset_root,
            source_type = "caravan",
            dataset_name = dataset_name,
            gauge_id = canonical_gauge_id,
            gage_id = local_gauge_id,
            local_gauge_id = local_gauge_id,
            area_km2 = area_km2,
            flow_units = something(flow_units, "mm/day"),
            dropped_nan_rows = n_invalid,
            column_names = ["P", "T", "Ep", "Tidx", "flow(mm)"],
            source_columns = Dict(
                "P" => p_column,
                "T" => t_column,
                "Ep" => ep_column,
                "flow(mm)" => flow_column,
            ),
            dates = dates,
            metadata = metadata_row,
        )

        return (; data_pairs...), metadata
    finally
        close(ds)
    end
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
