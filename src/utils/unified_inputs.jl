module UnifiedInputs

using ..DataLoader
using ..DataFrames
using ..HydroModels
using ..JSON3
using ..Redis
using ..Statistics

import ..get_data
import ..has_data
import ..LAST_CALIBRATION_RESULT_HANDLE
import ..LAST_SIMULATION_RESULT_HANDLE

const FORCING_ALIASES = Dict(
    "P" => [
        "P", "prcp", "prcp(mm/day)", "precip", "precipitation", "rain", "rainfall",
        "ppt", "precip_mm_day", "rain_mm_day",
    ],
    "T" => [
        "T", "temp", "temperature", "tmean", "tmean(C)", "tavg", "air_temperature",
        "mean_temperature",
    ],
    "Ep" => [
        "Ep", "pet", "pet(mm)", "etp", "evap", "evapotranspiration",
        "potential_evapotranspiration", "potential_evaporation",
    ],
)

const OBSERVED_ALIASES = [
    "obs", "observed", "qobs", "qmean", "q", "runoff", "discharge", "streamflow",
    "flow", "flow(mm)", "target",
]

const RUNTIME_FIELDS = Set([
    "solver", "interpolation", "seed", "period", "warmup", "init_states",
    "config", "output_variable", "input_mapping",
])

const COMMON_SOURCE_TYPES = Set([:csv, :json, :redis, :camels, :npz, :data_handle])
const PARAMETER_SOURCE_TYPES = Set([:csv, :json, :redis, :camels, :npz, :data_handle, :calibration_result])

const SOLVER_ALIASES = Dict{String,String}(
    "ode" => "ODE",
    "continuous" => "ODE",
    "continuoussolver" => "ODE",
    "连续" => "ODE",
    "连续求解" => "ODE",
    "连续求解器" => "ODE",
    "离散" => "DISCRETE",
    "离散求解" => "DISCRETE",
    "离散求解器" => "DISCRETE",
    "discrete" => "DISCRETE",
    "discretesolver" => "DISCRETE",
    "mutable" => "MUTABLE",
    "mutablesolver" => "MUTABLE",
    "可变" => "MUTABLE",
    "可变求解" => "MUTABLE",
    "可变求解器" => "MUTABLE",
    "immutable" => "IMMUTABLE",
    "immutablesolver" => "IMMUTABLE",
    "不可变" => "IMMUTABLE",
    "不可变求解" => "IMMUTABLE",
    "不可变求解器" => "IMMUTABLE",
)

const INTERPOLATION_ALIASES = Dict{String,String}(
    "linear" => "LINEAR",
    "line" => "LINEAR",
    "线性" => "LINEAR",
    "线性插值" => "LINEAR",
    "constant" => "CONSTANT",
    "step" => "CONSTANT",
    "piecewiseconstant" => "CONSTANT",
    "阶梯" => "CONSTANT",
    "常量" => "CONSTANT",
    "常量插值" => "CONSTANT",
    "direct" => "DIRECT",
    "raw" => "DIRECT",
    "identity" => "DIRECT",
    "直接" => "DIRECT",
    "直接映射" => "DIRECT",
)

_as_dict(value::AbstractDict) = Dict{String,Any}(string(k) => v for (k, v) in pairs(value))

function _normalize_name(name::AbstractString)
    lowered = lowercase(strip(String(name)))
    return replace(lowered, r"[^a-z0-9]+" => "")
end

function _as_float_vector(value)
    if value isa AbstractVector
        return Float64.(collect(value))
    elseif value isa AbstractArray
        ndims(value) == 1 && return Float64.(collect(value))
        (ndims(value) == 2 && min(size(value)...) == 1) && return Float64.(vec(value))
        throw(ArgumentError("Expected a 1D numeric array, got size $(size(value))"))
    end

    throw(ArgumentError("Expected numeric array, got $(typeof(value))"))
end

function _series_score_for_target(target::String, values::Vector{Float64})
    isempty(values) && return 0.0

    nonneg_ratio = count(v -> isfinite(v) && v >= 0, values) / length(values)
    zero_ratio = count(v -> isfinite(v) && abs(v) <= 1e-9, values) / length(values)
    finite_vals = filter(isfinite, values)
    variation_ok = length(finite_vals) > 1 && std(finite_vals) > 0

    if target == "P"
        return 0.25 * nonneg_ratio + 0.10 * zero_ratio + (variation_ok ? 0.05 : 0.0)
    elseif target == "Ep"
        return 0.30 * nonneg_ratio + 0.05 * zero_ratio + (variation_ok ? 0.03 : 0.0)
    elseif target == "T"
        return (variation_ok ? 0.08 : 0.0) + (nonneg_ratio > 0.2 ? 0.04 : 0.0)
    end

    return 0.0
end

function _name_score(target::String, candidate::String, aliases::Vector{String})
    target_norm = _normalize_name(target)
    cand_norm = _normalize_name(candidate)
    target_norm == cand_norm && return 1.0, "exact"

    alias_norms = _normalize_name.(aliases)
    if cand_norm in alias_norms
        return 0.9, "alias"
    end

    for alias_norm in alias_norms
        occursin(alias_norm, cand_norm) && return 0.72, "partial_alias"
        occursin(cand_norm, alias_norm) && return 0.66, "partial_alias"
    end

    return 0.0, "none"
end

function _jsonish_to_object(value)
    if value isa AbstractDict
        return _as_dict(value)
    elseif value isa AbstractString
        parsed = JSON3.read(String(value))
        parsed isa AbstractDict || throw(ArgumentError("JSON text must decode to an object"))
        return _as_dict(parsed)
    end

    throw(ArgumentError("Expected JSON object or JSON text, got $(typeof(value))"))
end

function _parse_int_like(value, field::String)
    if value isa Integer
        return Int(value)
    elseif value isa AbstractFloat
        rounded = round(Int, value)
        rounded == value || throw(ArgumentError("$field must be integer-compatible"))
        return rounded
    end

    text = strip(string(value))
    isempty(text) && throw(ArgumentError("$field cannot be empty"))
    return parse(Int, text)
end

function _to_string_vector(value, field::String)
    value isa AbstractVector || throw(ArgumentError("$field must be an array"))
    return String[string(v) for v in value]
end

function _as_bool(value, field::String)
    if value isa Bool
        return value
    elseif value isa AbstractString
        lower = lowercase(strip(String(value)))
        lower in ("true", "1", "yes", "y") && return true
        lower in ("false", "0", "no", "n") && return false
    end
    throw(ArgumentError("$field must be a boolean"))
end

function _normalize_choice(value, field::String, aliases::Dict{String,String}, allowed::Set{String})
    raw = strip(string(value))
    isempty(raw) && throw(ArgumentError("$field cannot be empty"))

    ascii_key = _normalize_name(raw)
    multilingual_key = lowercase(replace(raw, r"[^0-9A-Za-z一-龥]+" => ""))
    canonical = if haskey(aliases, ascii_key)
        aliases[ascii_key]
    elseif haskey(aliases, multilingual_key)
        aliases[multilingual_key]
    else
        uppercase(raw)
    end

    canonical in allowed || throw(ArgumentError(
        "$field must be one of: $(join(sort!(collect(allowed)), ", "))",
    ))
    return canonical
end

function _extract_handle_identifier(spec::Dict{String,Any})
    for key in ("handle", "data_handle", "key", "id", "name")
        haskey(spec, key) || continue
        value = strip(string(spec[key]))
        isempty(value) || return value
    end
    return nothing
end

_supported_source_types(role::String) = role == "parameters" ? PARAMETER_SOURCE_TYPES : COMMON_SOURCE_TYPES

function _normalize_source_body(::Val{:csv}, spec::Dict{String,Any}, role::String)
    haskey(spec, "path") || throw(ArgumentError("inputs.$role.path is required for csv source"))
    normalized = Dict{String,Any}("path" => string(spec["path"]))
    haskey(spec, "column") && (normalized["column"] = string(spec["column"]))
    haskey(spec, "row_index") && (normalized["row_index"] = _parse_int_like(spec["row_index"], "inputs.$role.row_index"))
    return normalized
end

function _normalize_source_body(::Val{:json}, spec::Dict{String,Any}, role::String)
    normalized = Dict{String,Any}()
    if haskey(spec, "path")
        normalized["path"] = string(spec["path"])
    elseif haskey(spec, "data")
        normalized["data"] = spec["data"]
    else
        throw(ArgumentError("inputs.$role json source requires either path or data"))
    end
    haskey(spec, "column") && (normalized["column"] = string(spec["column"]))
    return normalized
end

function _normalize_source_body(::Val{:redis}, spec::Dict{String,Any}, role::String)
    haskey(spec, "key") || throw(ArgumentError("inputs.$role.key is required for redis source"))
    normalized = Dict{String,Any}("key" => string(spec["key"]))
    haskey(spec, "host") && (normalized["host"] = string(spec["host"]))
    haskey(spec, "port") && (normalized["port"] = _parse_int_like(spec["port"], "inputs.$role.port"))
    haskey(spec, "column") && (normalized["column"] = string(spec["column"]))
    return normalized
end

function _normalize_source_body(::Val{:camels}, spec::Dict{String,Any}, role::String)
    gage_id = if haskey(spec, "gage_id")
        _parse_int_like(spec["gage_id"], "inputs.$role.gage_id")
    elseif haskey(spec, "gauge_id")
        _parse_int_like(spec["gauge_id"], "inputs.$role.gage_id")
    else
        throw(ArgumentError("inputs.$role.gage_id is required for camels source"))
    end

    normalized = Dict{String,Any}("gage_id" => gage_id)
    haskey(spec, "dataset_path") && (normalized["dataset_path"] = string(spec["dataset_path"]))
    return normalized
end

function _normalize_source_body(::Val{:npz}, spec::Dict{String,Any}, role::String)
    haskey(spec, "path") || throw(ArgumentError("inputs.$role.path is required for npz source"))
    normalized = Dict{String,Any}("path" => string(spec["path"]))
    haskey(spec, "column") && (normalized["column"] = string(spec["column"]))
    return normalized
end

function _normalize_source_body(::Val{:calibration_result}, spec::Dict{String,Any}, role::String)
    normalized = Dict{String,Any}()
    if haskey(spec, "data")
        normalized["data"] = spec["data"]
    elseif haskey(spec, "path")
        normalized["path"] = string(spec["path"])
    else
        handle = _extract_handle_identifier(spec)
        isnothing(handle) && throw(ArgumentError("inputs.$role requires one of: data, path, or handle/data_handle/key for calibration_result source"))
        normalized["handle"] = handle
    end
    haskey(spec, "column") && (normalized["column"] = string(spec["column"]))
    return normalized
end

function _normalize_source_body(::Val{:data_handle}, spec::Dict{String,Any}, role::String)
    handle = _extract_handle_identifier(spec)
    if isnothing(handle)
        if role == "parameters"
            throw(ArgumentError("inputs.$role.handle is required for data_handle source (or omit inputs.parameters to reuse same-session calibrated parameters)"))
        end
        throw(ArgumentError("inputs.$role.handle is required for data_handle source"))
    end

    normalized = Dict{String,Any}("handle" => handle)
    haskey(spec, "column") && (normalized["column"] = string(spec["column"]))
    return normalized
end

function _normalize_source_body(::Val{T}, _spec::Dict{String,Any}, role::String) where {T}
    throw(ArgumentError("Unsupported inputs.$role.source_type: $(String(T))"))
end

function _normalize_source_spec(raw_spec, role::String)
    if role == "parameters"
        if raw_spec isa AbstractVector
            return Dict{String,Any}(
                "source_type" => "json",
                "data" => collect(raw_spec),
            )
        elseif raw_spec isa AbstractDict && !haskey(raw_spec, "source_type") && !haskey(raw_spec, :source_type)
            return Dict{String,Any}(
                "source_type" => "json",
                "data" => _as_dict(raw_spec),
            )
        end
    elseif role == "runtime"
        if raw_spec isa AbstractDict && !haskey(raw_spec, "source_type") && !haskey(raw_spec, :source_type)
            return Dict{String,Any}(
                "source_type" => "json",
                "data" => _as_dict(raw_spec),
            )
        end
    end

    raw_spec isa AbstractDict || throw(ArgumentError("inputs.$role must be an object"))
    spec = _as_dict(raw_spec)

    haskey(spec, "source_type") ||
        throw(ArgumentError("inputs.$role.source_type is required"))

    source_type = Symbol(lowercase(string(spec["source_type"])))
    source_type in _supported_source_types(role) || throw(ArgumentError(
        "Unsupported inputs.$role.source_type: $(String(source_type))",
    ))

    normalized = Dict{String,Any}("source_type" => string(source_type))
    merge!(normalized, _normalize_source_body(Val(source_type), spec, role))

    return normalized
end

function _normalize_inputs_block(raw_inputs)
    raw_inputs isa AbstractDict || throw(ArgumentError("inputs must be an object"))
    inputs = _as_dict(raw_inputs)
    haskey(inputs, "forcing") || throw(ArgumentError("inputs.forcing is required"))

    normalized = Dict{String,Any}(
        "forcing" => _normalize_source_spec(inputs["forcing"], "forcing"),
    )

    haskey(inputs, "observation") && (normalized["observation"] = _normalize_source_spec(inputs["observation"], "observation"))
    haskey(inputs, "parameters") && (normalized["parameters"] = _normalize_source_spec(inputs["parameters"], "parameters"))
    haskey(inputs, "runtime") && (normalized["runtime"] = _normalize_source_spec(inputs["runtime"], "runtime"))

    return normalized
end

function _normalize_options_block(raw_options)
    if raw_options === nothing
        return Dict{String,Any}(
            "auto_infer" => true,
            "strict_infer" => false,
            "input_mapping" => nothing,
        )
    end

    raw_options isa AbstractDict || throw(ArgumentError("options must be an object"))
    options = _as_dict(raw_options)

    input_mapping = if haskey(options, "input_mapping")
        mapping = options["input_mapping"]
        if mapping === nothing
            nothing
        else
            mapping isa AbstractDict || throw(ArgumentError("options.input_mapping must be an object"))
            Dict{String,String}(string(k) => string(v) for (k, v) in pairs(mapping))
        end
    else
        nothing
    end

    return Dict{String,Any}(
        "auto_infer" => haskey(options, "auto_infer") ? _as_bool(options["auto_infer"], "options.auto_infer") : true,
        "strict_infer" => haskey(options, "strict_infer") ? _as_bool(options["strict_infer"], "options.strict_infer") : false,
        "input_mapping" => input_mapping,
    )
end

function _normalize_output_block(raw_output, forcing_source_type::String)
    default_result_type = forcing_source_type == "camels" ? "csv" : forcing_source_type

    if raw_output === nothing
        output = Dict{String,Any}("result_source_type" => default_result_type)
        default_result_type != "redis" && (output["output_dir"] = "./result")
        return output
    end

    raw_output isa AbstractDict || throw(ArgumentError("output must be an object"))
    output_in = _as_dict(raw_output)
    result_source_type = lowercase(string(get(output_in, "result_source_type", default_result_type)))
    result_source_type in ("csv", "json", "redis") ||
        throw(ArgumentError("output.result_source_type must be one of: csv, json, redis"))

    output = Dict{String,Any}("result_source_type" => result_source_type)

    if haskey(output_in, "output_dir")
        output["output_dir"] = string(output_in["output_dir"])
    elseif result_source_type != "redis"
        output["output_dir"] = "./result"
    end

    haskey(output_in, "result_key") && (output["result_key"] = string(output_in["result_key"]))
    haskey(output_in, "result_host") && (output["result_host"] = string(output_in["result_host"]))
    haskey(output_in, "result_port") && (output["result_port"] = _parse_int_like(output_in["result_port"], "output.result_port"))

    return output
end

function normalize_workflow_request(args::AbstractDict)
    req = _as_dict(args)
    haskey(req, "model") || throw(ArgumentError("model is required"))
    haskey(req, "inputs") || throw(ArgumentError("inputs is required"))

    inputs = _normalize_inputs_block(req["inputs"])
    options = _normalize_options_block(get(req, "options", nothing))
    output = _normalize_output_block(get(req, "output", nothing), inputs["forcing"]["source_type"])

    normalized = Dict{String,Any}(
        "model" => string(req["model"]),
        "inputs" => inputs,
        "options" => options,
        "output" => output,
    )

    for optional_key in (
        "period", "warmup", "metrics", "method", "ratio", "threshold", "objective",
        "algorithm", "n_samples", "n_trials", "maxiters", "population_size", "log_transform",
        "fixed_params", "param_bounds", "objectives", "parameter_sets", "calibration_period",
        "validation_period", "obs_column", "goal", "budget", "sensitivity_samples",
        "boundary_tolerance", "convergence_threshold", "plateau_window", "custom_metrics",
        "custom_weights", "parallel", "save_to_storage", "objective_threshold", "trial_results",
    )
        haskey(req, optional_key) && (normalized[optional_key] = req[optional_key])
    end

    return normalized
end

function _load_source(spec::Dict{String,Any}, role::String)
    source_type = Symbol(spec["source_type"])
    return _load_source(Val(source_type), spec, role)
end

function _load_source(::Val{:data_handle}, spec::Dict{String,Any}, _role::String)
    handle = string(spec["handle"])
    has_data(handle) || throw(ArgumentError("Data handle '$handle' not found"))
    payload = get_data(handle)
    return payload, (; source_type = "data_handle", data_handle = handle)
end

function _load_source(::Val{T}, spec::Dict{String,Any}, _role::String) where {T}
    payload, metadata = DataLoader.load_data(Val(T), spec)
    return payload, metadata
end

function _extract_numeric_columns(payload::NamedTuple)
    columns = Dict{String,Vector{Float64}}()

    for name in keys(payload)
        value = payload[name]
        try
            columns[string(name)] = _as_float_vector(value)
        catch
        end
    end

    return columns
end

function _extract_numeric_columns(payload::AbstractDict)
    if haskey(payload, "forcing_nt")
        return _extract_numeric_columns(payload["forcing_nt"])
    end

    columns = Dict{String,Vector{Float64}}()
    for (k, v) in pairs(payload)
        try
            columns[string(k)] = _as_float_vector(v)
        catch
        end
    end

    return columns
end

function _extract_numeric_columns(payload)
    throw(ArgumentError("Unsupported payload type $(typeof(payload)) for numeric column extraction"))
end

function _normalize_runtime_map(raw_runtime::AbstractDict)
    runtime = Dict{String,Any}(string(k) => v for (k, v) in pairs(raw_runtime))
    result = Dict{String,Any}()

    solver = _normalize_choice(
        get(runtime, "solver", "DISCRETE"),
        "runtime.solver",
        SOLVER_ALIASES,
        Set(["ODE", "DISCRETE", "MUTABLE", "IMMUTABLE"]),
    )
    result["solver"] = solver

    interpolation = _normalize_choice(
        get(runtime, "interpolation", "LINEAR"),
        "runtime.interpolation",
        INTERPOLATION_ALIASES,
        Set(["LINEAR", "CONSTANT", "DIRECT"]),
    )
    result["interpolation"] = interpolation

    if haskey(runtime, "seed") && !(runtime["seed"] === nothing)
        result["seed"] = _parse_int_like(runtime["seed"], "runtime.seed")
    end
    if haskey(runtime, "warmup") && !(runtime["warmup"] === nothing)
        result["warmup"] = _parse_int_like(runtime["warmup"], "runtime.warmup")
    end
    if haskey(runtime, "period") && !(runtime["period"] === nothing)
        result["period"] = _to_string_vector(runtime["period"], "runtime.period")
    end

    if haskey(runtime, "init_states") && !(runtime["init_states"] === nothing)
        init_raw = runtime["init_states"]
        init_raw isa AbstractDict || throw(ArgumentError("runtime.init_states must be an object"))
        result["init_states"] = Dict{String,Float64}(string(k) => Float64(v) for (k, v) in pairs(init_raw))
    end

    if haskey(runtime, "input_mapping") && !(runtime["input_mapping"] === nothing)
        mapping_raw = runtime["input_mapping"]
        mapping_raw isa AbstractDict || throw(ArgumentError("runtime.input_mapping must be an object"))
        result["input_mapping"] = Dict{String,String}(string(k) => string(v) for (k, v) in pairs(mapping_raw))
    end

    if haskey(runtime, "config") && !(runtime["config"] === nothing)
        cfg_raw = runtime["config"]
        cfg_raw isa AbstractDict || throw(ArgumentError("runtime.config must be an object"))
        result["config"] = Dict{String,Any}(string(k) => v for (k, v) in pairs(cfg_raw))
    end

    if haskey(runtime, "output_variable") && !(runtime["output_variable"] === nothing)
        config = get(result, "config", Dict{String,Any}())
        config["output_variable"] = string(runtime["output_variable"])
        result["config"] = config
    end

    return result
end

function _resolve_runtime(spec::Nothing)
    return Dict{String,Any}(
        "solver" => "DISCRETE",
        "interpolation" => "LINEAR",
    ), Dict{String,Any}("source" => "default")
end

function _resolve_runtime(spec::Dict{String,Any})
    source_type = Symbol(spec["source_type"])
    return _resolve_runtime_source(Val(source_type), spec)
end

function _resolve_runtime_source(::Val{:json}, spec::Dict{String,Any})
    runtime_obj = if haskey(spec, "data")
        _jsonish_to_object(spec["data"])
    else
        parsed = JSON3.read(read(spec["path"], String))
        parsed isa AbstractDict || throw(ArgumentError("runtime JSON file must decode to an object"))
        _as_dict(parsed)
    end
    return _normalize_runtime_map(runtime_obj), Dict{String,Any}("source" => "json")
end

function _resolve_runtime_source(::Val{:csv}, spec::Dict{String,Any})
    payload, _ = DataLoader.load_data(Val(:csv), spec)
    columns = _extract_numeric_columns(payload)
    row_index = Int(get(spec, "row_index", 1))
    runtime_obj = Dict{String,Any}()
    for (k, v) in columns
        row_index <= length(v) || continue
        k in RUNTIME_FIELDS || continue
        runtime_obj[k] = v[row_index]
    end
    return _normalize_runtime_map(runtime_obj), Dict{String,Any}("source" => "csv")
end

function _resolve_runtime_source(::Val{:data_handle}, spec::Dict{String,Any})
    payload, _ = _load_source(Val(:data_handle), spec, "runtime")
    payload isa AbstractDict || throw(ArgumentError("runtime data_handle must resolve to an object"))
    return _normalize_runtime_map(payload), Dict{String,Any}("source" => "data_handle")
end

function _resolve_runtime_source(::Val{:redis}, spec::Dict{String,Any})
    host = string(get(spec, "host", "127.0.0.1"))
    port = Int(get(spec, "port", 6379))
    conn = Redis.RedisConnection(host = host, port = port)
    try
        payload = Redis.get(conn, string(spec["key"]))
        isnothing(payload) && throw(ArgumentError("Redis key not found"))
        parsed = JSON3.read(payload)
        parsed isa AbstractDict || throw(ArgumentError("runtime redis payload must decode to an object"))
        return _normalize_runtime_map(_as_dict(parsed)), Dict{String,Any}("source" => "redis")
    finally
        try
            close(conn)
        catch
        end
    end
end

function _resolve_runtime_source(::Val{T}, _spec::Dict{String,Any}) where {T}
    throw(ArgumentError("runtime source_type '$(String(T))' is not supported"))
end

function _extract_parameter_payload(raw)
    if raw isa AbstractString
        parsed = JSON3.read(String(raw))
        return _extract_parameter_payload(parsed)
    elseif raw isa AbstractDict
        obj = _as_dict(raw)
        for key in (
            "best_params", "best_parameters", "calibrated_params", "final_parameters",
            "params_used", "parameters", "params",
        )
            haskey(obj, key) || continue
            nested = obj[key]
            if nested isa AbstractDict || nested isa AbstractVector || nested isa AbstractString
                return nested, key
            end
        end
        return obj, "object"
    end

    return raw, "value"
end

function _resolve_parameter_object(raw_obj::AbstractDict, param_names::Vector{String}; fallback_params = nothing)
    obj = _as_dict(raw_obj)
    lookup = Dict(_normalize_name(k) => k for k in keys(obj))

    resolved = Dict{String,Float64}()
    missing = String[]
    provided = String[]

    for pname in param_names
        key = get(lookup, _normalize_name(pname), nothing)
        if isnothing(key)
            push!(missing, pname)
            continue
        end

        resolved[pname] = Float64(obj[key])
        push!(provided, pname)
    end

    filled_from_fallback = String[]
    if !(fallback_params === nothing)
        for pname in missing
            haskey(fallback_params, pname) || continue
            resolved[pname] = Float64(fallback_params[pname])
            push!(filled_from_fallback, pname)
        end
    end

    remaining_missing = [pname for pname in missing if !(pname in filled_from_fallback)]
    return resolved, remaining_missing, provided, filled_from_fallback
end

function _resolve_session_parameter_fallback(model)
    param_names = String.(HydroModels.get_param_names(model))
    for (handle, origin) in (
        (LAST_CALIBRATION_RESULT_HANDLE, "last_calibration"),
        (LAST_SIMULATION_RESULT_HANDLE, "last_simulation"),
    )
        has_data(handle) || continue
        payload = get_data(handle)
        candidate, extracted_from = _extract_parameter_payload(payload)

        if candidate isa AbstractVector
            length(candidate) == length(param_names) || continue
            resolved = Dict{String,Float64}(
                param_names[idx] => Float64(candidate[idx]) for idx in eachindex(param_names)
            )
            return resolved, Dict{String,Any}(
                "source" => "session_fallback",
                "origin" => origin,
                "handle" => handle,
                "format" => "array",
                "extracted_from" => extracted_from,
            )
        elseif candidate isa AbstractDict
            resolved, missing, _, _ = _resolve_parameter_object(candidate, param_names)
            isempty(missing) || continue
            return resolved, Dict{String,Any}(
                "source" => "session_fallback",
                "origin" => origin,
                "handle" => handle,
                "format" => "object",
                "extracted_from" => extracted_from,
            )
        end
    end

    return nothing, nothing
end

function _load_parameter_raw(::Val{:json}, spec::Dict{String,Any})
    return haskey(spec, "data") ? spec["data"] : JSON3.read(read(spec["path"], String))
end

function _load_parameter_raw(::Val{:data_handle}, spec::Dict{String,Any})
    payload, _ = _load_source(Val(:data_handle), spec, "parameters")
    return payload
end

function _load_parameter_raw(::Val{:calibration_result}, spec::Dict{String,Any})
    if haskey(spec, "data")
        return spec["data"]
    elseif haskey(spec, "path")
        return JSON3.read(read(spec["path"], String))
    end

    haskey(spec, "handle") || throw(ArgumentError("inputs.parameters.handle is required for calibration_result source"))
    return get_data(string(spec["handle"]))
end

function _load_parameter_raw(::Val{:csv}, spec::Dict{String,Any})
    payload, _ = DataLoader.load_data(Val(:csv), spec)
    columns = _extract_numeric_columns(payload)
    row_index = Int(get(spec, "row_index", 1))
    return Dict{String,Any}(k => v[row_index] for (k, v) in columns if row_index <= length(v))
end

function _load_parameter_raw(::Val{:redis}, spec::Dict{String,Any})
    host = string(get(spec, "host", "127.0.0.1"))
    port = Int(get(spec, "port", 6379))
    conn = Redis.RedisConnection(host = host, port = port)
    try
        payload = Redis.get(conn, string(spec["key"]))
        if isnothing(payload)
            missing_key = string(spec["key"])
            throw(ArgumentError("Redis key not found: $missing_key"))
        end
        return JSON3.read(payload)
    finally
        try
            close(conn)
        catch
        end
    end
end

function _load_parameter_raw(::Val{T}, _spec::Dict{String,Any}) where {T}
    throw(ArgumentError("parameters source_type '$(String(T))' is not supported"))
end

function _resolve_parameters(
    spec::Union{Nothing,Dict{String,Any}},
    model;
    fallback_params = nothing,
    require_complete::Bool = true,
)
    if isnothing(spec)
        return nothing, Dict{String,Any}("source" => "none"), String[]
    end

    param_names = String.(HydroModels.get_param_names(model))
    source_type = string(spec["source_type"])
    raw = _load_parameter_raw(Val(Symbol(source_type)), spec)

    candidate, extracted_from = _extract_parameter_payload(raw)

    if candidate isa AbstractVector
        length(candidate) == length(param_names) || throw(ArgumentError(
            "Parameter vector length $(length(candidate)) does not match model parameter count $(length(param_names))",
        ))
        return Float64.(collect(candidate)), Dict{String,Any}(
            "source" => source_type,
            "format" => "array",
            "extracted_from" => extracted_from,
        ), String[]
    elseif candidate isa AbstractDict
        resolved, missing, provided, filled_from_fallback = _resolve_parameter_object(
            candidate,
            param_names;
            fallback_params = fallback_params,
        )

        if require_complete && !isempty(missing)
            throw(ArgumentError("Missing parameter(s) in parameters source: $(join(missing, ", "))"))
        end

        warnings = String[]
        if !isempty(filled_from_fallback)
            push!(warnings, "Filled missing parameters from same-session calibration/simulation context: $(join(filled_from_fallback, ", "))")
        end
        if !require_complete && !isempty(missing)
            push!(warnings, "Parameters remain partial and will be completed downstream when needed: $(join(missing, ", "))")
        end

        return resolved, Dict{String,Any}(
            "source" => source_type,
            "format" => "object",
            "extracted_from" => extracted_from,
            "provided_parameters" => provided,
            "missing_parameters" => missing,
            "filled_from_fallback" => filled_from_fallback,
        ), warnings
    end

    throw(ArgumentError("parameters source must decode to an object or array"))
end

function _resolve_input_mapping(target::String, input_mapping::Union{Nothing,Dict{String,String}}, available::Set{String})
    isnothing(input_mapping) && return nothing

    if haskey(input_mapping, target)
        mapped = input_mapping[target]
        mapped in available && return mapped
    end

    for (source_col, mapped_target) in input_mapping
        mapped_target == target || continue
        source_col in available && return source_col
    end

    return nothing
end

function _infer_forcing_mapping(
    required_inputs::Vector{String},
    columns::Dict{String,Vector{Float64}};
    input_mapping::Union{Nothing,Dict{String,String}} = nothing,
    strict_infer::Bool = false,
    auto_infer::Bool = true,
)
    available_names = collect(keys(columns))
    available_set = Set(available_names)
    used_columns = Set{String}()

    chosen = Dict{String,String}()
    report = Dict{String,Any}()
    warnings = String[]

    for target in required_inputs
        entry = Dict{String,Any}()

        mapped = _resolve_input_mapping(target, input_mapping, available_set)
        if !isnothing(mapped)
            chosen[target] = mapped
            push!(used_columns, mapped)
            entry["column"] = mapped
            entry["confidence"] = 1.0
            entry["method"] = "explicit_mapping"
            entry["candidates"] = Any[]
            report[target] = entry
            continue
        end

        auto_infer || throw(ArgumentError("Missing explicit input mapping for model input '$target'"))

        aliases = get(FORCING_ALIASES, target, [target])
        scored_candidates = Vector{Tuple{Float64,String,String}}()
        for cname in available_names
            name_score, name_method = _name_score(target, cname, aliases)
            sem_score = _series_score_for_target(target, columns[cname])
            total = name_score + sem_score
            push!(scored_candidates, (total, cname, name_method))
        end

        isempty(scored_candidates) && throw(ArgumentError("No candidate columns available for input '$target'"))

        sort!(scored_candidates; by = x -> x[1], rev = true)

        selected = nothing
        selected_conf = -Inf
        selected_method = "none"

        for (score, cname, method) in scored_candidates
            if !(cname in used_columns)
                selected = cname
                selected_conf = score
                selected_method = method
                break
            end
        end

        if isnothing(selected)
            selected_conf, selected, selected_method = scored_candidates[1]
            push!(warnings, "Input '$target' reuses column '$selected' because no unique candidates were available")
        end

        strict_infer && selected_conf < 0.65 && throw(ArgumentError(
            "Low-confidence forcing inference for '$target' (column '$selected', confidence=$(round(selected_conf, digits = 3))). Provide options.input_mapping or better column names.",
        ))

        if selected_conf < 0.65
            push!(warnings, "Low-confidence inference: mapped '$target' to '$selected' (confidence=$(round(selected_conf, digits = 3)))")
        elseif selected_method != "exact"
            push!(warnings, "Auto-mapped '$target' from column '$selected' via $selected_method inference")
        end

        chosen[target] = selected
        push!(used_columns, selected)

        entry["column"] = selected
        entry["confidence"] = selected_conf
        entry["method"] = selected_method
        entry["candidates"] = [
            Dict(
                "column" => cname,
                "confidence" => score,
                "method" => method,
            ) for (score, cname, method) in scored_candidates[1:min(5, length(scored_candidates))]
        ]
        report[target] = entry
    end

    return chosen, report, warnings
end

function _align_forcing_lengths(required_inputs::Vector{String}, columns::Dict{String,Vector{Float64}}, chosen::Dict{String,String})
    lengths = [length(columns[chosen[name]]) for name in required_inputs]
    min_len = minimum(lengths)
    forcing_vectors = Vector{Vector{Float64}}(undef, length(required_inputs))
    for (idx, name) in enumerate(required_inputs)
        forcing_vectors[idx] = columns[chosen[name]][1:min_len]
    end
    return forcing_vectors, min_len
end

function resolve_forcing(
    forcing_spec::Dict{String,Any},
    model;
    input_mapping::Union{Nothing,Dict{String,String}} = nothing,
    strict_infer::Bool = false,
    auto_infer::Bool = true,
)
    payload, metadata = _load_source(forcing_spec, "forcing")
    columns = _extract_numeric_columns(payload)
    isempty(columns) && throw(ArgumentError("No numeric forcing columns were found in forcing source"))

    required_inputs = String.(HydroModels.get_input_names(model))
    chosen, report, warnings = _infer_forcing_mapping(
        required_inputs,
        columns;
        input_mapping = input_mapping,
        strict_infer = strict_infer,
        auto_infer = auto_infer,
    )

    vectors, min_len = _align_forcing_lengths(required_inputs, columns, chosen)
    if any(length(columns[chosen[name]]) != min_len for name in required_inputs)
        push!(warnings, "Forcing columns have different lengths; truncated all inputs to $min_len")
    end

    symbols = Tuple(Symbol.(required_inputs))
    forcing_nt = NamedTuple{symbols}(Tuple(vectors))

    inference_report = Dict{String,Any}(
        "role" => "forcing",
        "required_inputs" => required_inputs,
        "mapping" => Dict(name => chosen[name] for name in required_inputs),
        "details" => report,
    )

    return forcing_nt, metadata, inference_report, warnings
end

function _infer_observation_column(
    columns::Dict{String,Vector{Float64}};
    explicit_column::Union{Nothing,String} = nothing,
    strict_infer::Bool = false,
)
    available_names = collect(keys(columns))
    lookup = Dict(_normalize_name(name) => name for name in available_names)

    if !isnothing(explicit_column)
        key = get(lookup, _normalize_name(explicit_column), nothing)
        isnothing(key) && throw(ArgumentError(
            "Observation column '$explicit_column' not found. Available columns: $(join(sort(available_names), ", "))",
        ))
        return key, 1.0, "explicit", Any[]
    end

    scored = Vector{Tuple{Float64,String,String}}()
    for cname in available_names
        name_score, method = _name_score("obs", cname, OBSERVED_ALIASES)
        series = columns[cname]
        finite = filter(isfinite, series)
        var_score = (length(finite) > 1 && std(finite) > 0) ? 0.05 : 0.0
        nonneg_score = count(v -> isfinite(v) && v >= 0, series) / length(series) * 0.10
        total = name_score + var_score + nonneg_score
        push!(scored, (total, cname, method))
    end

    isempty(scored) && throw(ArgumentError("No numeric columns available for observed runoff inference"))
    sort!(scored; by = x -> x[1], rev = true)
    best_score, best_name, best_method = scored[1]

    strict_infer && best_score < 0.65 && throw(ArgumentError(
        "Low-confidence observed runoff inference for column '$best_name' (confidence=$(round(best_score, digits = 3))). Provide inputs.observation.column explicitly.",
    ))

    candidates = [
        Dict("column" => cname, "confidence" => score, "method" => method)
        for (score, cname, method) in scored[1:min(length(scored), 5)]
    ]

    return best_name, best_score, best_method, candidates
end

function resolve_observation(observation_spec::Dict{String,Any}; strict_infer::Bool = false)
    payload, metadata = _load_source(observation_spec, "observation")

    if payload isa AbstractDict && haskey(payload, "obs")
        obs_vec = Float64.(payload["obs"])
        report = Dict{String,Any}(
            "role" => "observation",
            "column" => get(payload, "obs_column", "obs"),
            "confidence" => 1.0,
            "method" => "direct_obs",
            "candidates" => Any[],
        )
        return obs_vec, metadata, report, String[]
    end

    columns = _extract_numeric_columns(payload)
    isempty(columns) && throw(ArgumentError("No numeric columns were found in observation source"))

    explicit_col = haskey(observation_spec, "column") ? string(observation_spec["column"]) :
        (haskey(observation_spec, "obs_column") ? string(observation_spec["obs_column"]) : nothing)

    chosen_col, confidence, method, candidates = _infer_observation_column(
        columns;
        explicit_column = explicit_col,
        strict_infer = strict_infer,
    )

    warnings = String[]
    if method == "explicit"
        # no-op
    elseif confidence < 0.65
        push!(warnings, "Low-confidence observed runoff inference: '$chosen_col' (confidence=$(round(confidence, digits = 3)))")
    else
        push!(warnings, "Auto-inferred observed runoff column '$chosen_col'")
    end

    report = Dict{String,Any}(
        "role" => "observation",
        "column" => chosen_col,
        "confidence" => confidence,
        "method" => method,
        "candidates" => candidates,
    )

    return columns[chosen_col], metadata, report, warnings
end

function resolve_common_inputs(
    args::AbstractDict,
    model;
    require_observation::Bool = false,
    require_parameters::Bool = false,
    allow_partial_parameters::Bool = false,
)
    request = normalize_workflow_request(args)
    inputs = request["inputs"]
    options = request["options"]

    input_mapping = options["input_mapping"]
    strict_infer = Bool(options["strict_infer"])
    auto_infer = Bool(options["auto_infer"])

    forcing_nt, forcing_meta, forcing_report, forcing_warnings = resolve_forcing(
        inputs["forcing"],
        model;
        input_mapping = input_mapping,
        strict_infer = strict_infer,
        auto_infer = auto_infer,
    )

    obs_vec = nothing
    observation_report = nothing
    observation_meta = nothing
    observation_warnings = String[]
    if require_observation || haskey(inputs, "observation")
        haskey(inputs, "observation") || throw(ArgumentError("inputs.observation is required for this workflow"))
        obs_vec, observation_meta, observation_report, observation_warnings = resolve_observation(
            inputs["observation"];
            strict_infer = strict_infer,
        )
    end

    params_input = nothing
    params_report = Dict{String,Any}("source" => "none")
    parameter_warnings = String[]
    if require_parameters || haskey(inputs, "parameters")
        session_params, session_report = _resolve_session_parameter_fallback(model)

        if haskey(inputs, "parameters")
            params_input, params_report, parameter_warnings = _resolve_parameters(
                inputs["parameters"],
                model;
                fallback_params = session_params,
                require_complete = require_parameters || !allow_partial_parameters,
            )
        elseif require_parameters
            if session_params === nothing
                throw(ArgumentError("inputs.parameters is required for this workflow"))
            end

            params_input = session_params
            params_report = session_report
            push!(parameter_warnings, "inputs.parameters omitted; reused parameters from same-session context")
        end
    end

    runtime_cfg, runtime_report = _resolve_runtime(get(inputs, "runtime", nothing))

    if haskey(runtime_cfg, "input_mapping")
        runtime_mapping = runtime_cfg["input_mapping"]
        runtime_mapping isa Dict{String,String} ||
            throw(ArgumentError("runtime.input_mapping must resolve to a string map"))
        request["options"]["input_mapping"] = runtime_mapping
    end

    warnings = String[]
    append!(warnings, forcing_warnings)
    append!(warnings, observation_warnings)
    append!(warnings, parameter_warnings)

    reports = Dict{String,Any}(
        "forcing" => forcing_report,
        "parameters" => params_report,
        "runtime" => runtime_report,
    )
    observation_report === nothing || (reports["observation"] = observation_report)

    metadata = Dict{String,Any}(
        "forcing" => forcing_meta,
        "runtime" => runtime_report,
    )
    observation_meta === nothing || (metadata["observation"] = observation_meta)

    return Dict{String,Any}(
        "request" => request,
        "forcing_nt" => forcing_nt,
        "observation" => obs_vec,
        "parameters" => params_input,
        "runtime" => runtime_cfg,
        "warnings" => warnings,
        "inference_report" => reports,
        "metadata" => metadata,
    )
end

export normalize_workflow_request, resolve_common_inputs, resolve_forcing, resolve_observation

end
