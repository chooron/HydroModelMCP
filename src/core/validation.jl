"""
Validation workflow built on unified input protocol.
"""
module Validation

using ..Simulation
using ..Metrics
using ..UnifiedInputs
using Dates

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

function _slice_forcing(forcing_nt::NamedTuple, indices)
    names = keys(forcing_nt)
    values = Tuple([forcing_nt[name][indices] for name in names])
    return NamedTuple{names}(values)
end

function _parse_date_like(value, field::String)
    if value isa Date
        return value
    elseif value isa DateTime
        return Date(value)
    end

    text = strip(string(value))
    isempty(text) && throw(ArgumentError("$field cannot be empty"))

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

    throw(ArgumentError("$field must be parseable as date, for example 2020-01-01"))
end

function _build_synthetic_dates(calibration_period::AbstractDict, validation_period::AbstractDict, length_hint::Int)
    cal_start = _parse_date_like(calibration_period["start"], "calibration.start")
    cal_end = _parse_date_like(calibration_period["end"], "calibration.end")
    val_start = _parse_date_like(validation_period["start"], "validation.start")
    val_end = _parse_date_like(validation_period["end"], "validation.end")

    cal_start <= cal_end || throw(ArgumentError("calibration.start must be earlier than or equal to calibration.end"))
    val_start <= val_end || throw(ArgumentError("validation.start must be earlier than or equal to validation.end"))

    global_start = minimum((cal_start, val_start))
    global_end = maximum((cal_end, val_end))
    horizon_days = max(Dates.value(global_end - global_start), 1)

    length_hint >= 1 || throw(ArgumentError("length_hint must be >= 1"))
    if length_hint == 1
        return [global_start]
    end

    return [
        global_start + Day(round(Int, (idx - 1) * horizon_days / (length_hint - 1)))
        for idx in 1:length_hint
    ]
end

function _period_indices(period::AbstractDict, dates::Union{Nothing,Vector{Date}}, length_hint::Int, role::String)
    if haskey(period, "start_index") && haskey(period, "end_index")
        start_idx = Int(period["start_index"])
        end_idx = Int(period["end_index"])
        1 <= start_idx <= end_idx <= length_hint ||
            throw(ArgumentError("$role period indices out of bounds: [$start_idx, $end_idx], length=$length_hint"))
        return start_idx:end_idx
    end

    haskey(period, "start") && haskey(period, "end") ||
        throw(ArgumentError("$role period must include either start/end or start_index/end_index"))

    if isnothing(dates)
        start_idx = try
            Int(period["start"])
        catch
            nothing
        end
        end_idx = try
            Int(period["end"])
        catch
            nothing
        end

        if !(start_idx === nothing || end_idx === nothing)
            1 <= start_idx <= end_idx <= length_hint ||
                throw(ArgumentError("$role period indices out of bounds: [$start_idx, $end_idx], length=$length_hint"))
            return start_idx:end_idx
        end

        throw(ArgumentError(
            "$role period uses start/end, but forcing dates are unavailable. Provide start_index/end_index, or use a forcing source with a parseable date column (date/datetime/timestamp/time).",
        ))
    end

    start_date = _parse_date_like(period["start"], "$role.start")
    end_date = _parse_date_like(period["end"], "$role.end")
    start_date <= end_date || throw(ArgumentError("$role.start must be earlier than or equal to $role.end"))

    idx = findall(d -> start_date <= d <= end_date, dates)
    isempty(idx) && throw(ArgumentError("$role period selected no samples"))
    return idx
end

function run_validation(args::AbstractDict)
    request = UnifiedInputs.normalize_workflow_request(args)
    haskey(request, "calibration_period") || throw(ArgumentError("calibration_period is required"))
    haskey(request, "validation_period") || throw(ArgumentError("validation_period is required"))

    model_name = request["model"]
    canonical_model_name, _, model = Simulation._load_model(model_name)

    resolved = UnifiedInputs.resolve_common_inputs(request, model;
        require_observation = true,
        require_parameters = true,
    )

    forcing_nt = resolved["forcing_nt"]
    obs_vec = Float64.(resolved["observation"])
    params_input = resolved["parameters"]

    params_vec = Simulation._component_vector_from_params(model, params_input)
    params_used = Dict{String,Float64}()
    for pname in keys(params_vec.params)
        params_used[string(pname)] = Float64(params_vec.params[pname])
    end

    runtime_cfg = resolved["runtime"]
    solver_sym = Symbol(uppercase(string(get(runtime_cfg, "solver", "DISCRETE"))))
    interp_sym = Symbol(uppercase(string(get(runtime_cfg, "interpolation", "LINEAR"))))
    init_states = Simulation._build_init_states(model, get(runtime_cfg, "init_states", nothing))
    config_dict = haskey(runtime_cfg, "config") && runtime_cfg["config"] isa AbstractDict ?
        Dict{String,Any}(string(k) => v for (k, v) in pairs(runtime_cfg["config"])) : Dict{String,Any}()

    forcing_meta = get(resolved["metadata"], "forcing", nothing)
    dates_raw = if forcing_meta isa NamedTuple
        haskey(forcing_meta, :dates) ? forcing_meta[:dates] : nothing
    elseif forcing_meta isa AbstractDict
        get(forcing_meta, "dates", get(forcing_meta, :dates, nothing))
    else
        nothing
    end
    dates = isnothing(dates_raw) ? nothing : Date.(dates_raw)

    n_total = minimum((length(obs_vec), minimum([length(forcing_nt[k]) for k in keys(forcing_nt)])))
    obs_vec = obs_vec[1:n_total]
    forcing_nt = _slice_forcing(forcing_nt, 1:n_total)
    !isnothing(dates) && (dates = dates[1:n_total])

    cal_period = request["calibration_period"]
    val_period = request["validation_period"]

    validation_warnings = String[]
    if isnothing(dates) &&
       haskey(cal_period, "start") && haskey(cal_period, "end") &&
       haskey(val_period, "start") && haskey(val_period, "end")
        dates = _build_synthetic_dates(cal_period, val_period, n_total)
        push!(validation_warnings,
            "Forcing source has no date column; approximated calendar periods using synthetic timeline derived from requested start/end bounds")
    end

    cal_idx = _period_indices(cal_period, dates, n_total, "calibration")
    val_idx = _period_indices(val_period, dates, n_total, "validation")

    cal_forcing = _slice_forcing(forcing_nt, cal_idx)
    val_forcing = _slice_forcing(forcing_nt, val_idx)
    cal_obs = obs_vec[cal_idx]
    val_obs = obs_vec[val_idx]

    cal_sim = Simulation._execute_core(
        model,
        cal_forcing,
        params_vec,
        init_states,
        solver_sym,
        interp_sym,
        config_dict,
    )

    val_sim = Simulation._execute_core(
        model,
        val_forcing,
        params_vec,
        init_states,
        solver_sym,
        interp_sym,
        config_dict,
    )

    metric_names = haskey(request, "metrics") ? String.(request["metrics"]) : ["NSE", "KGE", "RMSE"]
    cal_metrics = Metrics.compute_metrics(cal_sim, cal_obs, metric_names)
    val_metrics = Metrics.compute_metrics(val_sim, val_obs, metric_names)

    warnings = String.(resolved["warnings"])
    append!(warnings, validation_warnings)

    return Dict{String,Any}(
        "status" => "success",
        "model" => canonical_model_name,
        "parameters" => params_used,
        "parameter_source" => resolved["inference_report"]["parameters"],
        "calibration_metrics" => cal_metrics,
        "validation_metrics" => val_metrics,
        "calibration_period" => cal_period,
        "validation_period" => val_period,
        "warnings" => warnings,
        "inference_report" => resolved["inference_report"],
    )
end

end
