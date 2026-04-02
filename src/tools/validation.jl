"""
Validation tool using unified v2 protocol.
"""

using ModelContextProtocol
using JSON3
using .Schemas

function _same_session_validation_periods(last_calibration)
    last_calibration isa AbstractDict || return nothing

    if haskey(last_calibration, "calibration_period") && haskey(last_calibration, "validation_period")
        return Dict{String,Any}(
            "calibration_period" => last_calibration["calibration_period"],
            "validation_period" => last_calibration["validation_period"],
        )
    end

    stage2 = get(last_calibration, "stage2_evaluation", nothing)
    stage2 isa AbstractDict || return nothing

    train_indices = get(stage2, "train_indices", nothing)
    test_indices = get(stage2, "test_indices", nothing)
    train_indices isa AbstractVector || return nothing
    test_indices isa AbstractVector || return nothing
    length(train_indices) >= 2 || return nothing
    length(test_indices) >= 2 || return nothing

    return Dict{String,Any}(
        "calibration_period" => Dict(
            "start_index" => Int(train_indices[1]),
            "end_index" => Int(train_indices[end]),
        ),
        "validation_period" => Dict(
            "start_index" => Int(test_indices[1]),
            "end_index" => Int(test_indices[end]),
        ),
    )
end

function _load_stored_calibration_result(result_id::String, category::Union{Nothing,String})
    if !(category === nothing)
        return Storage.load_result(STORAGE_BACKEND, category, result_id)
    end

    for candidate in ("calibration",)
        result_id in String.(Storage.list_results(STORAGE_BACKEND, candidate)) || continue
        return Storage.load_result(STORAGE_BACKEND, candidate, result_id)
    end

    throw(ArgumentError("Stored calibration result not found: $result_id"))
end

function _materialize_validation_parameter_aliases(args::AbstractDict)
    request = normalize_string_dict(args)
    haskey(request, "inputs") || return request, String[]

    raw_inputs = request["inputs"]
    raw_inputs isa AbstractDict || return request, String[]
    inputs = normalize_string_dict(raw_inputs)
    request["inputs"] = inputs
    haskey(inputs, "parameters") || return request, String[]

    raw_parameters = inputs["parameters"]
    raw_parameters isa AbstractDict || return request, String[]
    parameters = normalize_string_dict(raw_parameters)
    lowercase(string(get(parameters, "source_type", ""))) == "calibration_result" || return request, String[]

    warnings = String[]
    if !haskey(parameters, "data")
        for alias_key in ("calibration_result", "result", "payload")
            haskey(parameters, alias_key) || continue
            parameters["data"] = parameters[alias_key]
            push!(warnings, "inputs.parameters.$alias_key was normalized to calibration_result.data")
            break
        end
    end

    if !haskey(parameters, "data") && haskey(parameters, "result_id")
        result_id = String(strip(string(parameters["result_id"])))
        isempty(result_id) && throw(ArgumentError("inputs.parameters.result_id cannot be empty"))

        category = if haskey(parameters, "storage_category")
            String(lowercase(strip(string(parameters["storage_category"]))))
        elseif haskey(parameters, "category")
            String(lowercase(strip(string(parameters["category"]))))
        else
            nothing
        end

        parameters["data"] = _load_stored_calibration_result(result_id, category)
        push!(warnings, "inputs.parameters.result_id resolved from stored calibration result")
    end

    inputs["parameters"] = parameters
    return request, warnings
end

function _validation_period_fallback_from_parameters(request::Dict{String,Any})
    inputs = get(request, "inputs", nothing)
    inputs isa AbstractDict || return nothing
    haskey(inputs, "parameters") || return nothing

    parameters = inputs["parameters"]
    parameters isa AbstractDict || return nothing
    lowercase(string(get(parameters, "source_type", ""))) == "calibration_result" || return nothing
    haskey(parameters, "data") || return nothing

    raw_payload = parameters["data"]
    if raw_payload isa AbstractString
        raw_payload = JSON3.read(String(raw_payload))
    end

    raw_payload isa AbstractDict || return nothing
    payload = Dict{String,Any}(string(k) => v for (k, v) in pairs(raw_payload))
    return _same_session_validation_periods(payload)
end

function _inject_validation_period_fallback!(request::Dict{String,Any})
    has_cal = haskey(request, "calibration_period")
    has_val = haskey(request, "validation_period")
    has_cal == has_val || throw(ArgumentError(
        "calibration_period and validation_period must be provided together",
    ))
    has_cal && return String[]

    fallback = _validation_period_fallback_from_parameters(request)
    if !(fallback === nothing)
        request["calibration_period"] = fallback["calibration_period"]
        request["validation_period"] = fallback["validation_period"]
        return ["calibration_period and validation_period omitted; reused split embedded in inputs.parameters calibration_result"]
    end

    last_calibration = load_last_result(LAST_CALIBRATION_RESULT_HANDLE)
    fallback = _same_session_validation_periods(last_calibration)
    fallback === nothing && throw(ArgumentError(
        "calibration_period and validation_period are required unless a same-session calibrate_model result provides reusable split indices",
    ))

    request["calibration_period"] = fallback["calibration_period"]
    request["validation_period"] = fallback["validation_period"]
    return ["calibration_period and validation_period omitted; reused same-session split from last calibrate_model result"]
end

validation_tool = MCPTool(
    name = "run_validation",
    description = "Run validation using unified v2 protocol. inputs.parameters supports json/csv/data_handle/calibration_result sources, inline parameter objects, or same-session fallback from the latest calibration/simulation when omitted. calibration_result sources may use data/path/handle directly, aliases such as calibration_result/result/payload, or a stored result_id with optional category/storage_category. calibration_period and validation_period may be provided explicitly, or omitted together to reuse the split embedded in calibration_result or the most recent same-session calibration. Periods accept start_index/end_index or start/end dates, with synthetic timeline fallback when forcing dates are unavailable.",
    input_schema = Dict{String,Any}(
        "type" => "object",
        "properties" => Dict{String,Any}(
            "model" => MODEL_NAME_SCHEMA,
            "inputs" => WORKFLOW_INPUTS_SCHEMA,
            "output" => WORKFLOW_OUTPUT_SCHEMA,
            "options" => WORKFLOW_OPTIONS_SCHEMA,
            "calibration_period" => CALIBRATION_PERIOD_SCHEMA,
            "validation_period" => VALIDATION_PERIOD_SCHEMA,
            "metrics" => METRICS_SCHEMA,
        ),
        "required" => ["model", "inputs"],
    ),
    handler = function(args::AbstractDict)
        validation_error = validate_required_params(
            args,
            ["model", "inputs"],
        )
        !isnothing(validation_error) && return create_error_response(validation_error)

        try
            prepared_args, alias_warnings = _materialize_validation_parameter_aliases(args)
            request = UnifiedInputs.normalize_workflow_request(prepared_args)
            if haskey(request, "output") && haskey(request["output"], "output_dir")
                request["output"]["output_dir"] = resolve_workspace_path(string(request["output"]["output_dir"]))
            end

            for role in ("forcing", "observation", "parameters", "runtime")
                haskey(request["inputs"], role) || continue
                spec = request["inputs"][role]
                if haskey(spec, "path")
                    spec["path"] = resolve_workspace_path(string(spec["path"]); must_exist = true)
                end
            end

            fallback_warnings = _inject_validation_period_fallback!(request)

            result = Validation.run_validation(request)
            merged_warnings = String[]
            append!(merged_warnings, alias_warnings)
            append!(merged_warnings, fallback_warnings)
            if !isempty(merged_warnings)
                warnings = haskey(result, "warnings") ? String.(result["warnings"]) : String[]
                append!(warnings, merged_warnings)
                result["warnings"] = warnings
            end
            return TextContent(text = JSON3.write(result))
        catch e
            return create_error_response(e)
        end
    end,
)

export validation_tool
