"""
Validation tool using unified v2 protocol.
"""

using ModelContextProtocol
using JSON3
using .Schemas

validation_tool = MCPTool(
    name = "run_validation",
    description = "Run validation on calibration/validation periods using unified v2 protocol. inputs.parameters supports json/csv/data_handle/calibration_result sources, inline parameter objects, or same-session fallback from the latest calibration/simulation when omitted. Periods accept start_index/end_index or start/end dates, with synthetic timeline fallback when forcing dates are unavailable.",
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
        "required" => ["model", "inputs", "calibration_period", "validation_period"],
    ),
    handler = function(args::AbstractDict)
        validation_error = validate_required_params(
            args,
            ["model", "inputs", "calibration_period", "validation_period"],
        )
        !isnothing(validation_error) && return create_error_response(validation_error)

        try
            request = UnifiedInputs.normalize_workflow_request(args)
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

            result = Validation.run_validation(request)
            return TextContent(text = JSON3.write(result))
        catch e
            return create_error_response(e)
        end
    end,
)

export validation_tool
