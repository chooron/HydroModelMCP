"""
Ensemble tool using unified v2 protocol.
"""

using ModelContextProtocol
using JSON3
using .Schemas

ensemble_parameter_tool = MCPTool(
    name = "run_ensemble_parameters",
    description = "Run ensemble simulation using unified v2 protocol and parameter_sets.",
    input_schema = Dict{String,Any}(
        "type" => "object",
        "properties" => Dict{String,Any}(
            "model" => MODEL_NAME_SCHEMA,
            "inputs" => WORKFLOW_INPUTS_SCHEMA,
            "output" => WORKFLOW_OUTPUT_SCHEMA,
            "options" => WORKFLOW_OPTIONS_SCHEMA,
            "parameter_sets" => PARAMETER_SETS_SCHEMA,
            "parallel" => PARALLEL_SCHEMA,
            "save_to_storage" => SAVE_TO_STORAGE_SCHEMA,
        ),
        "required" => ["model", "inputs", "parameter_sets"],
    ),
    handler = function(args::AbstractDict)
        validation_error = validate_required_params(args, ["model", "inputs", "parameter_sets"])
        !isnothing(validation_error) && return create_error_response(validation_error)

        if !(args["parameter_sets"] isa AbstractVector)
            return create_error_response("parameter_sets must be an array")
        end
        isempty(args["parameter_sets"]) && return create_error_response("parameter_sets cannot be empty")

        try
            request = UnifiedInputs.normalize_workflow_request(args)

            for role in ("forcing", "observation", "parameters", "runtime")
                haskey(request["inputs"], role) || continue
                spec = request["inputs"][role]
                if haskey(spec, "path")
                    spec["path"] = resolve_workspace_path(string(spec["path"]); must_exist = true)
                end
            end

            if haskey(request, "output") && haskey(request["output"], "output_dir")
                request["output"]["output_dir"] = resolve_workspace_path(string(request["output"]["output_dir"]))
            end

            result = Ensemble.run_ensemble(request)

            if get(args, "save_to_storage", true)
                Storage.save_result(
                    STORAGE_BACKEND,
                    "ensemble",
                    result["result_id"],
                    result,
                )
            end

            return TextContent(text = JSON3.write(result))
        catch e
            return create_error_response(e)
        end
    end,
)

export ensemble_parameter_tool
