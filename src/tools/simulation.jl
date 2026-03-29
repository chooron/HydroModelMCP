using ModelContextProtocol

function _normalize_simulation_request(params::AbstractDict)
    args = normalize_string_dict(params)

    validation_error = validate_required_params(args, ["model", "source_type"])
    !isnothing(validation_error) && throw(ArgumentError(validation_error))

    source_type = lowercase(string(args["source_type"]))
    source_type in ("csv", "json") ||
        throw(ArgumentError("source_type must be one of: csv, json"))

    normalized = Dict{String,Any}(
        "model" => string(args["model"]),
        "source_type" => source_type,
        "solver" => optional_string(get(args, "solver", "DISCRETE"), "DISCRETE"),
        "interpolation" => optional_string(get(args, "interpolation", "LINEAR"), "LINEAR"),
        "output_dir" => resolve_workspace_path(get(args, "output_dir", "./result")),
    )

    if source_type == "csv"
        path_error = validate_required_params(args, ["path"])
        !isnothing(path_error) && throw(ArgumentError(path_error))
        normalized["path"] = resolve_workspace_path(string(args["path"]); must_exist = true)
    else
        if haskey(args, "path")
            normalized["path"] = resolve_workspace_path(string(args["path"]); must_exist = true)
        elseif haskey(args, "data")
            normalized["data"] = args["data"]
        else
            throw(ArgumentError("JSON simulation requires either 'path' or 'data'"))
        end
    end

    for optional_key in ("params", "period", "warmup", "seed", "input_mapping", "init_states")
        haskey(args, optional_key) && (normalized[optional_key] = args[optional_key])
    end

    return normalized
end

simulation_tool = MCPTool(
    name = "run_simulation",
    description = "Run a hydrological model from a workspace file or JSON payload and write results into the result workspace.",
    input_schema = Dict(
        "type" => "object",
        "properties" => Dict(
            "model" => Dict("type" => "string", "description" => "Model name or alias."),
            "source_type" => Dict(
                "type" => "string",
                "enum" => ["csv", "json"],
                "description" => "Input data format.",
            ),
            "path" => Dict("type" => "string", "description" => "Relative or absolute path to the forcing file."),
            "data" => Dict("type" => "object", "description" => "Inline JSON forcing payload."),
            "params" => Dict("type" => "object", "description" => "Optional parameter values keyed by parameter name."),
            "period" => Dict(
                "type" => "array",
                "items" => Dict("type" => "string"),
                "description" => "Optional period metadata recorded with the run.",
            ),
            "warmup" => Dict("type" => "integer", "description" => "Optional warmup metadata recorded with the run."),
            "solver" => Dict(
                "type" => "string",
                "enum" => ["ODE", "DISCRETE", "MUTABLE", "IMMUTABLE"],
                "default" => "DISCRETE",
            ),
            "interpolation" => Dict(
                "type" => "string",
                "enum" => ["LINEAR", "CONSTANT", "DIRECT"],
                "default" => "LINEAR",
            ),
            "input_mapping" => Dict(
                "type" => "object",
                "description" => "Optional model-input to source-column mapping.",
            ),
            "output_dir" => Dict(
                "type" => "string",
                "description" => "Workspace-relative output directory. Defaults to ./result.",
                "default" => "./result",
            ),
            "seed" => Dict("type" => "integer", "description" => "Seed used when random parameters are generated."),
            "init_states" => Dict("type" => "object", "description" => "Optional initial state values."),
        ),
        "required" => ["model", "source_type"],
    ),
    handler = function(params)
        try
            request = _normalize_simulation_request(params)
            result = Simulation.run_simulation(request)
            return create_json_response(result)
        catch e
            return create_error_response(e)
        end
    end,
)

export simulation_tool
