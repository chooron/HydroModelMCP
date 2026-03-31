using ModelContextProtocol

const WORKFLOW_SOURCE_SCHEMA = Dict(
    "type" => "object",
    "description" => "Unified source descriptor.",
    "properties" => Dict(
        "source_type" => Dict("type" => "string", "enum" => ["csv", "json", "redis", "camels", "npz", "data_handle", "calibration_result"]),
        "path" => Dict("type" => "string"),
        "data" => Dict("description" => "Inline object or JSON text string for json/calibration_result source."),
        "key" => Dict("type" => "string"),
        "host" => Dict("type" => "string"),
        "port" => Dict("type" => "integer"),
        "dataset_path" => Dict("type" => "string"),
        "gage_id" => Dict("type" => "integer"),
        "gauge_id" => Dict("type" => "integer"),
        "handle" => Dict("type" => "string"),
        "data_handle" => Dict("type" => "string"),
        "column" => Dict("type" => "string"),
        "row_index" => Dict("type" => "integer"),
    ),
    "required" => ["source_type"],
)

const WORKFLOW_INPUTS_SCHEMA = Dict(
    "type" => "object",
    "properties" => Dict(
        "forcing" => WORKFLOW_SOURCE_SCHEMA,
        "observation" => WORKFLOW_SOURCE_SCHEMA,
        "parameters" => WORKFLOW_SOURCE_SCHEMA,
        "runtime" => WORKFLOW_SOURCE_SCHEMA,
    ),
    "required" => ["forcing"],
)

const WORKFLOW_OUTPUT_SCHEMA = Dict(
    "type" => "object",
    "properties" => Dict(
        "result_source_type" => Dict("type" => "string", "enum" => ["csv", "json", "redis"]),
        "output_dir" => Dict("type" => "string"),
        "result_key" => Dict("type" => "string"),
        "result_host" => Dict("type" => "string"),
        "result_port" => Dict("type" => "integer"),
        "base_name" => Dict("type" => "string"),
    ),
)

const WORKFLOW_OPTIONS_SCHEMA = Dict(
    "type" => "object",
    "properties" => Dict(
        "auto_infer" => Dict("type" => "boolean", "default" => true),
        "strict_infer" => Dict("type" => "boolean", "default" => false),
        "input_mapping" => Dict("type" => "object"),
    ),
)

function _normalize_simulation_request(params::AbstractDict)
    request = UnifiedInputs.normalize_workflow_request(params)
    output_cfg = request["output"]

    if haskey(output_cfg, "output_dir")
        output_cfg["output_dir"] = resolve_workspace_path(string(output_cfg["output_dir"]))
    end

    forcing = request["inputs"]["forcing"]
    for role in ("forcing", "observation", "parameters", "runtime")
        haskey(request["inputs"], role) || continue
        spec = request["inputs"][role]
        if haskey(spec, "path")
            must_exist = true
            spec["path"] = resolve_workspace_path(string(spec["path"]); must_exist = must_exist)
        end
    end

    if forcing["source_type"] == "redis"
        # keep as-is
    elseif get(output_cfg, "result_source_type", "csv") == "redis"
        # keep as-is
    end

    return request
end

simulation_tool = MCPTool(
    name = "run_simulation",
    description = "Run a hydrological model using unified v2 protocol: model + inputs + output + options.",
    input_schema = Dict(
        "type" => "object",
        "properties" => Dict(
            "model" => Dict("type" => "string", "description" => "Model name or alias."),
            "inputs" => WORKFLOW_INPUTS_SCHEMA,
            "output" => WORKFLOW_OUTPUT_SCHEMA,
            "options" => WORKFLOW_OPTIONS_SCHEMA,
        ),
        "required" => ["model", "inputs"],
    ),
    handler = function(params)
        try
            request = _normalize_simulation_request(params)
            result = Simulation.run_simulation(request)
            store_last_result(LAST_SIMULATION_RESULT_HANDLE, result)
            return create_json_response(result)
        catch e
            return create_error_response(e)
        end
    end,
)

export simulation_tool, WORKFLOW_SOURCE_SCHEMA, WORKFLOW_INPUTS_SCHEMA, WORKFLOW_OUTPUT_SCHEMA, WORKFLOW_OPTIONS_SCHEMA
