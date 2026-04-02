using ModelContextProtocol

const WORKFLOW_CARAVAN_GAUGE_ID_SCHEMA = Dict(
    "oneOf" => [
        Dict("type" => "integer"),
        Dict("type" => "string"),
    ],
    "description" => "Caravan gauge/basin id. Prefer the full global id like camels_01013500; local ids such as 01013500 also work when dataset_name/source_dataset is provided.",
)

const WORKFLOW_CARAVAN_DATASET_NAME_SCHEMA = Dict(
    "type" => "string",
    "enum" => ["camels", "camelsaus", "camelsbr", "camelscl", "camelsgb", "hysets", "lamah"],
    "description" => "Caravan source dataset name.",
)

const WORKFLOW_SOURCE_SCHEMA = Dict(
    "type" => "object",
    "description" => "Unified source descriptor. calibration_result sources may provide data/path/handle directly, or aliases such as calibration_result/result/payload/result_id. For source_type=caravan, dataset_name/source_dataset is required together with gauge_id/gage_id/basin_id, and missing gauges must fail directly instead of falling back to local CSV guesses.",
    "properties" => Dict(
        "source_type" => Dict("type" => "string", "enum" => ["csv", "json", "redis", "caravan", "npz", "data_handle", "calibration_result"]),
        "path" => Dict("type" => "string"),
        "data" => Dict("description" => "Inline object or JSON text string for json/calibration_result source."),
        "calibration_result" => Dict("description" => "Alias for data when source_type=calibration_result."),
        "result" => Dict("description" => "Alias for data when source_type=calibration_result."),
        "payload" => Dict("description" => "Alias for data when source_type=calibration_result."),
        "key" => Dict("type" => "string"),
        "host" => Dict("type" => "string"),
        "port" => Dict("type" => "integer"),
        "dataset_path" => Dict("type" => "string", "description" => "Optional alias of Caravan dataset_root."),
        "dataset_root" => Dict("type" => "string", "description" => "Optional Caravan dataset root containing attributes/ and timeseries/."),
        "timeseries_root" => Dict("type" => "string", "description" => "Optional Caravan timeseries root."),
        "netcdf_root" => Dict("type" => "string", "description" => "Optional Caravan netCDF root; omit it to use CARAVAN_DATASET_ROOT/CARAVAN_NETCDF_ROOT when available."),
        "dataset_name" => WORKFLOW_CARAVAN_DATASET_NAME_SCHEMA,
        "source_dataset" => WORKFLOW_CARAVAN_DATASET_NAME_SCHEMA,
        "gage_id" => WORKFLOW_CARAVAN_GAUGE_ID_SCHEMA,
        "gauge_id" => WORKFLOW_CARAVAN_GAUGE_ID_SCHEMA,
        "basin_id" => WORKFLOW_CARAVAN_GAUGE_ID_SCHEMA,
        "handle" => Dict("type" => "string"),
        "data_handle" => Dict("type" => "string"),
        "result_id" => Dict("type" => "string"),
        "category" => Dict("type" => "string"),
        "storage_category" => Dict("type" => "string"),
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
