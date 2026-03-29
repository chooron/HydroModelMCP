using ModelContextProtocol

const STORED_RESULT_SPECS = [
    Dict(
        "category" => "calibration",
        "index_uri" => "hydro://calibration/results",
        "item_uri_prefix" => "hydro://calibration/results",
        "index_name" => "Calibration Results List",
        "item_name" => "Calibration Result",
        "description" => "Stored calibration results available in the current backend.",
    ),
    Dict(
        "category" => "sensitivity",
        "index_uri" => "hydro://sensitivity/results",
        "item_uri_prefix" => "hydro://sensitivity/results",
        "index_name" => "Sensitivity Results List",
        "item_name" => "Sensitivity Result",
        "description" => "Stored sensitivity-analysis results available in the current backend.",
    ),
    Dict(
        "category" => "ensemble",
        "index_uri" => "hydro://ensemble/results",
        "item_uri_prefix" => "hydro://ensemble/results",
        "index_name" => "Ensemble Results List",
        "item_name" => "Ensemble Result",
        "description" => "Stored ensemble-run results available in the current backend.",
    ),
]

function _list_result_ids(storage_backend, category::String)
    return sort!(String.(Storage.list_results(storage_backend, category)))
end

function _create_result_index_resource(storage_backend, spec::Dict{String,String})
    category = spec["category"]
    MCPResource(
        uri = spec["index_uri"],
        name = spec["index_name"],
        title = spec["index_name"],
        description = spec["description"],
        mime_type = "application/json",
        data_provider = () -> begin
            result_ids = _list_result_ids(storage_backend, category)
            Dict(
                "category" => category,
                "results" => result_ids,
                "count" => length(result_ids),
                "uri_template" => spec["item_uri_prefix"] * "/{result_id}",
                "refresh_note" => "New stored result URIs appear after the server rebuilds its resource registry.",
            )
        end,
    )
end

function _create_result_resource(storage_backend, spec::Dict{String,String}, result_id::String)
    category = spec["category"]
    item_name = spec["item_name"]
    MCPResource(
        uri = spec["item_uri_prefix"] * "/" * result_id,
        name = item_name * " " * result_id,
        title = item_name,
        description = "Stored $category result '$result_id'.",
        mime_type = "application/json",
        data_provider = () -> Storage.load_result(storage_backend, category, result_id),
    )
end

function create_storage_resources(storage_backend)
    resources = MCPResource[]

    for spec in STORED_RESULT_SPECS
        push!(resources, _create_result_index_resource(storage_backend, spec))
        for result_id in _list_result_ids(storage_backend, spec["category"])
            push!(resources, _create_result_resource(storage_backend, spec, result_id))
        end
    end

    return resources
end

export STORED_RESULT_SPECS, create_storage_resources
