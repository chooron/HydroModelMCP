using ModelContextProtocol

const STORAGE_TOOL_CATEGORIES = ["calibration", "sensitivity", "ensemble"]

function _normalize_storage_category(value)
    category = lowercase(strip(string(value)))
    category in STORAGE_TOOL_CATEGORIES || throw(ArgumentError(
        "category must be one of: $(join(STORAGE_TOOL_CATEGORIES, ", "))",
    ))
    return category
end

function _list_storage_payload_for_category(category::String)
    ids = sort!(String.(Storage.list_results(STORAGE_BACKEND, category)))
    return Dict{String,Any}(
        "category" => category,
        "count" => length(ids),
        "result_ids" => ids,
    )
end

list_stored_results_tool = MCPTool(
    name = "list_stored_results",
    description = "List persisted result IDs from storage backend by category or across all categories.",
    input_schema = Dict{String,Any}(
        "type" => "object",
        "properties" => Dict{String,Any}(
            "category" => STORAGE_CATEGORY_SCHEMA,
        ),
        "required" => [],
    ),
    handler = function(params)
        try
            if haskey(params, "category")
                category = _normalize_storage_category(params["category"])
                payload = _list_storage_payload_for_category(category)
                payload["status"] = "success"
                return create_json_response(payload)
            end

            categories_payload = Dict{String,Any}[]
            total = 0
            for category in STORAGE_TOOL_CATEGORIES
                item = _list_storage_payload_for_category(category)
                total += item["count"]
                push!(categories_payload, item)
            end

            return create_json_response(Dict{String,Any}(
                "status" => "success",
                "total_count" => total,
                "categories" => categories_payload,
            ))
        catch e
            return create_error_response(e)
        end
    end,
)

get_stored_result_tool = MCPTool(
    name = "get_stored_result",
    description = "Fetch one persisted result by result_id from storage backend (optionally scoped by category).",
    input_schema = Dict{String,Any}(
        "type" => "object",
        "properties" => Dict{String,Any}(
            "result_id" => RESULT_ID_SCHEMA,
            "category" => STORAGE_CATEGORY_SCHEMA,
        ),
        "required" => ["result_id"],
    ),
    handler = function(params)
        validation_error = validate_required_params(params, ["result_id"])
        !isnothing(validation_error) && return create_error_response(validation_error)

        try
            result_id = string(params["result_id"])

            if haskey(params, "category")
                category = _normalize_storage_category(params["category"])
                payload = Storage.load_result(STORAGE_BACKEND, category, result_id)
                payload isa AbstractDict || throw(ArgumentError("Stored payload is not an object"))
                normalized = Dict{String,Any}(string(k) => v for (k, v) in pairs(payload))
                normalized["status"] = "success"
                normalized["storage_category"] = category
                normalized["result_id"] = result_id
                return create_json_response(normalized)
            end

            for category in STORAGE_TOOL_CATEGORIES
                result_ids = String.(Storage.list_results(STORAGE_BACKEND, category))
                result_id in result_ids || continue
                payload = Storage.load_result(STORAGE_BACKEND, category, result_id)
                payload isa AbstractDict || throw(ArgumentError("Stored payload is not an object"))
                normalized = Dict{String,Any}(string(k) => v for (k, v) in pairs(payload))
                normalized["status"] = "success"
                normalized["storage_category"] = category
                normalized["result_id"] = result_id
                return create_json_response(normalized)
            end

            return create_error_response("Result not found: $result_id")
        catch e
            return create_error_response(e)
        end
    end,
)

export list_stored_results_tool, get_stored_result_tool
