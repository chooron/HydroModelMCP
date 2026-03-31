using ModelContextProtocol

function _semantic_candidate_models(query::String, models::Vector{String})
    normalized = lowercase(strip(query))
    isempty(normalized) && return String[]

    generic_keywords = (
        "runoff", "rainfall", "hydrology", "hydrologic", "watershed", "streamflow",
        "precipitation", "basin", "model",
    )

    has_non_ascii = any(!isascii, normalized)
    has_generic_keyword = any(occursin(keyword, normalized) for keyword in generic_keywords)
    if !(has_non_ascii || has_generic_keyword)
        return String[]
    end

    preferred = ["exphydro", "gr4j", "hbv", "sacramento", "hymod", "xaj"]
    available = Set(models)
    return [name for name in preferred if name in available]
end

list_models_tool = MCPTool(
    name = "list_models",
    description = "List the hydrological models exposed by HydroModelMCP.",
    parameters = ToolParameter[],
    handler = function(_params)
        models = Discovery.list_models()
        payload = Dict(
            "status" => "success",
            "count" => length(models),
            "models" => models,
        )
        return create_json_response(payload)
    end,
)

find_model_tool = MCPTool(
    name = "find_model",
    description = "Resolve a hydrological model name and return ranked matches.",
    input_schema = Dict(
        "type" => "object",
        "properties" => Dict(
            "query" => Dict(
                "type" => "string",
                "description" => "Model name or partial model name.",
            ),
        ),
        "required" => ["query"],
    ),
    handler = function(params)
        validation_error = validate_required_params(params, ["query"])
        !isnothing(validation_error) && return create_error_response(validation_error)

        query = params["query"]
        models = Discovery.list_models()
        matches = Dict{String,Any}[]
        for model_name in models
            score = let
                canonical = Discovery.find_model(query)
                canonical == model_name ? 1.0 :
                    (occursin(lowercase(strip(query)), lowercase(model_name)) ? 0.75 : 0.0)
            end
            score > 0 || continue
            push!(matches, Dict(
                "name" => model_name,
                "full_name" => model_name,
                "match_score" => score,
            ))
        end

        if isempty(matches)
            fallback_models = _semantic_candidate_models(query, models)
            for (idx, model_name) in enumerate(fallback_models)
                push!(matches, Dict(
                    "name" => model_name,
                    "full_name" => model_name,
                    "match_score" => max(0.45, 0.7 - 0.05 * (idx - 1)),
                ))
            end
        end

        sort!(matches; by = item -> (-Float64(item["match_score"]), String(item["name"])))

        payload = Dict(
            "status" => "success",
            "query" => query,
            "matches" => matches,
        )
        return create_json_response(payload)
    end,
)

get_model_info_tool = MCPTool(
    name = "get_model_info",
    description = "Return basic information about a hydrological model.",
    input_schema = Dict(
        "type" => "object",
        "properties" => Dict(
            "model" => Dict(
                "type" => "string",
                "description" => "Model name.",
            ),
        ),
        "required" => ["model"],
    ),
    handler = function(params)
        validation_error = validate_required_params(params, ["model"])
        !isnothing(validation_error) && return create_error_response(validation_error)

        info = Discovery.get_model_info(params["model"])
        payload = Dict{String,Any}("status" => "success")
        merge!(payload, info)
        return create_json_response(payload)
    end,
)

get_model_variables_tool = MCPTool(
    name = "get_model_variables",
    description = "Return variable metadata for a hydrological model.",
    input_schema = Dict(
        "type" => "object",
        "properties" => Dict(
            "model" => Dict(
                "type" => "string",
                "description" => "Model name.",
            ),
        ),
        "required" => ["model"],
    ),
    handler = function(params)
        validation_error = validate_required_params(params, ["model"])
        !isnothing(validation_error) && return create_error_response(validation_error)

        variables = Discovery.get_variables_detail(params["model"])
        payload = Dict(
            "status" => "success",
            "model" => Discovery.find_model(params["model"]),
            "variables" => variables,
        )
        return create_json_response(payload)
    end,
)

get_model_parameters_tool = MCPTool(
    name = "get_model_parameters",
    description = "Return parameter metadata for a hydrological model.",
    input_schema = Dict(
        "type" => "object",
        "properties" => Dict(
            "model" => Dict(
                "type" => "string",
                "description" => "Model name.",
            ),
        ),
        "required" => ["model"],
    ),
    handler = function(params)
        validation_error = validate_required_params(params, ["model"])
        !isnothing(validation_error) && return create_error_response(validation_error)

        canonical = Discovery.find_model(params["model"])
        parameters = Discovery.get_parameters_detail(params["model"])
        payload = Dict(
            "status" => "success",
            "model" => canonical,
            "parameters" => parameters,
        )
        return create_json_response(payload)
    end,
)

export list_models_tool, find_model_tool, get_model_info_tool, get_model_variables_tool,
       get_model_parameters_tool
