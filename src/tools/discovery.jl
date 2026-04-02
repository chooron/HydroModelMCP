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

function _rank_model_matches(query::String, models::Vector{String})
    normalized = lowercase(strip(query))
    isempty(normalized) && return Dict{String,Any}[]

    canonical = Discovery.find_model(query)
    matches = Dict{String,Any}[]

    if !isnothing(canonical)
        push!(matches, Dict(
            "name" => canonical,
            "full_name" => canonical,
            "match_score" => 1.0,
            "match_reason" => "best_match",
        ))
    end

    for model_name in models
        !isnothing(canonical) && model_name == canonical && continue

        candidate = lowercase(model_name)
        score, reason = if startswith(candidate, normalized)
            0.85, "prefix"
        elseif occursin(normalized, candidate)
            0.7, "substring"
        else
            0.0, ""
        end

        score > 0 || continue
        push!(matches, Dict(
            "name" => model_name,
            "full_name" => model_name,
            "match_score" => score,
            "match_reason" => reason,
        ))
    end

    sort!(matches; by = item -> (-Float64(item["match_score"]), String(item["name"])))
    return matches
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
            "top_k" => Dict(
                "type" => "integer",
                "default" => 5,
                "description" => "Maximum number of ranked matches returned.",
            ),
        ),
        "required" => ["query"],
    ),
    handler = function(params)
        validation_error = validate_required_params(params, ["query"])
        !isnothing(validation_error) && return create_error_response(validation_error)

        query = params["query"]
        models = Discovery.list_models()
        matches = _rank_model_matches(query, models)

        if isempty(matches)
            fallback_models = _semantic_candidate_models(query, models)
            for (idx, model_name) in enumerate(fallback_models)
                push!(matches, Dict(
                    "name" => model_name,
                    "full_name" => model_name,
                    "match_score" => max(0.45, 0.7 - 0.05 * (idx - 1)),
                    "match_reason" => "semantic_fallback",
                ))
            end
        end

        top_k = max(1, Int(get(params, "top_k", 5)))
        top_matches = matches[1:min(end, top_k)]
        resolved = isempty(top_matches) ? nothing : top_matches[1]["name"]

        payload = Dict(
            "status" => "success",
            "query" => query,
            "resolved_model" => resolved,
            "matches" => top_matches,
            "next_tools" => ["get_model_info", "get_model_parameters", "get_model_variables"],
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
