import ModelContextProtocol: handle_request

const RESOURCE_TEMPLATE_PROVIDER_REGISTRY = IdDict{ModelContextProtocol.Server, Dict{String,Function}}()
const RESOURCE_TEMPLATE_MATCH_CACHE = Dict{String,Tuple{Regex,Vector{String}}}()

ModelContextProtocol.REQUEST_PARAMS_MAP["resources/templates/list"] = ModelContextProtocol.ListResourcesParams

function register_resource_template_providers!(server::ModelContextProtocol.Server, providers::Dict{String,Function})
    RESOURCE_TEMPLATE_PROVIDER_REGISTRY[server] = providers
    return server
end

function build_resource_template_providers(storage_backend)
    providers = Dict{String,Function}(
        "hydro://models/{model_name}/info" => args -> merge(
            Dict("status" => "success"),
            Discovery.get_model_info(args["model_name"]),
        ),
        "hydro://models/{model_name}/parameters" => args -> Dict(
            "status" => "success",
            "model" => Discovery.find_model(args["model_name"]),
            "parameters" => Discovery.get_parameters_detail(args["model_name"]),
        ),
        "hydro://models/{model_name}/variables" => args -> Dict(
            "status" => "success",
            "model" => Discovery.find_model(args["model_name"]),
            "variables" => Discovery.get_variables_detail(args["model_name"]),
        ),
        "hydro://models/{model_name}/knowledge" => args -> model_knowledge_payload(args["model_name"]),
        "hydro://hints/{feature}" => args -> llm_hint_payload(args["feature"]),
        "hydro://workflows/{intent}" => args -> workflow_playbook_payload(args["intent"]),
    )

    for spec in STORED_RESULT_SPECS
        uri_template = spec["item_uri_prefix"] * "/{result_id}"
        category = spec["category"]
        providers[uri_template] = args -> Storage.load_result(storage_backend, category, args["result_id"])
    end

    return providers
end

function _compile_uri_template(uri_template::String)
    names = String[]
    pattern = IOBuffer()
    print(pattern, '^')

    i = firstindex(uri_template)
    while i <= lastindex(uri_template)
        ch = uri_template[i]
        if ch == '{'
            close_idx = findnext('}', uri_template, i)
            isnothing(close_idx) && throw(ArgumentError("Invalid URI template: missing closing brace in '$uri_template'"))
            name = uri_template[nextind(uri_template, i):prevind(uri_template, close_idx)]
            push!(names, name)
            print(pattern, "(?<", name, ">[^/?#]+)")
            i = nextind(uri_template, close_idx)
            continue
        end

        if occursin(string(ch), raw"[.^$|()\[\]{}*+?\\]")
            print(pattern, '\\')
        end
        print(pattern, ch)
        i = nextind(uri_template, i)
    end

    print(pattern, '\$')
    return Regex(String(take!(pattern))), names
end

function _match_uri_template(uri_template::String, uri::String)
    regex, names = get!(RESOURCE_TEMPLATE_MATCH_CACHE, uri_template) do
        _compile_uri_template(uri_template)
    end

    matched = match(regex, uri)
    isnothing(matched) && return nothing

    return Dict{String,String}(name => matched[name] for name in names)
end

function _resource_template_to_dict(template::ModelContextProtocol.ResourceTemplate)
    payload = Dict{String,Any}(
        "uriTemplate" => template.uri_template,
        "name" => template.name,
        "description" => template.description,
    )
    !isnothing(template.mime_type) && (payload["mimeType"] = template.mime_type)
    !isnothing(template.title) && (payload["title"] = template.title)
    !isnothing(template.icons) && (payload["icons"] = [ModelContextProtocol.icon_to_dict(icon) for icon in template.icons])
    return payload
end

function _handle_list_resource_templates(server::ModelContextProtocol.Server, request::ModelContextProtocol.JSONRPCRequest)
    params = isnothing(request.params) ? ModelContextProtocol.ListResourcesParams() : request.params::ModelContextProtocol.ListResourcesParams

    result = Dict{String,Any}(
        "resourceTemplates" => [_resource_template_to_dict(template) for template in server.resource_templates],
    )

    if !isnothing(params.cursor) && params.cursor != ""
        result["nextCursor"] = params.cursor
    end

    return ModelContextProtocol.JSONRPCResponse(
        id = request.id,
        result = result,
    )
end

function _handle_read_template_resource(server::ModelContextProtocol.Server, request::ModelContextProtocol.JSONRPCRequest)
    params = request.params::ModelContextProtocol.ReadResourceParams
    providers = get(RESOURCE_TEMPLATE_PROVIDER_REGISTRY, server, Dict{String,Function}())

    for template in server.resource_templates
        template_args = _match_uri_template(template.uri_template, params.uri)
        isnothing(template_args) && continue

        provider = get(providers, template.uri_template, nothing)
        isnothing(provider) && continue

        try
            data = provider(template_args)
            text_payload = data isa AbstractString ? String(data) : JSON3.write(data)
            default_mime = data isa AbstractString ? "text/plain" : "application/json"
            contents = [Dict{String,Any}(
                "uri" => params.uri,
                "text" => text_payload,
                "mimeType" => something(template.mime_type, default_mime),
            )]

            return ModelContextProtocol.JSONRPCResponse(
                id = request.id,
                result = ModelContextProtocol.ReadResourceResult(contents = contents),
            )
        catch err
            message = sprint(showerror, err)
            code = occursin("not found", lowercase(message)) ?
                ModelContextProtocol.ErrorCodes.RESOURCE_NOT_FOUND :
                ModelContextProtocol.ErrorCodes.INTERNAL_ERROR

            return ModelContextProtocol.JSONRPCError(
                id = request.id,
                error = ModelContextProtocol.ErrorInfo(
                    code = code,
                    message = code == ModelContextProtocol.ErrorCodes.RESOURCE_NOT_FOUND ?
                        "Resource not found: $(params.uri)" :
                        "Error reading resource: $(message)",
                ),
            )
        end
    end

    return nothing
end

function handle_request(server::ModelContextProtocol.Server, request::ModelContextProtocol.JSONRPCRequest)::ModelContextProtocol.Response
    if request.method == "resources/templates/list"
        return _handle_list_resource_templates(server, request)
    elseif request.method == "resources/read"
        response = _handle_read_template_resource(server, request)
        !isnothing(response) && return response
    end

    return invoke(
        ModelContextProtocol.handle_request,
        Tuple{ModelContextProtocol.Server, ModelContextProtocol.Request},
        server,
        request,
    )
end

export build_resource_template_providers, register_resource_template_providers!
