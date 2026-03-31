using ModelContextProtocol

list_mcp_surfaces_tool = MCPTool(
    name = "list_mcp_surfaces",
    description = "List tools/resources/templates/prompts as a fallback when clients cannot directly call protocol methods like resources/templates/list or prompts/list.",
    input_schema = Dict(
        "type" => "object",
        "properties" => Dict(
            "include_descriptions" => Dict(
                "type" => "boolean",
                "default" => false,
                "description" => "Include descriptions/titles for each surface item.",
            ),
        ),
    ),
    handler = function(params)
        try
            include_descriptions = Bool(get(params, "include_descriptions", false))

            tools_payload = if include_descriptions
                [
                    Dict(
                        "name" => string(tool.name),
                        "description" => string(tool.description),
                    )
                    for tool in ALL_TOOLS
                ]
            else
                String[string(tool.name) for tool in ALL_TOOLS]
            end

            resources_payload = if include_descriptions
                [
                    Dict(
                        "uri" => string(resource.uri),
                        "name" => string(resource.name),
                        "title" => isnothing(resource.title) ? nothing : string(resource.title),
                    )
                    for resource in ALL_RESOURCES
                ]
            else
                String[string(resource.uri) for resource in ALL_RESOURCES]
            end

            templates_payload = if include_descriptions
                [
                    Dict(
                        "uri_template" => string(template.uri_template),
                        "name" => string(template.name),
                        "description" => string(template.description),
                    )
                    for template in ALL_RESOURCE_TEMPLATES
                ]
            else
                String[string(template.uri_template) for template in ALL_RESOURCE_TEMPLATES]
            end

            prompts_payload = if include_descriptions
                [
                    Dict(
                        "name" => string(prompt.name),
                        "title" => isnothing(prompt.title) ? nothing : string(prompt.title),
                        "description" => string(prompt.description),
                    )
                    for prompt in ALL_PROMPTS
                ]
            else
                String[string(prompt.name) for prompt in ALL_PROMPTS]
            end

            payload = Dict(
                "status" => "success",
                "tools" => tools_payload,
                "resources" => resources_payload,
                "resource_templates" => templates_payload,
                "prompts" => prompts_payload,
                "protocol_method_fallback" => Dict(
                    "preferred_protocol_methods" => [
                        "resources/templates/list",
                        "resources/read",
                        "prompts/list",
                    ],
                    "fallback_note" => "Use this tool when the client SDK cannot call protocol-level methods directly.",
                ),
            )

            return create_json_response(payload)
        catch e
            return create_error_response(e)
        end
    end,
)

export list_mcp_surfaces_tool
