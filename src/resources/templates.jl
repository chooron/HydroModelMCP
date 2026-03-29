using ModelContextProtocol

const ALL_RESOURCE_TEMPLATES = ResourceTemplate[
    ResourceTemplate(
        name = "model_info",
        uri_template = "hydro://models/{model_name}/info",
        mime_type = "application/json",
        description = "Detailed information about a specific hydrological model."
    ),
    ResourceTemplate(
        name = "model_parameters",
        uri_template = "hydro://models/{model_name}/parameters",
        mime_type = "application/json",
        description = "Parameter bounds and descriptions for a specific hydrological model."
    ),
    ResourceTemplate(
        name = "model_variables",
        uri_template = "hydro://models/{model_name}/variables",
        mime_type = "application/json",
        description = "Input, state, and output variables for a specific hydrological model."
    ),
    ResourceTemplate(
        name = "calibration_result",
        uri_template = "hydro://calibration/results/{result_id}",
        mime_type = "application/json",
        description = "A stored calibration result by identifier."
    ),
    ResourceTemplate(
        name = "sensitivity_result",
        uri_template = "hydro://sensitivity/results/{result_id}",
        mime_type = "application/json",
        description = "A stored sensitivity-analysis result by identifier."
    )
]

const resource_templates_resource = MCPResource(
    uri = "hydro://meta/resource-templates",
    name = "Resource Template Metadata",
    description = "Metadata about template-style resource URIs supported by this project.",
    mime_type = "application/json",
    data_provider = () -> Dict(
        "templates" => [
            Dict(
                "name" => template.name,
                "uri_template" => template.uri_template,
                "mime_type" => template.mime_type,
                "description" => template.description
            )
            for template in ALL_RESOURCE_TEMPLATES
        ],
        "note" => "ModelContextProtocol.jl v0.4.0 currently resolves exact resource URIs during resources/read, so this server registers exact model URIs and exposes template metadata separately."
    )
)

export ALL_RESOURCE_TEMPLATES, resource_templates_resource
