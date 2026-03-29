using ModelContextProtocol

const RESOURCE_TEMPLATE_METADATA = [
    Dict(
        "name" => "model_info",
        "uri_template" => "hydro://models/{model_name}/info",
        "mime_type" => "application/json",
        "description" => "Detailed information about a specific hydrological model.",
        "availability" => "dynamically readable through the template provider",
        "fallback_tools" => ["find_model", "get_model_info"],
        "example_uri" => "hydro://models/exphydro/info",
    ),
    Dict(
        "name" => "model_parameters",
        "uri_template" => "hydro://models/{model_name}/parameters",
        "mime_type" => "application/json",
        "description" => "Parameter metadata for a specific hydrological model.",
        "availability" => "dynamically readable through the template provider",
        "fallback_tools" => ["find_model", "get_model_parameters"],
        "example_uri" => "hydro://models/exphydro/parameters",
    ),
    Dict(
        "name" => "model_variables",
        "uri_template" => "hydro://models/{model_name}/variables",
        "mime_type" => "application/json",
        "description" => "Variable metadata for a specific hydrological model.",
        "availability" => "dynamically readable through the template provider",
        "fallback_tools" => ["find_model", "get_model_variables"],
        "example_uri" => "hydro://models/exphydro/variables",
    ),
    Dict(
        "name" => "model_knowledge",
        "uri_template" => "hydro://models/{model_name}/knowledge",
        "mime_type" => "application/json",
        "description" => "Supplementary model knowledge card derived from static model.json content.",
        "availability" => "dynamically readable through the template provider",
        "fallback_tools" => ["find_model", "get_model_info"],
        "example_uri" => "hydro://models/gr4j/knowledge",
    ),
    Dict(
        "name" => "calibration_result",
        "uri_template" => "hydro://calibration/results/{result_id}",
        "mime_type" => "application/json",
        "description" => "A stored calibration result by identifier.",
        "availability" => "dynamically readable through the template provider; exact resources may also be registered for startup-visible IDs",
        "fallback_tools" => String[],
        "example_uri" => "hydro://calibration/results/<result_id>",
    ),
    Dict(
        "name" => "sensitivity_result",
        "uri_template" => "hydro://sensitivity/results/{result_id}",
        "mime_type" => "application/json",
        "description" => "A stored sensitivity-analysis result by identifier.",
        "availability" => "dynamically readable through the template provider; exact resources may also be registered for startup-visible IDs",
        "fallback_tools" => String[],
        "example_uri" => "hydro://sensitivity/results/<result_id>",
    ),
    Dict(
        "name" => "ensemble_result",
        "uri_template" => "hydro://ensemble/results/{result_id}",
        "mime_type" => "application/json",
        "description" => "A stored ensemble-run result by identifier.",
        "availability" => "dynamically readable through the template provider; exact resources may also be registered for startup-visible IDs",
        "fallback_tools" => ["run_ensemble_parameters"],
        "example_uri" => "hydro://ensemble/results/<result_id>",
    ),
]

const ALL_RESOURCE_TEMPLATES = ResourceTemplate[
    ResourceTemplate(
        name = metadata["name"],
        uri_template = metadata["uri_template"],
        mime_type = metadata["mime_type"],
        description = metadata["description"],
    )
    for metadata in RESOURCE_TEMPLATE_METADATA
]

const resource_templates_resource = MCPResource(
    uri = "hydro://meta/resource-templates",
    name = "Resource Template Metadata",
    title = "Resource Template Catalog",
    description = "Metadata about template-style resource URIs supported by this project.",
    mime_type = "application/json",
    data_provider = () -> Dict(
        "templates" => RESOURCE_TEMPLATE_METADATA,
        "limitations" => [
            "HydroModelMCP patches the installed ModelContextProtocol.jl runtime so resources/templates/list is exposed and resources/read can resolve supported URI templates dynamically.",
            "HydroModelMCP keeps runtime model metadata tool-first while allowing template-addressable reads for model info, parameters, variables, and supplementary knowledge.",
            "Stored result URIs may still be registered exactly when result IDs are already present at startup.",
            "If a new stored result is created after startup, rebuild or restart the server to expose its exact resource URI.",
        ],
        "preferred_model_tools" => ["find_model", "get_model_info", "get_model_parameters", "get_model_variables"],
    ),
)

export ALL_RESOURCE_TEMPLATES, RESOURCE_TEMPLATE_METADATA, resource_templates_resource
