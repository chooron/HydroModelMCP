using ModelContextProtocol

function build_model_catalog()
    models = Discovery.list_models()
    catalog = [
        Dict(
            "name" => model_name,
            "knowledge_uri" => model_knowledge_uri(model_name),
            "knowledge_card_available" => has_model_knowledge(model_name),
        )
        for model_name in models
    ]

    return Dict(
        "models" => catalog,
        "count" => length(catalog),
        "resource_templates" => Dict(
            "info" => "hydro://models/{model_name}/info",
            "parameters" => "hydro://models/{model_name}/parameters",
            "variables" => "hydro://models/{model_name}/variables",
            "knowledge" => "hydro://models/{model_name}/knowledge",
        ),
        "preferred_tools" => Dict(
            "resolve_model" => "find_model",
            "get_info" => "get_model_info",
            "get_parameters" => "get_model_parameters",
            "get_variables" => "get_model_variables",
        ),
        "note" => "Runtime model details are resolved on demand through unified discovery tools. Supplementary knowledge cards are exposed through the hydro://models/{model_name}/knowledge template.",
    )
end

const model_catalog_resource = MCPResource(
    uri = "hydro://models/catalog",
    name = "Model Catalog",
    title = "Hydrological Model Catalog",
    description = "Available hydrological models with preferred discovery tools and template metadata.",
    mime_type = "application/json",
    data_provider = build_model_catalog,
)

const model_discovery_guide_resource = MCPResource(
    uri = "hydro://guides/model-discovery",
    name = "Model Discovery Guide",
    title = "Model Discovery Workflow",
    description = "Recommended sequences for browsing and resolving model metadata through HydroModelMCP.",
    mime_type = "application/json",
    data_provider = () -> Dict(
        "recommended_sequences" => [
            Dict(
                "name" => "resolve_from_user_query",
                "steps" => ["find_model", "get_model_info", "get_model_parameters"],
                "why" => "Best path when the user gives a partial or approximate model name.",
            ),
            Dict(
                "name" => "browse_then_resolve",
                "steps" => ["resources/read hydro://models/catalog", "find_model", "get_model_info"],
                "why" => "Good for clients that want a compact browseable catalog before precise lookup.",
            ),
        ],
        "notes" => [
            "This server keeps runtime metadata tool-first and exposes template-addressable model knowledge backed by static model.json content.",
            "Use the unified discovery tools for exact model details such as parameters, variables, and execution contracts.",
        ],
    ),
)

function create_model_resources()
    return MCPResource[
        model_catalog_resource,
        model_discovery_guide_resource,
    ]
end

const algorithm_guide_resource = MCPResource(
    uri = "hydro://guides/algorithms",
    name = "Algorithm Selection Guide",
    title = "Calibration Algorithm Guide",
    description = "Recommendations for choosing calibration algorithms based on problem characteristics.",
    mime_type = "application/json",
    data_provider = () -> Dict(
        "recommendations" => [
            Dict(
                "condition" => "Low budget (<500 evals) or high dimensions (>10 params)",
                "algorithm" => "BBO",
                "reason" => "Efficient when evaluations are limited.",
            ),
            Dict(
                "condition" => "High budget (>5000 evals) and low dimensions (<10 params)",
                "algorithm" => "PSO",
                "reason" => "Supports broader exploration when budget allows.",
            ),
            Dict(
                "condition" => "Multiple objectives",
                "algorithm" => "NSGA2",
                "reason" => "Designed to approximate a Pareto front.",
            ),
        ],
        "algorithms" => Dict(
            "BBO" => Dict("name" => "Biogeography-Based Optimization", "best_for" => "Limited budget"),
            "PSO" => Dict("name" => "Particle Swarm Optimization", "best_for" => "High budget"),
            "DE" => Dict("name" => "Differential Evolution", "best_for" => "Continuous problems"),
            "CMAES" => Dict("name" => "CMA-ES", "best_for" => "Smooth landscapes"),
            "NSGA2" => Dict("name" => "NSGA-II", "best_for" => "Multi-objective"),
        ),
    ),
)

export algorithm_guide_resource, create_model_resources, model_catalog_resource, model_discovery_guide_resource
