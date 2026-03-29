using ModelContextProtocol

function build_model_catalog()
    models = Discovery.list_models()
    catalog = [
        Dict(
            "name" => model_name,
            "info_uri" => "hydro://models/$model_name/info",
            "parameters_uri" => "hydro://models/$model_name/parameters",
            "variables_uri" => "hydro://models/$model_name/variables"
        )
        for model_name in models
    ]

    return Dict("models" => catalog, "count" => length(catalog))
end

const model_catalog_resource = MCPResource(
    uri = "hydro://models/catalog",
    name = "Model Catalog",
    description = "Complete list of available hydrological models and their exact resource URIs.",
    mime_type = "application/json",
    data_provider = build_model_catalog
)

const algorithm_guide_resource = MCPResource(
    uri = "hydro://guides/algorithms",
    name = "Algorithm Selection Guide",
    description = "Recommendations for choosing calibration algorithms based on problem characteristics.",
    mime_type = "application/json",
    data_provider = () -> Dict(
        "recommendations" => [
            Dict(
                "condition" => "Low budget (<500 evals) or high dimensions (>10 params)",
                "algorithm" => "BBO",
                "reason" => "Efficient when evaluations are limited."
            ),
            Dict(
                "condition" => "High budget (>5000 evals) and low dimensions (<10 params)",
                "algorithm" => "PSO",
                "reason" => "Supports broader exploration when budget allows."
            ),
            Dict(
                "condition" => "Multiple objectives",
                "algorithm" => "NSGA2",
                "reason" => "Designed to approximate a Pareto front."
            )
        ],
        "algorithms" => Dict(
            "BBO" => Dict("name" => "Biogeography-Based Optimization", "best_for" => "Limited budget"),
            "PSO" => Dict("name" => "Particle Swarm Optimization", "best_for" => "High budget"),
            "DE" => Dict("name" => "Differential Evolution", "best_for" => "Continuous problems"),
            "CMAES" => Dict("name" => "CMA-ES", "best_for" => "Smooth landscapes"),
            "NSGA2" => Dict("name" => "NSGA-II", "best_for" => "Multi-objective")
        )
    )
)

function create_model_resources()
    resources = MCPResource[]

    for model_name in Discovery.list_models()
        let model_name = model_name
            append!(resources, [
                MCPResource(
                    uri = "hydro://models/$model_name/info",
                    name = "Model Info: $model_name",
                    description = "Detailed information for the $model_name model.",
                    mime_type = "application/json",
                    data_provider = () -> Discovery.get_model_info(model_name)
                ),
                MCPResource(
                    uri = "hydro://models/$model_name/parameters",
                    name = "Model Parameters: $model_name",
                    description = "Parameter bounds and descriptions for the $model_name model.",
                    mime_type = "application/json",
                    data_provider = () -> Dict(
                        "model" => model_name,
                        "parameters" => Discovery.get_parameters_detail(model_name)
                    )
                ),
                MCPResource(
                    uri = "hydro://models/$model_name/variables",
                    name = "Model Variables: $model_name",
                    description = "Input, state, and output variables for the $model_name model.",
                    mime_type = "application/json",
                    data_provider = () -> Dict(
                        "model" => model_name,
                        "variables" => Discovery.get_variables_detail(model_name)
                    )
                )
            ])
        end
    end

    return resources
end

export model_catalog_resource, algorithm_guide_resource, create_model_resources
