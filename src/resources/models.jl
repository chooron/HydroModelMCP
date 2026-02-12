using ModelContextProtocol
using JSON3
using URIs

# Resource 1: Model Catalog
model_catalog_resource = MCPResource(
    uri = "hydro://models/catalog",
    name = "Model Catalog",
    description = "Complete list of available hydrological models with basic info",
    mime_type = "application/json",
    data_provider = () -> begin
        models = Discovery.list_models()
        catalog = [
            Dict(
                "name" => m,
                "info_uri" => "hydro://models/$m/info",
                "params_uri" => "hydro://models/$m/parameters"
            ) for m in models
        ]
        return TextResourceContents(
            uri = URI("hydro://models/catalog"),
            mime_type = "application/json",
            text = JSON3.write(Dict("models" => catalog, "count" => length(models)))
        )
    end
)

# Resource 2: Algorithm Recommendations
algorithm_guide_resource = MCPResource(
    uri = "hydro://guides/algorithms",
    name = "Algorithm Selection Guide",
    description = "Recommendations for choosing calibration algorithms based on problem characteristics",
    mime_type = "application/json",
    data_provider = () -> begin
        guide = Dict(
            "recommendations" => [
                Dict("condition" => "Low budget (<500 evals) OR high dimensions (>10 params)",
                     "algorithm" => "BBO", "reason" => "Efficient for limited budgets"),
                Dict("condition" => "High budget (>5000 evals) AND low dimensions (<10 params)",
                     "algorithm" => "PSO", "reason" => "Thorough exploration with sufficient budget"),
                Dict("condition" => "Multiple objectives",
                     "algorithm" => "NSGA2", "reason" => "Pareto front generation"),
            ],
            "algorithms" => Dict(
                "BBO" => Dict("name" => "Biogeography-Based Optimization", "best_for" => "Limited budget"),
                "PSO" => Dict("name" => "Particle Swarm Optimization", "best_for" => "High budget"),
                "DE" => Dict("name" => "Differential Evolution", "best_for" => "Continuous problems"),
                "CMAES" => Dict("name" => "CMA-ES", "best_for" => "Smooth landscapes"),
                "NSGA2" => Dict("name" => "NSGA-II", "best_for" => "Multi-objective")
            )
        )
        return TextResourceContents(
            uri = URI("hydro://guides/algorithms"),
            mime_type = "application/json",
            text = JSON3.write(guide)
        )
    end
)

export model_catalog_resource, algorithm_guide_resource
