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
                "algorithm" => "DDS",
                "reason" => "Strategy alias for low-budget mode (mapped to stable DE backend).",
            ),
            Dict(
                "condition" => "High budget (>5000 evals)",
                "algorithm" => "SCE",
                "reason" => "Strategy alias for high-budget exploration (mapped to stable PSO backend).",
            ),
            Dict(
                "condition" => "Medium budget with moderate dimensions",
                "algorithm" => "BBO",
                "reason" => "Balanced default when DDS/SCE conditions are not dominant.",
            ),
            Dict(
                "condition" => "Multiple objectives",
                "algorithm" => "NSGA2",
                "reason" => "Designed to approximate a Pareto front.",
            ),
        ],
        "algorithms" => Dict(
            "DDS" => Dict(
                "name" => "Dynamically Dimensioned Search",
                "name_zh" => "动态维度搜索",
                "best_for" => "Low budget or high-dimensional calibration",
                "aliases" => ["DDS", "dynamically dimensioned search", "动态维度搜索"],
                "backend" => "DE",
                "implementation_note" => "Exposed as strategy alias; execution uses library DE backend for stability.",
            ),
            "SCE" => Dict(
                "name" => "Shuffled Complex Evolution",
                "name_zh" => "复合体进化",
                "best_for" => "High-budget global calibration",
                "aliases" => ["SCE", "shuffled complex evolution", "复合体进化"],
                "backend" => "PSO",
                "implementation_note" => "Exposed as strategy alias; execution uses library PSO backend for stability.",
            ),
            "BBO" => Dict(
                "name" => "Biogeography-Based Optimization",
                "name_zh" => "生物地理优化",
                "best_for" => "Limited budget",
                "aliases" => ["BBO", "biogeography-based optimization", "生物地理优化", "生物地理优化算法"],
            ),
            "PSO" => Dict(
                "name" => "Particle Swarm Optimization",
                "name_zh" => "粒子群优化",
                "best_for" => "High budget",
                "aliases" => ["PSO", "particle swarm optimization", "粒子群", "粒子群优化"],
            ),
            "DE" => Dict(
                "name" => "Differential Evolution",
                "name_zh" => "差分进化",
                "best_for" => "Continuous problems",
                "aliases" => ["DE", "differential evolution", "差分进化", "差分进化算法"],
            ),
            "CMAES" => Dict(
                "name" => "CMA-ES",
                "name_zh" => "协方差矩阵自适应进化策略",
                "best_for" => "Smooth landscapes",
                "aliases" => ["CMAES", "CMA-ES", "协方差矩阵自适应进化策略"],
            ),
            "ECA" => Dict(
                "name" => "Evolutionary Center Algorithm",
                "name_zh" => "进化中心算法",
                "best_for" => "Lightweight exploratory calibration",
                "aliases" => ["ECA", "evolutionary center algorithm", "进化中心算法"],
            ),
            "NSGA2" => Dict(
                "name" => "NSGA-II",
                "name_zh" => "多目标遗传算法2",
                "best_for" => "Multi-objective",
                "aliases" => ["NSGA2", "NSGA-II", "多目标遗传算法2"],
            ),
            "NSGA3" => Dict(
                "name" => "NSGA-III",
                "name_zh" => "多目标遗传算法3",
                "best_for" => "Many-objective",
                "aliases" => ["NSGA3", "NSGA-III", "多目标遗传算法3"],
            ),
        ),
        "runtime_solver_options" => [
            Dict(
                "canonical" => "DISCRETE",
                "aliases" => ["DISCRETE", "discrete", "离散", "离散求解", "离散求解器"],
                "best_for" => "Default robust hydrological routing",
            ),
            Dict(
                "canonical" => "ODE",
                "aliases" => ["ODE", "continuous", "连续", "连续求解", "连续求解器"],
                "best_for" => "Continuous-time dynamics or ODE-driven formulations",
            ),
            Dict(
                "canonical" => "MUTABLE",
                "aliases" => ["MUTABLE", "mutable", "可变", "可变求解"],
                "best_for" => "Advanced performance tuning with mutable state",
            ),
            Dict(
                "canonical" => "IMMUTABLE",
                "aliases" => ["IMMUTABLE", "immutable", "不可变", "不可变求解"],
                "best_for" => "Functional-style immutable state evolution",
            ),
        ],
        "runtime_interpolation_options" => [
            Dict("canonical" => "LINEAR", "aliases" => ["LINEAR", "linear", "线性", "线性插值"]),
            Dict("canonical" => "CONSTANT", "aliases" => ["CONSTANT", "constant", "step", "常量", "阶梯"]),
            Dict("canonical" => "DIRECT", "aliases" => ["DIRECT", "direct", "直接", "直接映射"]),
        ],
    ),
)

export algorithm_guide_resource, create_model_resources, model_catalog_resource, model_discovery_guide_resource
