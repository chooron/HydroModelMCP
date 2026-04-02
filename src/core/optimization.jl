module OptimizationStrategies

using ..Optimization
using ..OptimizationBBO
using ..OptimizationMetaheuristics

export available_algorithms,
    normalize_algorithm_name,
    recommend_algorithm,
    resolve_backend_algorithm,
    resolve_optimizer

const ALGORITHM_ALIASES = Dict{String,String}(
    "bbo" => "BBO",
    "biogeographybasedoptimization" => "BBO",
    "de" => "DE",
    "differentialevolution" => "DE",
    "pso" => "PSO",
    "particleswarmoptimization" => "PSO",
    "cmaes" => "CMAES",
    "eca" => "ECA",
    "dds" => "DDS",
    "dynamicallydimensionedsearch" => "DDS",
    "sce" => "SCE",
    "shuffledcomplexevolution" => "SCE",
    "auto" => "AUTO",
    "自动" => "AUTO",
    "低预算" => "DDS",
    "高预算" => "SCE",
)

const SUPPORTED_ALGORITHMS = Set(["AUTO", "DDS", "SCE", "BBO", "DE", "PSO", "CMAES", "ECA"])

const ALGORITHM_BACKEND = Dict(
    "BBO" => "BBO",
    "DE" => "DE",
    "PSO" => "PSO",
    "CMAES" => "CMAES",
    "ECA" => "ECA",
    # Strategy-level aliases: map to stable library optimizers
    "DDS" => "DE",
    "SCE" => "PSO",
)

available_algorithms() = sort!(collect(SUPPORTED_ALGORITHMS))

function normalize_algorithm_name(value::String)
    raw = strip(value)
    isempty(raw) && throw(ArgumentError("algorithm cannot be empty"))
    key = lowercase(replace(raw, r"[^0-9A-Za-z一-龥]+" => ""))
    canonical = get(ALGORITHM_ALIASES, key, uppercase(raw))
    canonical in SUPPORTED_ALGORITHMS || throw(ArgumentError(
        "Unknown algorithm '$value'. Supported: $(join(available_algorithms(), ", "))",
    ))
    return canonical
end

function recommend_algorithm(; budget::String = "medium", n_parameters::Int = 6)
    budget_norm = lowercase(strip(budget))
    if budget_norm == "low" || n_parameters >= 10
        return "DDS"
    elseif budget_norm == "high"
        return "SCE"
    elseif n_parameters >= 7
        return "DDS"
    end

    return "BBO"
end

function resolve_optimizer(algorithm::String)
    backend = resolve_backend_algorithm(algorithm)

    map = Dict(
        "BBO" => BBO_adaptive_de_rand_1_bin_radiuslimited(),
        "DE" => BBO_de_rand_1_bin(),
        "PSO" => OptimizationMetaheuristics.PSO(),
        "CMAES" => OptimizationMetaheuristics.CGSA(),
        "ECA" => OptimizationMetaheuristics.ECA(),
    )
    return map[backend]
end

function resolve_backend_algorithm(algorithm::String)
    canonical = normalize_algorithm_name(algorithm)
    canonical == "AUTO" && throw(ArgumentError("AUTO must be resolved before backend selection"))
    return get(ALGORITHM_BACKEND, canonical, canonical)
end

end # module OptimizationStrategies
