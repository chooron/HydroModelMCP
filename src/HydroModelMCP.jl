module HydroModelMCP

using CSV
using Dates
using Random
using DataFrames
using JSON3
using Redis
using UUIDs
using Base64
using Serialization

using ComponentArrays
using DataInterpolations
using Statistics

using ModelContextProtocol
using ModelContextProtocol: HttpTransport, MCPPrompt, MCPResource, MCPTool, StdioTransport

using HydroModels
using HydroModelLibrary

using Optimization
using OptimizationBBO
using OptimizationMetaheuristics
using GlobalSensitivity

include("schemas/Schemas.jl")
using .Schemas

include("data_handles.jl")
include("utils/dataloader.jl")
include("core/metrics.jl")
include("core/datasplitter.jl")
include("core/sampling.jl")
include("utils/unified_inputs.jl")
include("core/simulation.jl")
include("core/discovery.jl")
include("core/sensitivity.jl")
include("core/calibration.jl")
include("utils/storage.jl")
include("core/ensemble.jl")
include("core/validation.jl")

include("resources/models.jl")
include("resources/algorithms.jl")
include("resources/calibration.jl")
include("resources/parameters.jl")
include("resources/templates.jl")
include("resources/workspace.jl")
include("resources/hints.jl")
include("mcp_resource_templates.jl")

include("tools/helpers.jl")
include("tools/data_loading.jl")
include("tools/simulation.jl")
include("tools/discovery.jl")
include("tools/workspace.jl")
include("tools/session_cache.jl")
include("tools/calibration.jl")
include("tools/ensemble.jl")
include("tools/validation.jl")

include("prompts/experts.jl")
include("prompts/workflows.jl")

const SERVER_NAME = "hydro-model-agent-interface"
const SERVER_VERSION = "0.1.0"
const SERVER_DESCRIPTION = "Hydrological modeling MCP server backed by HydroModels.jl."

function parse_env_int(name::String, default::String)
    value = get(ENV, name, default)
    try
        return parse(Int, value)
    catch err
        throw(ArgumentError("Environment variable $name must be an integer, got '$value'."))
    end
end

function build_storage_backend()
    backend = lowercase(get(ENV, "STORAGE_BACKEND", "file"))
    ttl = parse_env_int("HYDRO_STORAGE_TTL", "604800")

    if backend == "redis"
        return Storage.RedisBackend(
            get(ENV, "REDIS_HOST", "127.0.0.1"),
            parse_env_int("REDIS_PORT", "6379"),
            "hydro";
            ttl = ttl
        )
    elseif backend == "file"
        return Storage.FileBackend(
            get(ENV, "HYDRO_STORAGE_PATH", joinpath(homedir(), ".hydro_mcp", "storage"));
            ttl = ttl
        )
    end

    throw(ArgumentError("STORAGE_BACKEND must be 'file' or 'redis', got '$backend'."))
end

const STORAGE_BACKEND = build_storage_backend()

const ALL_TOOLS = MCPTool[
    load_camels_data_tool,
    analyze_distribution_from_handle_tool,
    inspect_hydro_data_tool,
    load_hydro_csv_tool,
    list_models_tool,
    find_model_tool,
    get_model_info_tool,
    get_model_variables_tool,
    get_model_parameters_tool,
    list_workspace_files_tool,
    clear_session_cache_tool,
    simulation_tool,
    ensemble_parameter_tool,
    validation_tool,
    compute_metrics_tool,
    split_data_tool,
    sensitivity_tool,
    sensitivity_analysis_tool,
    sampling_tool,
    calibrate_tool,
    calibrate_multi_tool,
    diagnose_tool,
    configure_objectives_tool,
    init_calibration_setup_tool,
    compute_diagnostics_full_tool,
]

const ALL_PROMPTS = MCPPrompt[
    Experts.hydro_expert_prompt,
    Workflows.runoff_workspace_prompt,
    Workflows.calibration_workflow_prompt,
    Workflows.result_review_prompt,
]

function build_resources(storage_backend = STORAGE_BACKEND)
    resources = MCPResource[]

    append!(resources, create_model_resources())
    append!(resources, create_model_knowledge_resources())
    append!(resources, MCPResource[
        algorithm_guide_resource,
        objective_guide_resource,
        metrics_guide_resource,
        data_handle_guide_resource,
        runoff_workspace_guide_resource,
        result_artifact_guide_resource,
        resource_templates_resource,
    ])

    append!(resources, create_llm_hint_resources())

    append!(resources, create_storage_resources(storage_backend))

    return resources
end

const ALL_RESOURCES = build_resources()

function build_server(;
    transport = nothing,
    storage_backend = STORAGE_BACKEND
)
    resources = storage_backend === STORAGE_BACKEND ? ALL_RESOURCES : build_resources(storage_backend)

    server = mcp_server(
        name = SERVER_NAME,
        version = SERVER_VERSION,
        description = SERVER_DESCRIPTION,
        tools = ALL_TOOLS,
        resources = resources,
        prompts = ALL_PROMPTS
    )

    server.resource_templates = copy(ALL_RESOURCE_TEMPLATES)
    register_resource_template_providers!(server, build_resource_template_providers(storage_backend))

    if !isnothing(transport)
        server.transport = transport
    end

    return server
end

function run_server()
    server = build_server(transport = StdioTransport())
    start!(server)
end

function run_http_server(;
    host::String = get(ENV, "MCP_HOST", "127.0.0.1"),
    port::Int = parse_env_int("MCP_PORT", "3000"),
    allowed_origins::Vector{String} = String[]
)
    transport = HttpTransport(
        host = host,
        port = port,
        endpoint = "/",
        protocol_version = "2025-06-18",
        session_required = false,
        allowed_origins = isempty(allowed_origins) ? ["*"] : allowed_origins
    )

    server = build_server(transport = transport)
    ModelContextProtocol.connect(transport)

    println("HydroModelMCP HTTP server started.")
    println("  Address: http://$host:$port")
    println("  Protocol version: 2025-06-18")
    println("  Tools: $(length(server.tools))")
    println("  Resources: $(length(server.resources))")
    println("  Prompts: $(length(server.prompts))")

    start!(server)
end

export ALL_PROMPTS, ALL_RESOURCES, ALL_RESOURCE_TEMPLATES, ALL_TOOLS
export STORAGE_BACKEND, build_resources, build_server, build_storage_backend
export run_http_server, run_server

end # module HydroModelMCP
