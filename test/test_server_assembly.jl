using ModelContextProtocol
using JSON3
using .HydroModelMCP
using .HydroModelMCP.Discovery

@testset "Server Assembly Tests" begin
    server = HydroModelMCP.build_server()
    model_names = Discovery.list_models()

    tool_names = [tool.name for tool in server.tools]
    resource_uris = [string(resource.uri) for resource in server.resources]
    prompt_names = [prompt.name for prompt in server.prompts]
    template_uris = [template.uri_template for template in server.resource_templates]

    @test "list_models" in tool_names
    @test "load_caravan_data" in tool_names
    @test !("load_camels_data" in tool_names)
    @test "inspect_hydro_data" in tool_names
    @test "list_workspace_files" in tool_names
    @test "list_mcp_surfaces" in tool_names
    @test "clear_session_cache" in tool_names
    @test "list_stored_results" in tool_names
    @test "get_stored_result" in tool_names
    @test "diagnose_multiobjective" in tool_names
    @test "auto_calibration_workflow" in tool_names

    @test "hydrology_expert_review" in prompt_names
    @test "hydro_minimal_tool_router" in prompt_names
    @test "runoff_workspace_workflow" in prompt_names
    @test "calibration_workflow_plan" in prompt_names
    @test "hydrology_result_review" in prompt_names
    @test "hydro_minimal_workflow_router" in prompt_names

    @test "hydro://models/catalog" in resource_uris
    @test "hydro://models/knowledge-index" in resource_uris
    @test "hydro://models/knowledge-coverage" in resource_uris
    @test "hydro://guides/model-discovery" in resource_uris
    @test "hydro://guides/algorithms" in resource_uris
    @test "hydro://guides/objectives" in resource_uris
    @test "hydro://guides/metrics" in resource_uris
    @test "hydro://guides/data-handles" in resource_uris
    @test "hydro://guides/runoff-workspace" in resource_uris
    @test "hydro://guides/result-artifacts" in resource_uris
    @test "hydro://guides/llm-hints" in resource_uris
    @test "hydro://guides/llm-quickstart" in resource_uris
    @test "hydro://meta/resource-templates" in resource_uris
    @test "hydro://calibration/results" in resource_uris
    @test "hydro://sensitivity/results" in resource_uris
    @test "hydro://ensemble/results" in resource_uris

    first_model = first(model_names)
    @test !("hydro://models/$first_model/knowledge" in resource_uris)
    @test !("hydro://models/$first_model/info" in resource_uris)
    @test !("hydro://models/$first_model/parameters" in resource_uris)
    @test !("hydro://models/$first_model/variables" in resource_uris)

    @test "hydro://models/{model_name}/info" in template_uris
    @test "hydro://models/{model_name}/parameters" in template_uris
    @test "hydro://models/{model_name}/variables" in template_uris
    @test "hydro://models/{model_name}/knowledge" in template_uris
    @test "hydro://hints/{feature}" in template_uris
    @test "hydro://workflows/{intent}" in template_uris
    @test "hydro://ensemble/results/{result_id}" in template_uris

    @testset "config env prefers ENV over dotenv fallback" begin
        base = joinpath(dirname(@__DIR__), ".tmp_tests")
        mkpath(base)
        mktempdir(base; prefix = "dotenv-config-") do tmpdir
            dotenv_path = joinpath(tmpdir, ".env")
            open(dotenv_path, "w") do io
                write(io, "CARAVAN_DATASET_ROOT=G:/Dataset/FromDotEnv\n")
            end

            old_value = get(ENV, "CARAVAN_DATASET_ROOT", nothing)
            try
                haskey(ENV, "CARAVAN_DATASET_ROOT") && delete!(ENV, "CARAVAN_DATASET_ROOT")
                @test HydroModelMCP.get_config_env("CARAVAN_DATASET_ROOT", nothing; dotenv_path = dotenv_path) == "G:/Dataset/FromDotEnv"

                ENV["CARAVAN_DATASET_ROOT"] = "G:/Dataset/FromEnv"
                @test HydroModelMCP.get_config_env("CARAVAN_DATASET_ROOT", nothing; dotenv_path = dotenv_path) == "G:/Dataset/FromEnv"
            finally
                if old_value === nothing
                    haskey(ENV, "CARAVAN_DATASET_ROOT") && delete!(ENV, "CARAVAN_DATASET_ROOT")
                else
                    ENV["CARAVAN_DATASET_ROOT"] = old_value
                end
            end
        end
    end

    catalog_payload = HydroModelMCP.model_catalog_resource.data_provider()
    @test catalog_payload isa Dict
    @test catalog_payload["count"] == length(model_names)
    @test haskey(first(catalog_payload["models"]), "name")
    @test haskey(first(catalog_payload["models"]), "knowledge_uri")
    @test haskey(first(catalog_payload["models"]), "knowledge_card_available")
    @test haskey(catalog_payload, "preferred_tools")
    @test haskey(catalog_payload, "resource_templates")

    knowledge_payload = HydroModelMCP.model_knowledge_payload("gr4j")
    @test knowledge_payload["runtime_model"] == "gr4j"
    @test knowledge_payload["knowledge_card_available"] == true
    @test haskey(knowledge_payload, "knowledge_card")
    @test knowledge_payload["knowledge_card"]["model_name"]["display_name"] == "GR4J"

    missing_knowledge_payload = HydroModelMCP.model_knowledge_payload("exphydro")
    @test missing_knowledge_payload["runtime_model"] == "exphydro"
    @test missing_knowledge_payload["knowledge_card_available"] == false
    @test haskey(missing_knowledge_payload, "message")

    coverage_payload = HydroModelMCP.model_knowledge_coverage_resource.data_provider()
    @test "gr4j" in coverage_payload["runtime_models_with_cards"]
    @test "exphydro" in coverage_payload["runtime_models_without_cards"]

    template_list_response = ModelContextProtocol.handle_request(
        server,
        ModelContextProtocol.JSONRPCRequest(
            id = 101,
            method = "resources/templates/list",
            params = ModelContextProtocol.ListResourcesParams(),
        ),
    )
    @test template_list_response isa ModelContextProtocol.JSONRPCResponse
    listed_templates = template_list_response.result["resourceTemplates"]
    @test any(t -> t["uriTemplate"] == "hydro://models/{model_name}/knowledge", listed_templates)
    @test any(t -> t["uriTemplate"] == "hydro://hints/{feature}", listed_templates)
    @test any(t -> t["uriTemplate"] == "hydro://workflows/{intent}", listed_templates)

    parsed_template_request = ModelContextProtocol.parse_message(
        "{\"jsonrpc\":\"2.0\",\"id\":104,\"method\":\"resources/templates/list\",\"params\":{}}",
    )
    @test parsed_template_request isa ModelContextProtocol.JSONRPCRequest
    @test parsed_template_request.params isa ModelContextProtocol.ListResourcesParams

    template_read_response = ModelContextProtocol.handle_request(
        server,
        ModelContextProtocol.JSONRPCRequest(
            id = 102,
            method = "resources/read",
            params = ModelContextProtocol.ReadResourceParams(uri = "hydro://models/gr4j/knowledge"),
        ),
    )
    @test template_read_response isa ModelContextProtocol.JSONRPCResponse
    template_contents = template_read_response.result.contents
    @test length(template_contents) == 1
    @test template_contents[1]["uri"] == "hydro://models/gr4j/knowledge"
    @test occursin("GR4J", template_contents[1]["text"])

    dynamic_info_response = ModelContextProtocol.handle_request(
        server,
        ModelContextProtocol.JSONRPCRequest(
            id = 103,
            method = "resources/read",
            params = ModelContextProtocol.ReadResourceParams(uri = "hydro://models/gr4j/info"),
        ),
    )
    @test dynamic_info_response isa ModelContextProtocol.JSONRPCResponse
    @test occursin("\"model\":\"gr4j\"", dynamic_info_response.result.contents[1]["text"])

    hint_alias_response = ModelContextProtocol.handle_request(
        server,
        ModelContextProtocol.JSONRPCRequest(
            id = 105,
            method = "resources/read",
            params = ModelContextProtocol.ReadResourceParams(uri = "hydro://hints/率定"),
        ),
    )
    @test hint_alias_response isa ModelContextProtocol.JSONRPCResponse
    @test occursin("calibration_stage2", hint_alias_response.result.contents[1]["text"])

    hint_direct_response = ModelContextProtocol.handle_request(
        server,
        ModelContextProtocol.JSONRPCRequest(
            id = 106,
            method = "resources/read",
            params = ModelContextProtocol.ReadResourceParams(uri = "hydro://hints/run_simulation"),
        ),
    )
    @test hint_direct_response isa ModelContextProtocol.JSONRPCResponse
    @test occursin("simulation_v2", hint_direct_response.result.contents[1]["text"])

    workflow_template_response = ModelContextProtocol.handle_request(
        server,
        ModelContextProtocol.JSONRPCRequest(
            id = 107,
            method = "resources/read",
            params = ModelContextProtocol.ReadResourceParams(uri = "hydro://workflows/calibrate"),
        ),
    )
    @test workflow_template_response isa ModelContextProtocol.JSONRPCResponse
    @test occursin("auto_calibration_workflow", workflow_template_response.result.contents[1]["text"])

    template_payload = HydroModelMCP.resource_templates_resource.data_provider()
    @test template_payload isa Dict
    @test !isempty(template_payload["templates"])
    @test any(t -> t["name"] == "model_info", template_payload["templates"])
    @test any(t -> t["name"] == "model_knowledge", template_payload["templates"])
    @test any(t -> t["name"] == "llm_hint", template_payload["templates"])
    @test any(t -> t["name"] == "workflow_playbook", template_payload["templates"])
    @test any(t -> t["name"] == "ensemble_result", template_payload["templates"])

    hint_catalog_payload = HydroModelMCP.llm_hint_catalog_payload()
    @test hint_catalog_payload["count"] >= 1
    @test any(s -> s["feature"] == "calibration_stage2", hint_catalog_payload["features"])

    quickstart_payload = HydroModelMCP.llm_quickstart_payload()
    @test quickstart_payload["target_tool_map"]["simulation"] == "run_simulation"
    @test quickstart_payload["target_tool_map"]["calibration"] == "auto_calibration_workflow"

    mcp_surface_payload = JSON3.read(HydroModelMCP.list_mcp_surfaces_tool.handler(Dict()).text, Dict{String,Any})
    @test mcp_surface_payload["status"] == "success"
    @test haskey(mcp_surface_payload, "resource_templates")
    @test any(==("hydro://hints/{feature}"), mcp_surface_payload["resource_templates"])
    @test haskey(mcp_surface_payload, "prompts")
    @test any(==("runoff_workspace_workflow"), mcp_surface_payload["prompts"])
    @test haskey(mcp_surface_payload, "storage_result_tools")
    @test any(==("list_stored_results"), mcp_surface_payload["storage_result_tools"])
    @test any(==("get_stored_result"), mcp_surface_payload["storage_result_tools"])
    @test haskey(mcp_surface_payload, "llm_quickstart")
    @test mcp_surface_payload["llm_quickstart"]["resource"] == "hydro://guides/llm-quickstart"
    @test any(==("auto_calibration_workflow"), mcp_surface_payload["llm_quickstart"]["auto_workflow_tools"])

    expert_text = first(HydroModelMCP.Experts.hydro_expert_prompt.messages).content.text
    processed = ModelContextProtocol.process_template(expert_text, Dict("task" => "Assess runoff simulation", "context" => "Short record"))
    @test occursin("Assess runoff simulation", processed)
    @test occursin("Short record", processed)
    @test !occursin("{task}", processed)

    no_context = ModelContextProtocol.process_template(expert_text, Dict("task" => "Assess runoff simulation"))
    @test occursin("Assess runoff simulation", no_context)
    @test !occursin("Context:", no_context)
end

@testset "Dynamic Stored Result Resources" begin
    mktempdir() do tmpdir
        backend = HydroModelMCP.Storage.FileBackend(tmpdir; ttl = 0)
        HydroModelMCP.Storage.save_result(
            backend,
            "ensemble",
            "demo-result",
            Dict("result_id" => "demo-result", "n_members" => 2),
        )

        resources = HydroModelMCP.build_resources(backend)
        uris = [string(resource.uri) for resource in resources]

        @test "hydro://ensemble/results" in uris
        @test "hydro://ensemble/results/demo-result" in uris

        idx = findfirst(==("hydro://ensemble/results/demo-result"), uris)
        @test !isnothing(idx)
        payload = resources[idx].data_provider()
        @test payload["result_id"] == "demo-result"
        @test payload["n_members"] == 2
    end
end
