using .HydroModelMCP
using .HydroModelMCP.Discovery

@testset "Server Assembly Tests" begin
    server = HydroModelMCP.build_server()
    model_names = Discovery.list_models()

    tool_names = [tool.name for tool in server.tools]
    resource_uris = [string(resource.uri) for resource in server.resources]
    prompt_names = [prompt.name for prompt in server.prompts]

    @test "list_models" in tool_names
    @test "list_workspace_files" in tool_names
    @test "clear_session_cache" in tool_names
    @test "hydrology_expert_review" in prompt_names

    @test "hydro://models/catalog" in resource_uris
    @test "hydro://guides/algorithms" in resource_uris
    @test "hydro://guides/objectives" in resource_uris
    @test "hydro://meta/resource-templates" in resource_uris
    @test "hydro://calibration/results" in resource_uris

    first_model = first(model_names)
    @test "hydro://models/$first_model/info" in resource_uris
    @test "hydro://models/$first_model/parameters" in resource_uris
    @test "hydro://models/$first_model/variables" in resource_uris

    @test length(resource_uris) == 5 + 3 * length(model_names)

    catalog_payload = HydroModelMCP.model_catalog_resource.data_provider()
    @test catalog_payload isa Dict
    @test catalog_payload["count"] == length(model_names)
    @test haskey(first(catalog_payload["models"]), "variables_uri")

    template_payload = HydroModelMCP.resource_templates_resource.data_provider()
    @test template_payload isa Dict
    @test !isempty(template_payload["templates"])
end
