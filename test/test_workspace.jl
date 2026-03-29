using JSON3
using .HydroModelMCP

function _workspace_payload(response)
    text = response.text
    startswith(text, "Error:") && error(text)
    return JSON3.read(text, Dict{String,Any})
end

@testset "Workspace Tool Tests" begin
    @testset "list workspace csv files" begin
        payload = _workspace_payload(HydroModelMCP.list_workspace_files_tool.handler(Dict(
            "directory" => ".\\data",
            "extensions" => ["csv"],
        )))

        @test payload["status"] == "success"
        @test payload["count"] > 0
        @test all(endswith(lowercase(String(file["name"])), ".csv") for file in payload["files"])
    end

    @testset "reject path escape outside workspace" begin
        response = HydroModelMCP.list_workspace_files_tool.handler(Dict(
            "directory" => "..",
        ))

        @test occursin("Error:", response.text)
        @test occursin("escapes workspace root", response.text)
    end

    @testset "clear session cache removes intermediate handles" begin
        HydroModelMCP.store_data("workspace-test-handle", Dict("value" => [1.0, 2.0]))
        @test HydroModelMCP.has_data("workspace-test-handle")

        payload = _workspace_payload(HydroModelMCP.clear_session_cache_tool.handler(Dict()))

        @test payload["status"] == "success"
        @test payload["cleared_count"] >= 1
        @test !HydroModelMCP.has_data("workspace-test-handle")
    end
end
