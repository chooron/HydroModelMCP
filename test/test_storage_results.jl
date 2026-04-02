using JSON3
using UUIDs

function _parse_storage_tool_json(response)
    text = response.text
    startswith(text, "Error:") && error(text)
    return JSON3.read(text, Dict{String,Any})
end

@testset "Stored Result Tool Tests" begin
    calibration_id = "test-calibration-$(uuid4())"
    sensitivity_id = "test-sensitivity-$(uuid4())"

    HydroModelMCP.Storage.save_result(
        HydroModelMCP.STORAGE_BACKEND,
        "calibration",
        calibration_id,
        Dict("marker" => "calibration", "value" => 1),
    )
    HydroModelMCP.Storage.save_result(
        HydroModelMCP.STORAGE_BACKEND,
        "sensitivity",
        sensitivity_id,
        Dict("marker" => "sensitivity", "value" => 2),
    )

    @testset "list_stored_results supports category filter" begin
        payload = _parse_storage_tool_json(HydroModelMCP.list_stored_results_tool.handler(Dict(
            "category" => "calibration",
        )))

        @test payload["status"] == "success"
        @test payload["category"] == "calibration"
        @test calibration_id in String.(payload["result_ids"])
    end

    @testset "list_stored_results supports all categories" begin
        payload = _parse_storage_tool_json(HydroModelMCP.list_stored_results_tool.handler(Dict()))

        @test payload["status"] == "success"
        @test haskey(payload, "categories")
        @test haskey(payload, "total_count")

        categories = Dict(String(item["category"]) => item for item in payload["categories"])
        @test haskey(categories, "calibration")
        @test haskey(categories, "sensitivity")
        @test haskey(categories, "ensemble")
    end

    @testset "get_stored_result supports explicit category" begin
        payload = _parse_storage_tool_json(HydroModelMCP.get_stored_result_tool.handler(Dict(
            "category" => "calibration",
            "result_id" => calibration_id,
        )))

        @test payload["status"] == "success"
        @test payload["storage_category"] == "calibration"
        @test payload["result_id"] == calibration_id
        @test payload["marker"] == "calibration"
        @test payload["value"] == 1
    end

    @testset "get_stored_result supports cross-category lookup" begin
        payload = _parse_storage_tool_json(HydroModelMCP.get_stored_result_tool.handler(Dict(
            "result_id" => sensitivity_id,
        )))

        @test payload["status"] == "success"
        @test payload["storage_category"] == "sensitivity"
        @test payload["result_id"] == sensitivity_id
        @test payload["marker"] == "sensitivity"
        @test payload["value"] == 2
    end
end
