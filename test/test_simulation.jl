using JSON3
using .HydroModelMCP
using .HydroModelMCP.Simulation

function _workspace_tempdir(prefix::String)
    base = joinpath(dirname(@__DIR__), ".tmp_tests")
    mkpath(base)
    return mktempdir(base; prefix = prefix)
end

function _parse_tool_json(response)
    text = response.text
    startswith(text, "Error:") && error(text)
    return JSON3.read(text, Dict{String,Any})
end

@testset "Simulation Tool Contract Tests" begin
    source_file = joinpath(dirname(@__DIR__), "data", "03604000.csv")
    result_dir = _workspace_tempdir("simulation-")

    @testset "Normalization defaults" begin
        normalized = HydroModelMCP._normalize_simulation_request(Dict(
            "model" => "exphydro",
            "source_type" => "csv",
            "path" => ".\\data\\03604000.csv",
        ))

        @test normalized["model"] == "exphydro"
        @test normalized["source_type"] == "csv"
        @test normalized["path"] == source_file
        @test normalized["output_dir"] == joinpath(dirname(@__DIR__), "result")
    end

    @testset "run_simulation writes result artifacts and exposes random params" begin
        payload = _parse_tool_json(HydroModelMCP.simulation_tool.handler(Dict(
            "model" => "exphydro",
            "source_type" => "csv",
            "path" => source_file,
            "output_dir" => result_dir,
            "seed" => 1234,
        )))

        @test payload["status"] == "success"
        @test payload["params_source"] == "random"
        @test payload["params_seed"] == 1234
        @test payload["params_used"] isa Dict
        @test isfile(payload["output_path"])
        @test isfile(payload["metadata_path"])
        @test isfile(payload["summary_path"])
        @test any(occursin("Mapped input", String(w)) for w in payload["warnings"])
    end

    @testset "random params are reproducible for the same seed" begin
        first_run = Simulation.run_simulation(Dict(
            "model" => "exphydro",
            "source_type" => "csv",
            "path" => source_file,
            "output_dir" => result_dir,
            "seed" => 7,
        ))
        second_run = Simulation.run_simulation(Dict(
            "model" => "exphydro",
            "source_type" => "csv",
            "path" => source_file,
            "output_dir" => result_dir,
            "seed" => 7,
        ))

        @test first_run["params_used"] == second_run["params_used"]
    end

    @testset "compute_metrics accepts direct file paths and writes metrics artifacts" begin
        simulation_payload = Simulation.run_simulation(Dict(
            "model" => "exphydro",
            "source_type" => "csv",
            "path" => source_file,
            "output_dir" => result_dir,
            "seed" => 99,
        ))

        payload = _parse_tool_json(HydroModelMCP.compute_metrics_tool.handler(Dict(
            "simulated" => Dict(
                "source_type" => "csv",
                "path" => simulation_payload["output_path"],
            ),
            "observed" => Dict(
                "source_type" => "csv",
                "path" => source_file,
            ),
            "output_dir" => result_dir,
            "metrics" => JSON3.read("[\"KGE\",\"NSE\",\"RMSE\",\"MAE\",\"Bias\"]"),
        )))

        @test payload["status"] == "success"
        @test payload["metrics"] isa Dict
        @test haskey(payload["metrics"], "NSE")
        @test haskey(payload["metrics"], "KGE")
        @test haskey(payload["metrics"], "MAE")
        @test haskey(payload["metrics"], "Bias")
        @test payload["sample_size"] > 0
        @test isfile(payload["output_path"])
        @test any(occursin("Inferred simulated column", String(w)) for w in payload["warnings"])
        @test any(occursin("Inferred observed column", String(w)) for w in payload["warnings"])

        metrics_dir_payload = _parse_tool_json(HydroModelMCP.compute_metrics_tool.handler(Dict(
            "simulated" => Dict(
                "source_type" => "csv",
                "path" => simulation_payload["output_path"],
            ),
            "observed" => Dict(
                "source_type" => "csv",
                "path" => source_file,
            ),
            "output_dir" => joinpath(result_dir, "metrics"),
            "metrics" => JSON3.read("[\"NSE\"]"),
        )))

        metrics_output_path = String(metrics_dir_payload["output_path"])
        @test !occursin("metrics\\metrics", lowercase(metrics_output_path))
    end
end
