using CSV
using DataFrames
using .HydroModelMCP.Ensemble

function generate_ensemble_test_data()
    data_file = joinpath(dirname(@__DIR__), "data", "03604000.csv")
    df = CSV.File(data_file) |> DataFrame
    return (P = df[!, "prcp(mm/day)"], T = df[!, "tmean(C)"], Ep = df[!, "pet(mm)"])
end

const TEST_MODEL_NAME = "exphydro"

function _ensemble_request(param_sets, forcing_spec)
    return Dict(
        "model" => TEST_MODEL_NAME,
        "inputs" => Dict(
            "forcing" => forcing_spec,
        ),
        "parameter_sets" => param_sets,
        "parallel" => false,
        "save_to_storage" => false,
    )
end

@testset "Ensemble Simulation Tests" begin
    test_data = generate_ensemble_test_data()

    param_sets = [
        Dict("f" => 0.001, "Smax" => 1000.0, "Qmax" => 20.0, "Df" => 2.5, "Tmax" => 1.5, "Tmin" => -1.0),
        Dict("f" => 0.002, "Smax" => 1200.0, "Qmax" => 22.0, "Df" => 2.8, "Tmax" => 1.8, "Tmin" => -0.8),
        Dict("f" => 0.0015, "Smax" => 1100.0, "Qmax" => 21.0, "Df" => 2.6, "Tmax" => 1.6, "Tmin" => -0.9),
    ]

    @testset "Case 1: Ensemble with JSON Input" begin
        payload = _ensemble_request(param_sets, Dict(
            "source_type" => "json",
            "data" => Dict(
                "P" => test_data.P,
                "T" => test_data.T,
                "Ep" => test_data.Ep,
            ),
        ))

        response = Ensemble.run_ensemble(payload)

        @test haskey(response, "ensemble_members")
        @test length(response["ensemble_members"]) == length(param_sets)
        @test haskey(response, "ensemble_mean")
        @test haskey(response, "ensemble_std")
        @test length(response["ensemble_mean"]) == length(test_data.P)
        @test length(response["ensemble_std"]) == length(test_data.P)
    end

    @testset "Case 2: Parallel Ensemble Execution" begin
        payload = _ensemble_request(param_sets, Dict(
            "source_type" => "json",
            "data" => Dict(
                "P" => test_data.P,
                "T" => test_data.T,
                "Ep" => test_data.Ep,
            ),
        ))
        payload["parallel"] = true

        response = Ensemble.run_ensemble(payload)

        @test haskey(response, "ensemble_members")
        @test length(response["ensemble_members"]) == length(param_sets)
        @test haskey(response, "ensemble_mean")
        @test haskey(response, "ensemble_std")
    end

    @testset "Case 3: Ensemble with CSV Input" begin
        temp_dir = mktempdir()
        csv_path = joinpath(temp_dir, "test_forcing.csv")

        df = DataFrame(P = test_data.P, Ep = test_data.Ep, T = test_data.T)
        CSV.write(csv_path, df)

        payload = _ensemble_request(param_sets, Dict(
            "source_type" => "csv",
            "path" => csv_path,
        ))

        response = Ensemble.run_ensemble(payload)

        @test haskey(response, "ensemble_members")
        @test length(response["ensemble_members"]) == length(param_sets)

        rm(temp_dir, recursive = true, force = true)
    end

    @testset "Case 4: Single Parameter Set" begin
        payload = _ensemble_request([param_sets[1]], Dict(
            "source_type" => "json",
            "data" => Dict(
                "P" => test_data.P,
                "T" => test_data.T,
                "Ep" => test_data.Ep,
            ),
        ))

        response = Ensemble.run_ensemble(payload)

        @test length(response["ensemble_members"]) == 1
        @test haskey(response, "ensemble_mean")
    end
end
