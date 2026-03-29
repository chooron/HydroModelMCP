using CSV
using DataFrames
using .HydroModelMCP.Ensemble
using .HydroModelMCP.DataLoader

function generate_ensemble_test_data()
    data_file = joinpath(dirname(@__DIR__), "data", "03604000.csv")
    df = CSV.File(data_file) |> DataFrame
    return (P = df[!, "prcp(mm/day)"], T = df[!, "tmean(C)"], Ep = df[!, "pet(mm)"])
end

const TEST_MODEL_NAME = "exphydro"

@testset "Ensemble Simulation Tests" begin
    test_data = generate_ensemble_test_data()

    param_sets = [
        Dict("f" => 0.001, "Smax" => 1000.0, "Qmax" => 20.0, "Df" => 2.5, "Tmax" => 1.5, "Tmin" => -1.0),
        Dict("f" => 0.002, "Smax" => 1200.0, "Qmax" => 22.0, "Df" => 2.8, "Tmax" => 1.8, "Tmin" => -0.8),
        Dict("f" => 0.0015, "Smax" => 1100.0, "Qmax" => 21.0, "Df" => 2.6, "Tmax" => 1.6, "Tmin" => -0.9)
    ]

    @testset "Case 1: Ensemble with JSON Input" begin
        println("\n>>> Testing Ensemble with JSON Input...")

        payload = Dict(
            "model_name" => TEST_MODEL_NAME,
            "parameter_sets" => param_sets,
            "forcing" => Dict(
                "source_type" => "json",
                "data" => Dict(
                    "P" => test_data.P,
                    "T" => test_data.T,
                    "Ep" => test_data.Ep
                )
            ),
            "parallel" => false,
            "save_to_storage" => false
        )

        response = Ensemble.run_ensemble(payload)

        @test haskey(response, "ensemble_members")
        @test length(response["ensemble_members"]) == length(param_sets)
        @test haskey(response, "ensemble_mean")
        @test haskey(response, "ensemble_std")
        @test length(response["ensemble_mean"]) == length(test_data.P)
        @test length(response["ensemble_std"]) == length(test_data.P)

        println("   [Pass] Ensemble simulation successful.")
        println("   -> Number of members: $(length(response["ensemble_members"]))")
        println("   -> Result length: $(length(response["ensemble_mean"]))")
    end

    @testset "Case 2: Parallel Ensemble Execution" begin
        println("\n>>> Testing Parallel Ensemble Execution...")

        payload = Dict(
            "model_name" => TEST_MODEL_NAME,
            "parameter_sets" => param_sets,
            "forcing" => Dict(
                "source_type" => "json",
                "data" => Dict(
                    "P" => test_data.P,
                    "T" => test_data.T,
                    "Ep" => test_data.Ep
                )
            ),
            "parallel" => true,
            "save_to_storage" => false
        )

        response = Ensemble.run_ensemble(payload)

        @test haskey(response, "ensemble_members")
        @test length(response["ensemble_members"]) == length(param_sets)
        @test haskey(response, "ensemble_mean")
        @test haskey(response, "ensemble_std")

        println("   [Pass] Parallel ensemble execution successful.")
    end

    @testset "Case 3: Ensemble with CSV Input" begin
        println("\n>>> Testing Ensemble with CSV Input...")

        temp_dir = mktempdir()
        csv_path = joinpath(temp_dir, "test_forcing.csv")

        df = DataFrame(P = test_data.P, Ep = test_data.Ep, T = test_data.T)
        CSV.write(csv_path, df)

        payload = Dict(
            "model_name" => TEST_MODEL_NAME,
            "parameter_sets" => param_sets,
            "forcing" => Dict(
                "source_type" => "csv",
                "path" => csv_path
            ),
            "parallel" => false,
            "save_to_storage" => false
        )

        response = Ensemble.run_ensemble(payload)

        @test haskey(response, "ensemble_members")
        @test length(response["ensemble_members"]) == length(param_sets)

        println("   [Pass] CSV ensemble simulation successful.")
        rm(temp_dir, recursive = true, force = true)
    end

    @testset "Case 4: Single Parameter Set" begin
        println("\n>>> Testing Single Parameter Set...")

        payload = Dict(
            "model_name" => TEST_MODEL_NAME,
            "parameter_sets" => [param_sets[1]],
            "forcing" => Dict(
                "source_type" => "json",
                "data" => Dict(
                    "P" => test_data.P,
                    "T" => test_data.T,
                    "Ep" => test_data.Ep
                )
            ),
            "save_to_storage" => false
        )

        response = Ensemble.run_ensemble(payload)

        @test length(response["ensemble_members"]) == 1
        @test haskey(response, "ensemble_mean")

        println("   [Pass] Single parameter set handled correctly.")
    end
end
