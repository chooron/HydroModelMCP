using CSV
using DataFrames
using Random
using .HydroModelMCP.SensitivityAnalysis

@testset "SensitivityAnalysis Module Tests" begin
    test_dir = @__DIR__
    data_file = joinpath(dirname(test_dir), "data", "03604000.csv")
    df = CSV.File(data_file) |> DataFrame

    forcing_nt = (
        P = df[1:365, "prcp(mm/day)"],
        T = df[1:365, "tmean(C)"],
        Ep = df[1:365, "pet(mm)"]
    )
    obs = df[1:365, "flow(mm)"]
    test_model = "exphydro"

    @testset "Morris Method" begin
        println("\n   -> Running Morris sensitivity analysis (this may take a minute)...")

        result = run_sensitivity(
            test_model, forcing_nt, obs;
            method = "morris",
            n_samples = 20,
            objective = "NSE",
            threshold = 0.1,
            solver_type = "DISCRETE",
            interp_type = "LINEAR"
        )

        @test haskey(result, "model_name")
        @test haskey(result, "param_names")
        @test haskey(result, "sensitivities")
        @test haskey(result, "calibratable")
        @test haskey(result, "fixed")
        @test haskey(result, "method")
        @test haskey(result, "threshold")

        @test result["method"] == "morris"
        @test result["threshold"] == 0.1
        @test result["sensitivities"] isa Vector
        @test length(result["sensitivities"]) == length(result["param_names"])
        @test all(0.0 .<= result["sensitivities"] .<= 1.0)

        @test result["calibratable"] isa Vector{String}
        @test result["fixed"] isa Vector{String}
        @test length(result["calibratable"]) + length(result["fixed"]) == length(result["param_names"])

        println("   -> Model: $(result["model_name"])")
        println("   -> Total params: $(length(result["param_names"]))")
        println("   -> Calibratable: $(length(result["calibratable"])) params")
        println("   -> Fixed: $(length(result["fixed"])) params")
        println("   -> Calibratable params: $(result["calibratable"])")
    end

    @testset "Different Objectives" begin
        result_nse = run_sensitivity(
            test_model, forcing_nt, obs;
            method = "morris",
            n_samples = 15,
            objective = "NSE",
            threshold = 0.15
        )

        result_kge = run_sensitivity(
            test_model, forcing_nt, obs;
            method = "morris",
            n_samples = 15,
            objective = "KGE",
            threshold = 0.15
        )

        @test result_nse["objective"] == "NSE"
        @test result_kge["objective"] == "KGE"

        println("   -> NSE calibratable: $(result_nse["calibratable"])")
        println("   -> KGE calibratable: $(result_kge["calibratable"])")
    end

    @testset "Threshold Monotonicity" begin
        # Morris sampling is stochastic, so use the same seed for both runs.
        Random.seed!(20260327)
        result_low = run_sensitivity(
            test_model, forcing_nt, obs;
            method = "morris",
            n_samples = 15,
            threshold = 0.05
        )

        Random.seed!(20260327)
        result_high = run_sensitivity(
            test_model, forcing_nt, obs;
            method = "morris",
            n_samples = 15,
            threshold = 0.3
        )

        @test result_low["param_names"] == result_high["param_names"]
        @test result_low["sensitivities"] == result_high["sensitivities"]
        @test length(result_high["calibratable"]) <= length(result_low["calibratable"])

        println("   -> Threshold 0.05: $(length(result_low["calibratable"])) calibratable")
        println("   -> Threshold 0.30: $(length(result_high["calibratable"])) calibratable")
    end

    @testset "Invalid Model" begin
        @test_throws ArgumentError run_sensitivity(
            "NonExistentModel", forcing_nt, obs;
            method = "morris",
            n_samples = 10
        )
        println("   [Pass] Invalid model throws ArgumentError as expected")
    end

    @testset "Invalid Method" begin
        @test_throws ArgumentError run_sensitivity(
            test_model, forcing_nt, obs;
            method = "invalid_method",
            n_samples = 10
        )
        println("   [Pass] Invalid method throws ArgumentError as expected")
    end

    @testset "Custom Parameter Bounds" begin
        println("\n   -> Testing param_bounds_override...")

        result_default = run_sensitivity(
            test_model, forcing_nt, obs;
            method = "morris",
            n_samples = 15,
            threshold = 0.1
        )

        first_param = result_default["param_names"][1]
        default_sens = result_default["sensitivities"][1]

        pb_override = Dict{String,Vector{Float64}}(
            first_param => [0.0, 0.001]
        )

        result_override = run_sensitivity(
            test_model, forcing_nt, obs;
            method = "morris",
            n_samples = 15,
            threshold = 0.1,
            param_bounds_override = pb_override
        )

        @test haskey(result_override, "param_names")
        @test haskey(result_override, "sensitivities")
        @test length(result_override["param_names"]) == length(result_default["param_names"])

        println("   -> Default sensitivity[$(first_param)]: $(round(default_sens, digits = 4))")
        println("   -> Override sensitivity[$(first_param)]: $(round(result_override["sensitivities"][1], digits = 4))")
    end

    @testset "Parameter Subset Filter" begin
        println("\n   -> Testing param_names_filter...")

        result_all = run_sensitivity(
            test_model, forcing_nt, obs;
            method = "morris",
            n_samples = 15,
            threshold = 0.1
        )

        all_pnames = result_all["param_names"]
        if length(all_pnames) >= 2
            filter_pnames = all_pnames[1:2]
            result_subset = run_sensitivity(
                test_model, forcing_nt, obs;
                method = "morris",
                n_samples = 15,
                threshold = 0.1,
                param_names_filter = filter_pnames
            )

            @test length(result_subset["param_names"]) == 2
            @test result_subset["param_names"] == filter_pnames
            @test length(result_subset["sensitivities"]) == 2

            println("   -> Subset params: $(result_subset["param_names"])")
        else
            println("   -> Skipped: model has fewer than 2 parameters")
        end
    end

    println("\n   [Pass] All SensitivityAnalysis tests passed successfully!")
end
