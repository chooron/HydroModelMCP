using JSON3
using CSV
using DataFrames
using NPZ
using .HydroModelMCP

function _parse_data_loading_json(response)
    text = response.text
    startswith(text, "Error:") && error(text)
    return JSON3.read(text, Dict{String,Any})
end

function _write_mock_camels_dataset(path::String; gage_ids = [1013500, 3604000], n_steps::Int = 40)
    n_gauges = length(gage_ids)
    forcings = zeros(Float64, n_gauges, n_steps, 3)
    target = zeros(Float64, n_gauges, n_steps, 1)
    attributes = ones(Float64, n_gauges, 13)

    for (gauge_idx, _) in enumerate(gage_ids)
        for step in 1:n_steps
            forcings[gauge_idx, step, 1] = 1.0 + 0.02 * step + 0.1 * gauge_idx
            forcings[gauge_idx, step, 2] = 3.0 + 0.01 * step
            forcings[gauge_idx, step, 3] = 0.4 + 0.005 * step
            target[gauge_idx, step, 1] = 2.0 + 0.03 * step + 0.2 * gauge_idx
        end

        attributes[gauge_idx, 12] = 100.0 + 10.0 * gauge_idx
        attributes[gauge_idx, 13] = 120.0 + 10.0 * gauge_idx
    end

    npzwrite(path, Dict{String,Any}(
        "gage_ids" => collect(gage_ids),
        "forcings" => forcings,
        "target" => target,
        "attributes" => attributes,
    ))

    return path
end

@testset "Data Inspection Tool Tests" begin
    source_file = joinpath(dirname(@__DIR__), "data", "03604000.csv")

    @testset "inspect_hydro_data detects forcing and observed runoff" begin
        payload = _parse_data_loading_json(HydroModelMCP.inspect_hydro_data_tool.handler(Dict(
            "source" => Dict(
                "source_type" => "csv",
                "path" => source_file,
            ),
            "model" => "exphydro",
            "intended_use" => "calibration",
        )))

        @test payload["status"] == "success"
        @test payload["source"]["source_type"] == "csv"
        @test payload["source"]["row_count"] > 0
        @test payload["forcing_elements"]["sufficient"] == true
        @test payload["observed_runoff"]["present"] == true
        @test payload["readiness"]["ready_for_calibration"] == true
        @test payload["model_check"]["resolved_model"] == "exphydro"
        @test payload["model_check"]["sufficient"] == true
    end

    @testset "analyze_distribution_from_handle supports composite hydro handles" begin
        loaded = _parse_data_loading_json(HydroModelMCP.load_hydro_csv_tool.handler(Dict(
            "path" => source_file,
            "data_type" => "forcing",
            "handle_name" => "distribution_test_03604000",
        )))

        payload = _parse_data_loading_json(HydroModelMCP.analyze_distribution_from_handle_tool.handler(Dict(
            "data_handle" => loaded["data_handle"],
        )))

        @test payload["status"] == "success"
        @test payload["series_name"] == "flow(mm)"
        @test payload["num_observations"] > 0
        @test payload["magnitude_ratio"] >= 1.0
    end

    @testset "inspect_hydro_data reports missing PET and runoff for calibration" begin
        base = joinpath(dirname(@__DIR__), ".tmp_tests")
        mkpath(base)
        mktempdir(base; prefix = "inspect-") do tmpdir
            csv_path = joinpath(tmpdir, "forcing_only.csv")
            CSV.write(csv_path, DataFrame(
                :date => ["2001-01-01", "2001-01-02", "2001-01-03"],
                Symbol("prcp(mm/day)") => [1.0, 0.0, 2.5],
                Symbol("tmean(C)") => [8.0, 7.5, 9.0],
            ))

            payload = _parse_data_loading_json(HydroModelMCP.inspect_hydro_data_tool.handler(Dict(
                "source" => Dict(
                    "source_type" => "csv",
                    "path" => csv_path,
                ),
                "model" => "exphydro",
                "intended_use" => "calibration",
            )))

            @test payload["forcing_elements"]["sufficient"] == false
            @test "Ep" in payload["forcing_elements"]["missing_inputs"]
            @test payload["observed_runoff"]["present"] == false
            @test payload["readiness"]["ready_for_requested_use"] == false
            @test payload["readiness"]["ready_for_calibration"] == false
            @test any(occursin("Observed runoff", String(w)) for w in payload["warnings"])
            @test any(occursin("input_mapping", String(r)) for r in payload["recommendations"])
        end
    end

    @testset "inspect_hydro_data does not block simulation without model-specific check" begin
        base = joinpath(dirname(@__DIR__), ".tmp_tests")
        mkpath(base)
        mktempdir(base; prefix = "inspect-") do tmpdir
            csv_path = joinpath(tmpdir, "forcing_only.csv")
            CSV.write(csv_path, DataFrame(
                :date => ["2001-01-01", "2001-01-02", "2001-01-03"],
                Symbol("prcp(mm/day)") => [1.0, 0.0, 2.5],
                Symbol("tmean(C)") => [8.0, 7.5, 9.0],
            ))

            payload = _parse_data_loading_json(HydroModelMCP.inspect_hydro_data_tool.handler(Dict(
                "source" => Dict(
                    "source_type" => "csv",
                    "path" => csv_path,
                ),
                "intended_use" => "simulation",
            )))

            @test payload["forcing_elements"]["sufficient"] == false
            @test payload["readiness"]["model_checked"] == false
            @test payload["readiness"]["ready_for_requested_use"] == true
            @test isempty(payload["blocking_issues"])
        end
    end

    @testset "load_camels_data creates calibration-ready handle" begin
        base = joinpath(dirname(@__DIR__), ".tmp_tests")
        mkpath(base)
        mktempdir(base; prefix = "camels-load-") do tmpdir
            npz_path = _write_mock_camels_dataset(joinpath(tmpdir, "mock_camels.npz"))

            payload = _parse_data_loading_json(HydroModelMCP.load_camels_data_tool.handler(Dict(
                "dataset_path" => npz_path,
                "gage_id" => 1013500,
            )))

            @test payload["status"] == "success"
            @test payload["metadata"]["rows"] > 0

            handle = String(payload["data_handle"])
            stored = HydroModelMCP.get_data(handle)
            @test stored isa Dict
            @test haskey(stored, "forcing_nt")
            @test haskey(stored, "obs")
            @test stored["forcing_nt"] isa NamedTuple
            @test length(stored["obs"]) == payload["metadata"]["rows"]
        end
    end

    @testset "load_camels_data resolves dataset path from env" begin
        base = joinpath(dirname(@__DIR__), ".tmp_tests")
        mkpath(base)
        mktempdir(base; prefix = "camels-env-") do tmpdir
            npz_path = _write_mock_camels_dataset(joinpath(tmpdir, "mock_camels_env.npz"))

            old_camesl = get(ENV, "CAMESL_DATASET_PATH", nothing)
            old_camels = get(ENV, "CAMELS_DATASET_PATH", nothing)

            try
                ENV["CAMESL_DATASET_PATH"] = npz_path
                haskey(ENV, "CAMELS_DATASET_PATH") && delete!(ENV, "CAMELS_DATASET_PATH")

                payload = _parse_data_loading_json(HydroModelMCP.load_camels_data_tool.handler(Dict(
                    "gage_id" => 3604000,
                )))

                @test payload["status"] == "success"
                @test payload["metadata"]["dataset_path"] == npz_path
            finally
                if old_camesl === nothing
                    haskey(ENV, "CAMESL_DATASET_PATH") && delete!(ENV, "CAMESL_DATASET_PATH")
                else
                    ENV["CAMESL_DATASET_PATH"] = old_camesl
                end

                if old_camels === nothing
                    haskey(ENV, "CAMELS_DATASET_PATH") && delete!(ENV, "CAMELS_DATASET_PATH")
                else
                    ENV["CAMELS_DATASET_PATH"] = old_camels
                end
            end
        end
    end

    @testset "inspect_hydro_data supports camels source" begin
        base = joinpath(dirname(@__DIR__), ".tmp_tests")
        mkpath(base)
        mktempdir(base; prefix = "camels-inspect-") do tmpdir
            npz_path = _write_mock_camels_dataset(joinpath(tmpdir, "mock_camels_inspect.npz"))

            payload = _parse_data_loading_json(HydroModelMCP.inspect_hydro_data_tool.handler(Dict(
                "source" => Dict(
                    "source_type" => "camels",
                    "dataset_path" => npz_path,
                    "gage_id" => 1013500,
                ),
                "model" => "exphydro",
                "intended_use" => "calibration",
            )))

            @test payload["status"] == "success"
            @test payload["source"]["source_type"] == "camels"
            @test payload["source"]["gage_id"] == 1013500
            @test payload["forcing_elements"]["sufficient"] == true
            @test payload["observed_runoff"]["present"] == true
            @test payload["readiness"]["ready_for_calibration"] == true
            @test payload["model_check"]["sufficient"] == true
        end
    end
end
