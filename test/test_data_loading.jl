using JSON3
using CSV
using DataFrames
using Dates
using NCDatasets
using .HydroModelMCP

function _parse_data_loading_json(response)
    text = response.text
    startswith(text, "Error:") && error(text)
    return JSON3.read(text, Dict{String,Any})
end

function _write_mock_caravan_dataset_root(root::String; dataset_name::String = "camels", gauge_id::String = "01013500", n_steps::Int = 40)
    canonical_gauge_id = "$(dataset_name)_$(gauge_id)"
    netcdf_dir = joinpath(root, "timeseries", "netcdf", dataset_name)
    attributes_dir = joinpath(root, "attributes", dataset_name)
    mkpath(netcdf_dir)
    mkpath(attributes_dir)

    nc_path = joinpath(netcdf_dir, canonical_gauge_id * ".nc")
    ds = NCDataset(nc_path, "c")
    try
        defDim(ds, "date", n_steps)

        date_var = defVar(ds, "date", Int32, ("date",))
        date_var.attrib["units"] = "days since 2001-01-01 00:00:00"
        date_var[:] = Int32.(0:n_steps-1)

        p = Float32.(1.0 .+ 0.1 .* collect(1:n_steps))
        t = Float32.(5.0 .+ 0.05 .* collect(1:n_steps))
        ep = Float32.(0.4 .+ 0.01 .* collect(1:n_steps))
        q = Float32.(2.0 .+ 0.08 .* collect(1:n_steps))
        q[3] = NaN32

        defVar(ds, "total_precipitation_sum", p, ("date",))
        defVar(ds, "temperature_2m_mean", t, ("date",))
        defVar(ds, "potential_evaporation_sum", ep, ("date",))
        defVar(ds, "streamflow", q, ("date",))

        ds.attrib["Timezone"] = "UTC"
        ds.attrib["Units"] = "total_precipitation: Total precipitation [mm]"
    finally
        close(ds)
    end

    CSV.write(joinpath(attributes_dir, "attributes_other_$(dataset_name).csv"), DataFrame(
        gauge_id = [canonical_gauge_id],
        gauge_lat = [44.0],
        gauge_lon = [-72.0],
        gauge_name = ["Mock gauge"],
        country = ["US"],
        area = [123.4],
    ))

    return root, nc_path, canonical_gauge_id
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

    @testset "inspect_hydro_data supports data_handle source" begin
        loaded = _parse_data_loading_json(HydroModelMCP.load_hydro_csv_tool.handler(Dict(
            "path" => source_file,
            "data_type" => "forcing",
            "handle_name" => "inspect_handle_03604000",
        )))

        payload = _parse_data_loading_json(HydroModelMCP.inspect_hydro_data_tool.handler(Dict(
            "source" => Dict(
                "source_type" => "data_handle",
                "data_handle" => loaded["data_handle"],
            ),
            "model" => "exphydro",
            "intended_use" => "calibration",
        )))

        @test payload["status"] == "success"
        @test payload["source"]["source_type"] == "data_handle"
        @test payload["source"]["data_handle"] == loaded["data_handle"]
        @test payload["forcing_elements"]["sufficient"] == true
        @test payload["observed_runoff"]["present"] == true
        @test payload["readiness"]["ready_for_calibration"] == true
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

    @testset "load_camels_data returns retirement error" begin
        base = joinpath(dirname(@__DIR__), ".tmp_tests")
        mkpath(base)
        mktempdir(base; prefix = "camels-retired-") do _
            response = HydroModelMCP.load_camels_data_tool.handler(Dict(
                "gage_id" => "01013500",
                "dataset_name" => "camels",
            ))

            @test startswith(response.text, "Error:")
            @test occursin("retired", lowercase(response.text))
        end
    end

    @testset "inspect_hydro_data rejects retired camels source" begin
        base = joinpath(dirname(@__DIR__), ".tmp_tests")
        mkpath(base)
        mktempdir(base; prefix = "camels-inspect-retired-") do _
            response = HydroModelMCP.inspect_hydro_data_tool.handler(Dict(
                "source" => Dict(
                    "source_type" => "camels",
                    "gage_id" => "01013500",
                ),
                "model" => "exphydro",
                "intended_use" => "calibration",
            ))

            @test startswith(response.text, "Error:")
            @test occursin("retired", lowercase(response.text))
        end
    end

    @testset "load_caravan_data creates calibration-ready handle" begin
        base = joinpath(dirname(@__DIR__), ".tmp_tests")
        mkpath(base)
        mktempdir(base; prefix = "caravan-load-") do tmpdir
            caravan_root, _, canonical_gauge_id = _write_mock_caravan_dataset_root(tmpdir)

            payload = _parse_data_loading_json(HydroModelMCP.load_caravan_data_tool.handler(Dict(
                "dataset_root" => caravan_root,
                "dataset_name" => "camels",
                "gauge_id" => "01013500",
            )))

            @test payload["status"] == "success"
            @test payload["metadata"]["rows"] == 39
            @test payload["metadata"]["gauge_id"] == canonical_gauge_id
            @test payload["metadata"]["dataset_name"] == "camels"
            @test payload["metadata"]["units"] == "mm/day"

            handle = String(payload["data_handle"])
            stored = HydroModelMCP.get_data(handle)
            @test stored isa Dict
            @test stored["dataset_name"] == "camels"
            @test stored["gauge_id"] == canonical_gauge_id
            @test stored["obs_units"] == "mm/day"
            @test length(stored["obs"]) == payload["metadata"]["rows"]
        end
    end

    @testset "load_caravan_data requires dataset_name" begin
        base = joinpath(dirname(@__DIR__), ".tmp_tests")
        mkpath(base)
        mktempdir(base; prefix = "caravan-requires-dataset-") do tmpdir
            caravan_root, _, _ = _write_mock_caravan_dataset_root(tmpdir)

            response = HydroModelMCP.load_caravan_data_tool.handler(Dict(
                "dataset_root" => caravan_root,
                "gauge_id" => "01013500",
            ))

            @test startswith(response.text, "Error:")
            @test occursin("dataset_name", response.text)
        end
    end

    @testset "load_caravan_data returns not found without CSV fallback" begin
        base = joinpath(dirname(@__DIR__), ".tmp_tests")
        mkpath(base)
        mktempdir(base; prefix = "caravan-not-found-") do tmpdir
            caravan_root, _, _ = _write_mock_caravan_dataset_root(tmpdir)

            response = HydroModelMCP.load_caravan_data_tool.handler(Dict(
                "dataset_root" => caravan_root,
                "dataset_name" => "camels",
                "gauge_id" => "99999999",
            ))

            @test startswith(response.text, "Error:")
            @test occursin("not found under dataset 'camels'", response.text)
            @test !occursin("csv", lowercase(response.text))
        end
    end

    @testset "inspect_hydro_data supports caravan source" begin
        base = joinpath(dirname(@__DIR__), ".tmp_tests")
        mkpath(base)
        mktempdir(base; prefix = "caravan-inspect-") do tmpdir
            caravan_root, _, _ = _write_mock_caravan_dataset_root(tmpdir)

            payload = _parse_data_loading_json(HydroModelMCP.inspect_hydro_data_tool.handler(Dict(
                "source" => Dict(
                    "source_type" => "caravan",
                    "dataset_root" => caravan_root,
                    "dataset_name" => "camels",
                    "gauge_id" => "01013500",
                ),
                "model" => "exphydro",
                "intended_use" => "calibration",
            )))

            @test payload["status"] == "success"
            @test payload["source"]["source_type"] == "caravan"
            @test payload["source"]["row_count"] == 39
            @test payload["observed_runoff"]["present"] == true
            @test payload["readiness"]["ready_for_calibration"] == true
            @test payload["model_check"]["sufficient"] == true
        end
    end
end
