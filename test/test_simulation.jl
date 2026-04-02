using JSON3
using Redis
import UUIDs
using .HydroModelMCP
using .HydroModelMCP.Simulation
using .HydroModelMCP.Validation

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

function _sim_request(source_file::String; model::String = "exphydro", output_dir = nothing, seed = nothing)
    req = Dict{String,Any}(
        "model" => model,
        "inputs" => Dict{String,Any}(
            "forcing" => Dict{String,Any}(
                "source_type" => "csv",
                "path" => source_file,
            ),
        ),
    )

    if !(output_dir === nothing)
        req["output"] = Dict{String,Any}("output_dir" => output_dir)
    end

    if !(seed === nothing)
        req_inputs = req["inputs"]
        req_inputs isa AbstractDict || error("inputs must be dict")
        req_inputs["runtime"] = Dict{String,Any}(
            "source_type" => "json",
            "data" => Dict{String,Any}("seed" => seed),
        )
    end

    return req
end

function _redis_available(; host::String = "127.0.0.1", port::Int = 6379)
    conn = nothing
    probe_key = "hydromcp:test:probe:" * string(UUIDs.uuid4())

    try
        conn = Redis.RedisConnection(host = host, port = port)
        Redis.set(conn, probe_key, "ok")
        value = Redis.get(conn, probe_key)
        Redis.del(conn, probe_key)
        return value == "ok"
    catch
        return false
    finally
        if !(conn === nothing)
            try
                close(conn)
            catch
            end
        end
    end
end

@testset "Simulation Tool Contract Tests" begin
    source_file = joinpath(dirname(@__DIR__), "data", "03604000.csv")
    result_dir = _workspace_tempdir("simulation-")

    @testset "Normalization defaults" begin
        normalized = HydroModelMCP._normalize_simulation_request(_sim_request(".\\data\\03604000.csv"))

        @test normalized["model"] == "exphydro"
        @test normalized["inputs"]["forcing"]["source_type"] == "csv"
        @test normalized["inputs"]["forcing"]["path"] == source_file
        @test normalized["output"]["output_dir"] == joinpath(dirname(@__DIR__), "result")
    end

    @testset "Normalization supports redis forcing workflow" begin
        normalized = HydroModelMCP._normalize_simulation_request(Dict(
            "model" => "exphydro",
            "inputs" => Dict(
                "forcing" => Dict(
                    "source_type" => "redis",
                    "key" => "gage_03604000_forcing",
                ),
            ),
            "output" => Dict(
                "result_source_type" => "redis",
            ),
        ))

        @test normalized["inputs"]["forcing"]["source_type"] == "redis"
        @test normalized["inputs"]["forcing"]["key"] == "gage_03604000_forcing"
        @test normalized["output"]["result_source_type"] == "redis"
    end

    @testset "Normalization rejects retired camels forcing workflow" begin
        @test_throws ArgumentError HydroModelMCP._normalize_simulation_request(Dict(
            "model" => "exphydro",
            "inputs" => Dict(
                "forcing" => Dict(
                    "source_type" => "camels",
                    "gage_id" => 1013500,
                ),
            ),
        ))
    end

    @testset "Normalization supports caravan forcing workflow" begin
        normalized = HydroModelMCP._normalize_simulation_request(Dict(
            "model" => "exphydro",
            "inputs" => Dict(
                "forcing" => Dict(
                    "source_type" => "caravan",
                    "dataset_name" => "camels",
                    "gauge_id" => "01013500",
                ),
            ),
        ))

        @test normalized["inputs"]["forcing"]["source_type"] == "caravan"
        @test normalized["inputs"]["forcing"]["dataset_name"] == "camels"
        @test normalized["inputs"]["forcing"]["gauge_id"] == "01013500"
        @test normalized["output"]["result_source_type"] == "csv"
    end

    @testset "Normalization requires dataset_name for caravan forcing workflow" begin
        @test_throws ArgumentError HydroModelMCP._normalize_simulation_request(Dict(
            "model" => "exphydro",
            "inputs" => Dict(
                "forcing" => Dict(
                    "source_type" => "caravan",
                    "gauge_id" => "01013500",
                ),
            ),
        ))
    end

    @testset "Normalization rejects legacy flat forcing fields" begin
        @test_throws ArgumentError HydroModelMCP._normalize_simulation_request(Dict(
            "model" => "exphydro",
            "source_type" => "csv",
            "path" => ".\\data\\03604000.csv",
        ))
    end

    @testset "strict_infer rejects ambiguous forcing inference" begin
        base = joinpath(dirname(@__DIR__), ".tmp_tests")
        mkpath(base)
        mktempdir(base; prefix = "strict-infer-") do tmpdir
            ambiguous_path = joinpath(tmpdir, "ambiguous_columns.csv")
            open(ambiguous_path, "w") do io
                write(io, "precipitation,temp,epx,flow(mm)\n")
                for i in 1:32
                    write(io, "$(1.0 + 0.1 * i),$(5.0 + 0.05 * i),$(0.3 + 0.01 * i),$(2.0 + 0.02 * i)\n")
                end
            end

            response = HydroModelMCP.simulation_tool.handler(Dict(
                "model" => "exphydro",
                "inputs" => Dict(
                    "forcing" => Dict(
                        "source_type" => "csv",
                        "path" => ambiguous_path,
                    ),
                ),
                "options" => Dict(
                    "strict_infer" => true,
                ),
            ))

            @test startswith(response.text, "Error:")
            @test occursin("Strict infer rejected forcing inference", response.text)
        end
    end

    @testset "run_simulation writes result artifacts and exposes random params" begin
        payload = _parse_tool_json(HydroModelMCP.simulation_tool.handler(_sim_request(source_file;
            output_dir = result_dir,
            seed = 1234,
        )))

        @test payload["status"] == "success"
        @test payload["params_source"] == "random"
        @test payload["params_seed"] == 1234
        @test payload["params_used"] isa Dict
        @test isfile(payload["output_path"])
        @test isfile(payload["metadata_path"])
        @test isfile(payload["summary_path"])
        @test haskey(payload, "inference_report")
        @test haskey(payload["inference_report"], "forcing")
    end

    @testset "run_simulation supports caravan forcing source" begin
        base = joinpath(dirname(@__DIR__), ".tmp_tests")
        mkpath(base)
        mktempdir(base; prefix = "caravan-sim-") do tmpdir
            caravan_root, _, _ = _write_mock_caravan_dataset_root(tmpdir; n_steps = 48)

            payload = _parse_tool_json(HydroModelMCP.simulation_tool.handler(Dict(
                "model" => "exphydro",
                "inputs" => Dict(
                    "forcing" => Dict(
                        "source_type" => "caravan",
                        "dataset_root" => caravan_root,
                        "dataset_name" => "camels",
                        "gauge_id" => "01013500",
                    ),
                    "runtime" => Dict(
                        "source_type" => "json",
                        "data" => Dict("seed" => 2027),
                    ),
                ),
                "output" => Dict("output_dir" => result_dir),
            )))

            @test payload["status"] == "success"
            @test payload["params_source"] == "random"
            @test payload["params_seed"] == 2027
            @test isfile(payload["output_path"])
            @test isfile(payload["metadata_path"])
        end
    end

    @testset "random params are reproducible for the same seed" begin
        first_run = Simulation.run_simulation(_sim_request(source_file;
            output_dir = result_dir,
            seed = 7,
        ))
        second_run = Simulation.run_simulation(_sim_request(source_file;
            output_dir = result_dir,
            seed = 7,
        ))

        @test first_run["params_used"] == second_run["params_used"]
    end

    @testset "compute_metrics accepts direct file paths and writes metrics artifacts" begin
        simulation_payload = Simulation.run_simulation(_sim_request(source_file;
            output_dir = result_dir,
            seed = 99,
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
        @test payload["sample_size"] > 0
        @test isfile(payload["output_path"])
    end

    @testset "compute_metrics accepts Caravan observed source and same-session simulation inference" begin
        base = joinpath(dirname(@__DIR__), ".tmp_tests")
        mkpath(base)
        mktempdir(base; prefix = "caravan-metrics-") do tmpdir
            caravan_root, _, _ = _write_mock_caravan_dataset_root(tmpdir; n_steps = 48)

            simulation_payload = _parse_tool_json(HydroModelMCP.simulation_tool.handler(Dict(
                "model" => "exphydro",
                "inputs" => Dict(
                    "forcing" => Dict(
                        "source_type" => "caravan",
                        "dataset_root" => caravan_root,
                        "dataset_name" => "camels",
                        "gauge_id" => "01013500",
                    ),
                ),
                "output" => Dict("output_dir" => result_dir),
            )))

            @test simulation_payload["status"] == "success"
            @test haskey(simulation_payload, "forcing_source")
            @test simulation_payload["forcing_source"]["source_type"] == "caravan"

            payload = _parse_tool_json(HydroModelMCP.compute_metrics_tool.handler(Dict(
                "observed" => Dict(
                    "source_type" => "caravan",
                    "dataset_root" => caravan_root,
                    "dataset_name" => "camels",
                    "gauge_id" => "01013500",
                ),
                "output_dir" => result_dir,
                "metrics" => ["NSE", "KGE", "RMSE"],
            )))

            @test payload["status"] == "success"
            @test haskey(payload["metrics"], "NSE")
            @test haskey(payload["metrics"], "KGE")
            @test any(occursin("Inferred simulated source", warning) for warning in payload["warnings"])
            @test isfile(payload["output_path"])
        end
    end

    @testset "compute_metrics infers Caravan observed source from last simulation context" begin
        base = joinpath(dirname(@__DIR__), ".tmp_tests")
        mkpath(base)
        mktempdir(base; prefix = "caravan-metrics-infer-") do tmpdir
            caravan_root, _, _ = _write_mock_caravan_dataset_root(tmpdir; n_steps = 48)

            simulation_payload = _parse_tool_json(HydroModelMCP.simulation_tool.handler(Dict(
                "model" => "exphydro",
                "inputs" => Dict(
                    "forcing" => Dict(
                        "source_type" => "caravan",
                        "dataset_root" => caravan_root,
                        "dataset_name" => "camels",
                        "gauge_id" => "01013500",
                    ),
                ),
                "output" => Dict("output_dir" => result_dir),
            )))

            payload = _parse_tool_json(HydroModelMCP.compute_metrics_tool.handler(Dict(
                "simulated" => Dict(
                    "source_type" => "csv",
                    "path" => simulation_payload["output_path"],
                ),
                "output_dir" => result_dir,
                "metrics" => ["NSE", "RMSE"],
            )))

            @test payload["status"] == "success"
            @test any(occursin("Caravan forcing context", warning) for warning in payload["warnings"])
            @test haskey(payload["metrics"], "NSE")
        end
    end

    @testset "compute_metrics infers sources from session context" begin
        simulation_payload = _parse_tool_json(HydroModelMCP.simulation_tool.handler(Dict(
            "model" => "exphydro",
            "inputs" => Dict(
                "forcing" => Dict(
                    "source_type" => "csv",
                    "path" => source_file,
                ),
            ),
            "output" => Dict(
                "result_source_type" => "csv",
                "output_dir" => result_dir,
            ),
        )))

        @test simulation_payload["status"] == "success"

        payload = _parse_tool_json(HydroModelMCP.compute_metrics_tool.handler(Dict(
            "output_dir" => result_dir,
            "metrics" => JSON3.read("[\"KGE\",\"NSE\"]"),
        )))

        @test payload["status"] == "success"
        @test haskey(payload, "metrics")
        @test haskey(payload["metrics"], "KGE")
        @test haskey(payload["metrics"], "NSE")
        @test isfile(payload["output_path"])
        @test any(occursin("Inferred simulated source", warning) for warning in payload["warnings"])
    end

    @testset "run_simulation accepts inline partial parameters and runtime aliases" begin
        payload = _parse_tool_json(HydroModelMCP.simulation_tool.handler(Dict(
            "model" => "exphydro",
            "inputs" => Dict(
                "forcing" => Dict(
                    "source_type" => "csv",
                    "path" => source_file,
                ),
                "parameters" => Dict(
                    "f" => 0.03,
                    "Smax" => 600.0,
                ),
                "runtime" => Dict(
                    "solver" => "离散求解",
                    "interpolation" => "线性插值",
                ),
            ),
            "output" => Dict(
                "result_source_type" => "csv",
                "output_dir" => result_dir,
            ),
        )))

        @test payload["status"] == "success"
        @test haskey(payload, "params_used")
        @test length(payload["params_used"]) == 6
        @test payload["run_info"]["solver"] == "DISCRETE"
        @test payload["run_info"]["interpolation"] == "LINEAR"
    end

    @testset "run_validation reuses same-session parameters when omitted" begin
        calibration_payload = _parse_tool_json(HydroModelMCP.calibrate_tool.handler(Dict(
            "model" => "exphydro",
            "inputs" => Dict(
                "forcing" => Dict("source_type" => "csv", "path" => source_file),
                "observation" => Dict("source_type" => "csv", "path" => source_file, "column" => "flow(mm)"),
            ),
            "objective" => "KGE",
            "algorithm" => "BBO",
            "maxiters" => 15,
            "n_trials" => 1,
        )))
        @test calibration_payload["status"] == "success"

        payload = _parse_tool_json(HydroModelMCP.validation_tool.handler(Dict(
            "model" => "exphydro",
            "inputs" => Dict(
                "forcing" => Dict("source_type" => "csv", "path" => source_file),
                "observation" => Dict("source_type" => "csv", "path" => source_file, "column" => "flow(mm)"),
            ),
            "calibration_period" => Dict("start_index" => 1, "end_index" => 700),
            "validation_period" => Dict("start_index" => 701, "end_index" => 1200),
            "metrics" => ["NSE", "KGE"],
        )))

        @test payload["status"] == "success"
        @test haskey(payload, "parameter_source")
        @test payload["parameter_source"]["source"] == "session_fallback"
        @test any(occursin("reused parameters", warning) for warning in payload["warnings"])
    end

    @testset "run_validation accepts calibration_result-style inline parameters" begin
        calibration_payload = _parse_tool_json(HydroModelMCP.calibrate_tool.handler(Dict(
            "model" => "exphydro",
            "inputs" => Dict(
                "forcing" => Dict("source_type" => "csv", "path" => source_file),
                "observation" => Dict("source_type" => "csv", "path" => source_file, "column" => "flow(mm)"),
            ),
            "objective" => "KGE",
            "algorithm" => "PSO",
            "maxiters" => 10,
            "n_trials" => 1,
        )))
        @test calibration_payload["status"] == "success"

        payload = _parse_tool_json(HydroModelMCP.validation_tool.handler(Dict(
            "model" => "exphydro",
            "inputs" => Dict(
                "forcing" => Dict("source_type" => "csv", "path" => source_file),
                "observation" => Dict("source_type" => "csv", "path" => source_file, "column" => "flow(mm)"),
                "parameters" => Dict(
                    "best_params" => calibration_payload["best_params"],
                ),
                "runtime" => Dict(
                    "solver" => "continuous",
                    "interpolation" => "constant",
                ),
            ),
            "calibration_period" => Dict("start_index" => 1, "end_index" => 600),
            "validation_period" => Dict("start_index" => 601, "end_index" => 1100),
            "metrics" => ["NSE"],
        )))

        @test payload["status"] == "success"
        @test haskey(payload, "parameters")
        @test payload["parameter_source"]["source"] == "json"
        @test payload["parameters"] isa Dict
    end

    @testset "run_validation accepts stored calibration result_id and split fallback" begin
        calibration_payload = _parse_tool_json(HydroModelMCP.calibrate_tool.handler(Dict(
            "model" => "exphydro",
            "inputs" => Dict(
                "forcing" => Dict("source_type" => "csv", "path" => source_file),
                "observation" => Dict("source_type" => "csv", "path" => source_file, "column" => "flow(mm)"),
            ),
            "objective" => "KGE",
            "algorithm" => "BBO",
            "maxiters" => 10,
            "n_trials" => 1,
        )))
        @test calibration_payload["status"] == "success"
        @test haskey(calibration_payload, "result_id")

        payload = _parse_tool_json(HydroModelMCP.validation_tool.handler(Dict(
            "model" => "exphydro",
            "inputs" => Dict(
                "forcing" => Dict("source_type" => "csv", "path" => source_file),
                "observation" => Dict("source_type" => "csv", "path" => source_file, "column" => "flow(mm)"),
                "parameters" => Dict(
                    "source_type" => "calibration_result",
                    "result_id" => calibration_payload["result_id"],
                    "storage_category" => calibration_payload["storage_category"],
                ),
            ),
            "metrics" => ["NSE", "KGE"],
        )))

        @test payload["status"] == "success"
        @test payload["parameter_source"]["source"] == "calibration_result"
        @test any(occursin("result_id resolved", warning) for warning in payload["warnings"])
        @test any(occursin("reused split embedded", warning) for warning in payload["warnings"])
    end

    @testset "run_validation supports unified protocol" begin
        n = 40
        forcing_data = Dict(
            "P" => [1.0 + 0.01 * i for i in 1:n],
            "T" => fill(5.0, n),
            "Ep" => fill(0.5, n),
        )
        obs_data = Dict(
            "Qobs" => [0.2 + 0.02 * i for i in 1:n],
        )
        params = Dict(
            "f" => 0.0015,
            "Smax" => 1100.0,
            "Qmax" => 21.0,
            "Df" => 2.6,
            "Tmax" => 1.6,
            "Tmin" => -0.9,
        )

        result = Validation.run_validation(Dict(
            "model" => "exphydro",
            "inputs" => Dict(
                "forcing" => Dict(
                    "source_type" => "json",
                    "data" => forcing_data,
                ),
                "observation" => Dict(
                    "source_type" => "json",
                    "data" => obs_data,
                    "column" => "Qobs",
                ),
                "parameters" => Dict(
                    "source_type" => "json",
                    "data" => params,
                ),
            ),
            "calibration_period" => Dict("start_index" => 1, "end_index" => 20),
            "validation_period" => Dict("start_index" => 21, "end_index" => 40),
            "metrics" => ["NSE", "RMSE"],
        ))

        @test result["status"] == "success"
        @test haskey(result, "calibration_metrics")
        @test haskey(result, "validation_metrics")
    end

    @testset "run_validation supports date periods via synthetic timeline fallback" begin
        payload = _parse_tool_json(HydroModelMCP.validation_tool.handler(Dict(
            "model" => "exphydro",
            "inputs" => Dict(
                "forcing" => Dict(
                    "source_type" => "csv",
                    "path" => source_file,
                ),
                "observation" => Dict(
                    "source_type" => "csv",
                    "path" => source_file,
                    "column" => "flow(mm)",
                ),
                "parameters" => Dict(
                    "f" => 0.02,
                    "Smax" => 650.0,
                    "Qmax" => 24.0,
                    "Df" => 2.1,
                    "Tmax" => 1.4,
                    "Tmin" => -1.2,
                ),
            ),
            "calibration_period" => Dict(
                "start" => "2000-01-01",
                "end" => "2004-12-31",
            ),
            "validation_period" => Dict(
                "start" => "2005-01-01",
                "end" => "2008-12-31",
            ),
            "metrics" => ["NSE", "KGE"],
        )))

        @test payload["status"] == "success"
        @test haskey(payload, "calibration_metrics")
        @test haskey(payload, "validation_metrics")
        @test any(occursin("synthetic timeline", warning) for warning in payload["warnings"])
    end

    @testset "Redis e2e workflow: CSV path forcing -> redis result" begin
        redis_host = "127.0.0.1"
        redis_port = 6379

        if !_redis_available(host = redis_host, port = redis_port)
            @info "Skipping CSV-path Redis e2e test because Redis is unavailable at $(redis_host):$(redis_port)"
            @test true
        else
            result_key = "hydromcp:test:csv_result:" * string(UUIDs.uuid4())

            conn = Redis.RedisConnection(host = redis_host, port = redis_port)
            try
                payload = _parse_tool_json(HydroModelMCP.simulation_tool.handler(Dict(
                    "model" => "exphydro",
                    "inputs" => Dict(
                        "forcing" => Dict(
                            "source_type" => "csv",
                            "path" => source_file,
                        ),
                        "runtime" => Dict(
                            "source_type" => "json",
                            "data" => Dict("seed" => 2027),
                        ),
                    ),
                    "output" => Dict(
                        "result_source_type" => "redis",
                        "result_host" => redis_host,
                        "result_port" => redis_port,
                        "result_key" => result_key,
                    ),
                )))

                @test payload["status"] == "success"
                @test payload["source_type"] == "redis"
                @test payload["key"] == result_key
                @test !haskey(payload, "output_path")

                cached_result = JSON3.read(Redis.get(conn, result_key), Dict{String,Any})
                @test haskey(cached_result, "result")
                @test cached_result["result"] isa AbstractVector
            finally
                try
                    Redis.del(conn, result_key)
                catch
                end
                try
                    close(conn)
                catch
                end
            end
        end
    end
end
