using JSON3
using NPZ
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

function _write_mock_camels_dataset_for_sim(path::String; gage_ids = [1013500, 3604000], n_steps::Int = 48)
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

    @testset "Normalization supports camels forcing workflow" begin
        normalized = HydroModelMCP._normalize_simulation_request(Dict(
            "model" => "exphydro",
            "inputs" => Dict(
                "forcing" => Dict(
                    "source_type" => "camels",
                    "gage_id" => 1013500,
                ),
            ),
        ))

        @test normalized["inputs"]["forcing"]["source_type"] == "camels"
        @test normalized["inputs"]["forcing"]["gage_id"] == 1013500
        @test normalized["output"]["result_source_type"] == "csv"
    end

    @testset "Normalization rejects legacy flat forcing fields" begin
        @test_throws ArgumentError HydroModelMCP._normalize_simulation_request(Dict(
            "model" => "exphydro",
            "source_type" => "csv",
            "path" => ".\\data\\03604000.csv",
        ))
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

    @testset "run_simulation supports camels forcing source" begin
        base = joinpath(dirname(@__DIR__), ".tmp_tests")
        mkpath(base)
        mktempdir(base; prefix = "camels-sim-") do tmpdir
            npz_path = _write_mock_camels_dataset_for_sim(joinpath(tmpdir, "mock_camels.npz"))

            payload = _parse_tool_json(HydroModelMCP.simulation_tool.handler(Dict(
                "model" => "exphydro",
                "inputs" => Dict(
                    "forcing" => Dict(
                        "source_type" => "camels",
                        "dataset_path" => npz_path,
                        "gage_id" => 1013500,
                    ),
                    "runtime" => Dict(
                        "source_type" => "json",
                        "data" => Dict("seed" => 2026),
                    ),
                ),
                "output" => Dict("output_dir" => result_dir),
            )))

            @test payload["status"] == "success"
            @test payload["params_source"] == "random"
            @test payload["params_seed"] == 2026
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
