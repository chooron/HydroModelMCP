module Ensemble

using ..DataLoader
using ..Simulation
using Base.Threads
using CSV
using DataFrames
using Statistics
using UUIDs

function _inline_forcing_config(forcing_config::AbstractDict)
    source_type = Symbol(lowercase(string(forcing_config["source_type"])))
    forcing_nt, _ = DataLoader.load_data(Val(source_type), forcing_config)
    inline_data = Dict{String,Any}()
    for name in keys(forcing_nt)
        inline_data[string(name)] = collect(Float64.(forcing_nt[name]))
    end
    return Dict("source_type" => "json", "data" => inline_data)
end

function _extract_simulated_flow(result::AbstractDict)
    if haskey(result, "result")
        return Float64.(result["result"])
    end

    output_path = get(result, "output_path", get(result, "path", nothing))
    if output_path isa AbstractString && isfile(output_path)
        df = CSV.read(output_path, DataFrame)
        return Float64.(df[!, 1])
    end

    throw(ArgumentError("Could not extract simulated flow from ensemble member result"))
end

function _build_member_args(base_args::Dict{String,Any}, params)
    member_args = copy(base_args)
    member_args["params"] = params
    return member_args
end

function run_ensemble(args::AbstractDict)
    model_name = string(get(args, "model_name", get(args, "model", nothing)))
    model_name == "nothing" && throw(ArgumentError("Missing model_name"))

    parameter_sets = args["parameter_sets"]
    forcing_config = _inline_forcing_config(args["forcing"])
    n_members = length(parameter_sets)

    base_args = Dict{String,Any}(
        "model" => model_name,
        "source_type" => "json",
        "data" => forcing_config["data"],
        "solver" => get(args, "solver", "DISCRETE"),
        "interpolation" => get(args, "interpolation", get(args, "interpolator", "LINEAR")),
    )

    for optional_key in ("input_mapping", "period", "warmup", "output_dir")
        haskey(args, optional_key) && (base_args[optional_key] = args[optional_key])
    end

    ensemble_results = Vector{Dict{String,Any}}(undef, n_members)

    if get(args, "parallel", true) && n_members > 1
        @threads for i in 1:n_members
            member_result = Simulation.run_simulation(_build_member_args(base_args, parameter_sets[i]))
            ensemble_results[i] = Dict(
                "member_id" => i - 1,
                "parameters" => parameter_sets[i],
                "simulated_flow" => _extract_simulated_flow(member_result),
            )
        end
    else
        for i in 1:n_members
            member_result = Simulation.run_simulation(_build_member_args(base_args, parameter_sets[i]))
            ensemble_results[i] = Dict(
                "member_id" => i - 1,
                "parameters" => parameter_sets[i],
                "simulated_flow" => _extract_simulated_flow(member_result),
            )
        end
    end

    flow_matrix = hcat([result["simulated_flow"] for result in ensemble_results]...)
    ensemble_mean = vec(mean(flow_matrix, dims = 2))
    ensemble_std = vec(std(flow_matrix, dims = 2))

    return Dict(
        "n_members" => n_members,
        "ensemble_members" => ensemble_results,
        "ensemble_mean" => ensemble_mean,
        "ensemble_std" => ensemble_std,
        "result_id" => string(UUIDs.uuid4()),
        "model_name" => model_name,
    )
end

end
