module Ensemble

using ..Simulation
using ..Statistics
using ..UUIDs
using ..UnifiedInputs

function _build_member_params(model, raw_params)
    raw_params isa AbstractDict || throw(ArgumentError("Each parameter set must be an object"))
    return Dict{String,Float64}(string(k) => Float64(v) for (k, v) in pairs(raw_params))
end

function _build_member_args(base_request::Dict{String,Any}, params::Dict{String,Float64})
    member_request = deepcopy(base_request)
    member_request["inputs"]["parameters"] = Dict(
        "source_type" => "json",
        "data" => params,
    )
    return member_request
end

function run_ensemble(args::AbstractDict)
    request = UnifiedInputs.normalize_workflow_request(args)
    haskey(request, "parameter_sets") || throw(ArgumentError("parameter_sets is required"))

    param_sets_raw = request["parameter_sets"]
    param_sets_raw isa AbstractVector || throw(ArgumentError("parameter_sets must be an array"))
    isempty(param_sets_raw) && throw(ArgumentError("parameter_sets cannot be empty"))

    canonical_model_name, _, model = Simulation._load_model(request["model"])
    base_resolved = UnifiedInputs.resolve_common_inputs(request, model)

    base_runtime = base_resolved["runtime"]
    base_forcing = base_resolved["forcing_nt"]
    solver_sym = Symbol(uppercase(string(get(base_runtime, "solver", "DISCRETE"))))
    interp_sym = Symbol(uppercase(string(get(base_runtime, "interpolation", "LINEAR"))))
    init_states = Simulation._build_init_states(model, get(base_runtime, "init_states", nothing))
    config_dict = haskey(base_runtime, "config") && base_runtime["config"] isa AbstractDict ?
        Dict{String,Any}(string(k) => v for (k, v) in pairs(base_runtime["config"])) : Dict{String,Any}()

    ensemble_members = Vector{Dict{String,Any}}(undef, length(param_sets_raw))
    member_flows = Vector{Vector{Float64}}(undef, length(param_sets_raw))

    for idx in eachindex(param_sets_raw)
        params = _build_member_params(model, param_sets_raw[idx])
        params_vec = Simulation._component_vector_from_params(model, params)
        simulated_flow = Simulation._execute_core(
            model,
            base_forcing,
            params_vec,
            init_states,
            solver_sym,
            interp_sym,
            config_dict,
        )

        member_flows[idx] = simulated_flow
        ensemble_members[idx] = Dict(
            "member_id" => idx - 1,
            "parameters" => params,
            "simulated_flow" => simulated_flow,
        )
    end

    flow_matrix = hcat(member_flows...)
    ensemble_mean = vec(mean(flow_matrix, dims = 2))
    ensemble_std = vec(std(flow_matrix, dims = 2))

    return Dict{String,Any}(
        "status" => "success",
        "model" => canonical_model_name,
        "n_members" => length(param_sets_raw),
        "ensemble_members" => ensemble_members,
        "ensemble_mean" => ensemble_mean,
        "ensemble_std" => ensemble_std,
        "result_id" => string(UUIDs.uuid4()),
        "warnings" => base_resolved["warnings"],
        "inference_report" => base_resolved["inference_report"],
    )
end

end
