module Simulation

using ..ComponentArrays
using ..DataInterpolations
using ..DataInterpolations: ConstantInterpolation, LinearInterpolation
using ..DataLoader
using ..Dates
using ..HydroModelLibrary
using ..HydroModels
using ..Random
using ..Statistics
using ..UUIDs

const DEFAULT_INPUT_ALIASES = Dict(
    "P" => ["P", "prcp(mm/day)", "precipitation", "precip", "precip_mm_day"],
    "T" => ["T", "tmean(C)", "temperature", "temp", "mean_temperature"],
    "Ep" => ["Ep", "pet(mm)", "pet", "etp", "evapotranspiration"],
)

_resolve_solver(::Val{:ODE}) = HydroModels.ODESolver
_resolve_solver(::Val{:DISCRETE}) = HydroModels.DiscreteSolver
_resolve_solver(::Val{:MUTABLE}) = HydroModels.MutableSolver
_resolve_solver(::Val{:IMMUTABLE}) = HydroModels.ImmutableSolver
_resolve_solver(::Val{T}) where {T} = throw(ArgumentError("Unsupported solver: $T"))

_resolve_interpolator(::Val{:LINEAR}) = Val(LinearInterpolation)
_resolve_interpolator(::Val{:CONSTANT}) = Val(ConstantInterpolation)
_resolve_interpolator(::Val{:DIRECT}) = Val(HydroModels.DirectInterpolation)
_resolve_interpolator(::Val{T}) where {T} = throw(ArgumentError("Unsupported interpolation: $T"))

function _execute_core(
    model::M,
    forcing_nt::NamedTuple,
    params_vec::ComponentVector,
    init_states,
    solver_sym::Symbol,
    interp_sym::Symbol,
    config_dict::AbstractDict,
) where {M}
    input_names = HydroModels.get_input_names(model)
    missing_vars = setdiff(input_names, keys(forcing_nt))
    if !isempty(missing_vars)
        throw(ArgumentError("Missing forcing inputs: $(join(string.(missing_vars), ", "))"))
    end

    input_matrix = stack([Float64.(forcing_nt[name]) for name in input_names], dims = 1)
    hydro_config = HydroModels.HydroConfig(
        solver = _resolve_solver(Val(solver_sym)),
        interpolator = _resolve_interpolator(Val(interp_sym)),
    )

    result_matrix = if isnothing(init_states)
        model(
            input_matrix,
            params_vec;
            config = hydro_config,
        )
    else
        model(
            input_matrix,
            params_vec;
            initstates = init_states,
            config = hydro_config,
        )
    end

    output_names = HydroModels.get_output_names(model)
    target_var = get(config_dict, "output_variable", nothing)

    output_vector = if isnothing(target_var)
        result_matrix[end, :]
    else
        idx = findfirst(==(Symbol(target_var)), output_names)
        isnothing(idx) && throw(ArgumentError("Output variable '$target_var' was not found"))
        result_matrix[idx, :]
    end

    return collect(Float64.(output_vector))
end

function _load_model(model_name::String)
    available_models = String.(HydroModelLibrary.AVAILABLE_MODELS)
    idx = findfirst(candidate -> lowercase(candidate) == lowercase(strip(model_name)), available_models)
    isnothing(idx) && throw(ArgumentError("Model '$model_name' was not found"))

    canonical = available_models[idx]
    model_module = HydroModelLibrary.load_model(Symbol(canonical), reload = false)
    model = Base.invokelatest(wrapper -> wrapper.model, model_module)
    return canonical, model_module, model
end

function _normalize_input_mapping(mapping)
    if isnothing(mapping)
        return nothing
    elseif mapping isa AbstractDict
        return Dict{String,String}(string(k) => string(v) for (k, v) in pairs(mapping))
    end
    throw(ArgumentError("input_mapping must be an object"))
end

function _resolve_input_source(
    target::Symbol,
    available_keys::Vector{String},
    input_mapping::Union{Nothing,Dict{String,String}},
)
    target_name = string(target)

    if !isnothing(input_mapping)
        if haskey(input_mapping, target_name)
            return input_mapping[target_name], false
        end
        for (source_name, mapped_target) in input_mapping
            if mapped_target == target_name
                return source_name, false
            end
        end
    end

    aliases = get(DEFAULT_INPUT_ALIASES, target_name, [target_name])
    for alias in aliases
        alias in available_keys && return alias, alias != target_name
    end

    return nothing, false
end

function _normalize_forcing_data(
    forcing_data::NamedTuple,
    model,
    input_mapping::Union{Nothing,Dict{String,String}},
)
    available = Dict{String,Vector{Float64}}(
        string(name) => collect(Float64.(forcing_data[name]))
        for name in keys(forcing_data)
    )
    input_names = Tuple(HydroModels.get_input_names(model))
    normalized_values = Vector{Vector{Float64}}(undef, length(input_names))
    warnings = String[]

    for (idx, target_name) in enumerate(input_names)
        source_name, inferred = _resolve_input_source(target_name, collect(keys(available)), input_mapping)
        isnothing(source_name) &&
            throw(ArgumentError("Could not resolve forcing column for model input '$(target_name)'"))

        normalized_values[idx] = available[source_name]
        if inferred
            push!(warnings, "Mapped input $(target_name) from forcing column '$source_name'")
        end
    end

    normalized = NamedTuple{input_names}(Tuple(normalized_values))
    return normalized, warnings
end

function _component_vector_from_params(model, params_input)
    param_names = Tuple(HydroModels.get_param_names(model))

    if params_input isa AbstractDict
        values = Float64[]
        for pname in param_names
            key = string(pname)
            haskey(params_input, key) || throw(ArgumentError("Missing parameter '$key'"))
            push!(values, Float64(params_input[key]))
        end
        return ComponentVector(; params = NamedTuple{param_names}(Tuple(values)))
    elseif params_input isa AbstractVector
        length(params_input) == length(param_names) ||
            throw(ArgumentError("Parameter vector length does not match model parameter count"))
        return ComponentVector(; params = NamedTuple{param_names}(Tuple(Float64.(params_input))))
    end

    throw(ArgumentError("params must be an object or array"))
end

function _random_params(model_name::String, model; seed::Union{Nothing,Int} = nothing)
    if !isnothing(seed)
        Random.seed!(seed)
    end

    params_vec = HydroModelLibrary.get_random_params(model_name)
    param_names = HydroModels.get_param_names(model)
    params_used = Dict{String,Float64}()
    for pname in param_names
        params_used[string(pname)] = Float64(params_vec.params[pname])
    end
    return params_vec, params_used
end

function _build_init_states(model, init_states_input)
    state_names = Tuple(HydroModels.get_state_names(model))
    isempty(state_names) && return nothing

    values = if isnothing(init_states_input)
        zeros(length(state_names))
    elseif init_states_input isa AbstractDict
        [Float64(get(init_states_input, string(state), 0.0)) for state in state_names]
    else
        throw(ArgumentError("init_states must be an object"))
    end

    return ComponentVector(NamedTuple{state_names}(Tuple(values)))
end

function _forcing_config(args::AbstractDict)
    if haskey(args, "forcing")
        return args["forcing"]
    end

    config = Dict{String,Any}()
    for key in ("source_type", "path", "key", "data", "host", "port")
        haskey(args, key) && (config[key] = args[key])
    end
    isempty(config) && throw(ArgumentError("Missing forcing configuration"))
    return config
end

function _normalize_output_dir(output_dir)
    if output_dir === nothing
        return nothing
    elseif output_dir isa String
        return output_dir
    end
    return string(output_dir)
end

function run_simulation(args::AbstractDict)
    model_name = string(get(args, "model_name", get(args, "model", nothing)))
    model_name == "nothing" && throw(ArgumentError("Missing model name"))

    forcing_config = _forcing_config(args)
    canonical_model_name, _model_module, model = _load_model(model_name)
    source_type = Symbol(lowercase(string(forcing_config["source_type"])))
    forcing_data, forcing_metadata = DataLoader.load_data(Val(source_type), forcing_config)

    input_mapping = _normalize_input_mapping(get(args, "input_mapping", nothing))
    normalized_forcing, mapping_warnings = _normalize_forcing_data(forcing_data, model, input_mapping)

    params_input = get(args, "params", nothing)
    seed = haskey(args, "seed") ? Int(args["seed"]) :
        (haskey(args, "random_seed") ? Int(args["random_seed"]) : nothing)

    params_vec, params_used, params_source = if isnothing(params_input)
        generated_params, used = _random_params(canonical_model_name, model; seed = seed)
        generated_params, used, "random"
    else
        provided = _component_vector_from_params(model, params_input)
        used = Dict{String,Float64}()
        for pname in HydroModels.get_param_names(model)
            used[string(pname)] = Float64(provided.params[pname])
        end
        provided, used, "provided"
    end

    init_states = _build_init_states(model, get(args, "init_states", nothing))
    solver_sym = Symbol(uppercase(string(get(args, "solver", "DISCRETE"))))
    interp_sym = Symbol(uppercase(string(get(args, "interpolation", get(args, "interpolator", "LINEAR")))))
    config_dict = if haskey(args, "config") && args["config"] isa AbstractDict
        Dict{String,Any}(string(k) => v for (k, v) in pairs(args["config"]))
    else
        Dict{String,Any}()
    end

    warnings = copy(mapping_warnings)
    for optional_key in ("period", "warmup")
        haskey(args, optional_key) && push!(warnings, "Field '$optional_key' is recorded but not applied by the current simulation core")
    end

    t_start = time()
    result_vector = _execute_core(
        model,
        normalized_forcing,
        params_vec,
        init_states,
        solver_sym,
        interp_sym,
        config_dict,
    )
    runtime_seconds = round(time() - t_start; digits = 3)

    run_id = string(UUIDs.uuid4())
    timestamp = Dates.format(now(), "yyyymmddHHMMSS")
    output_dir = _normalize_output_dir(get(args, "output_dir", nothing))

    artifact_payload = Dict{String,Any}(
        "status" => "success",
        "run_id" => run_id,
        "warnings" => warnings,
        "params_used" => params_used,
        "params_source" => params_source,
        "params_seed" => seed,
        "run_info" => Dict(
            "model" => canonical_model_name,
            "solver" => string(solver_sym),
            "interpolation" => string(interp_sym),
            "runtime_seconds" => runtime_seconds,
            "period" => get(args, "period", nothing),
            "warmup" => get(args, "warmup", nothing),
        ),
    )

    save_context = (
        output_dir = output_dir,
        timestamp = timestamp,
        run_id = run_id,
        run_info = artifact_payload["run_info"],
        artifact_payload = artifact_payload,
        warnings = warnings,
        params_used = params_used,
        params_source = params_source,
        params_seed = seed,
        base_name = get(args, "base_name", nothing),
    )

    result = DataLoader.save_data(Val(source_type), result_vector, DataLoader._merge_metadata(forcing_metadata, save_context))
    result["status"] = "success"
    result["run_id"] = run_id
    result["warnings"] = warnings
    result["run_info"] = artifact_payload["run_info"]
    result["model"] = canonical_model_name
    result["params_used"] = params_used
    result["params_source"] = params_source
    result["params_seed"] = seed
    return result
end

end
