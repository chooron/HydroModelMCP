module Discovery

using ..HydroModelLibrary
using ..HydroModels

export list_models, find_model, get_model_info, get_parameters_detail, get_variables_detail

function list_models()
    return sort!(String.(HydroModelLibrary.AVAILABLE_MODELS))
end

function _score_model_match(query::String, candidate::String)
    q = lowercase(strip(query))
    c = lowercase(strip(candidate))

    isempty(q) && return 0.0
    q == c && return 1.0
    startswith(c, q) && return 0.9
    occursin(q, c) && return 0.75
    return 0.0
end

function find_model(name::String)
    isempty(strip(name)) && return nothing

    models = list_models()
    scored = [
        (model = model_name, score = _score_model_match(name, model_name))
        for model_name in models
    ]
    filter!(item -> item.score > 0, scored)
    isempty(scored) && return nothing

    sort!(scored; by = item -> (-item.score, item.model))
    return first(scored).model
end

function _load_model(name::String)
    valid_name = find_model(name)
    isnothing(valid_name) && throw(ArgumentError("Model '$name' was not found."))

    model_module = HydroModelLibrary.load_model(Symbol(valid_name), reload = false)
    model = Base.invokelatest(wrapper -> wrapper.model, model_module)
    return valid_name, model_module, model
end

function get_model_info(name::String)
    valid_name, _, model = _load_model(name)

    return Dict(
        "model" => valid_name,
        "full_name" => valid_name,
        "description" => sprint(show, MIME("text/plain"), model),
        "inputs" => string.(HydroModels.get_input_names(model)),
        "states" => string.(HydroModels.get_state_names(model)),
        "outputs" => string.(HydroModels.get_output_names(model)),
        "parameter_count" => length(HydroModels.get_param_names(model)),
    )
end

function get_variables_detail(model_name::String)
    valid_name, wrapper, _ = _load_model(model_name)
    vars_list = Base.invokelatest(module_wrapper -> module_wrapper.model_variables, wrapper)

    info_list = Dict{String,Any}[]
    for variable in vars_list
        push!(info_list, Dict(
            "model" => valid_name,
            "name" => string(variable),
            "description" => try
                HydroModels.getdescription(variable)
            catch
                ""
            end,
            "unit" => try
                string(HydroModels.getunit(variable))
            catch
                "-"
            end,
            "type" => "variable",
        ))
    end

    return info_list
end

function get_parameters_detail(model_name::String)
    valid_name, wrapper, _ = _load_model(model_name)
    params_list = Base.invokelatest(module_wrapper -> module_wrapper.model_parameters, wrapper)

    info_list = Dict{String,Any}[]
    for parameter in params_list
        bounds = try
            raw_bounds = HydroModels.getbounds(parameter)
            (Float64(raw_bounds[1]), Float64(raw_bounds[2]))
        catch
            nothing
        end

        push!(info_list, Dict(
            "model" => valid_name,
            "name" => string(parameter),
            "description" => try
                HydroModels.getdescription(parameter)
            catch
                ""
            end,
            "unit" => try
                string(HydroModels.getunit(parameter))
            catch
                "-"
            end,
            "min" => isnothing(bounds) ? nothing : bounds[1],
            "max" => isnothing(bounds) ? nothing : bounds[2],
            "default" => nothing,
            "type" => "parameter",
        ))
    end

    return info_list
end

end # module Discovery
