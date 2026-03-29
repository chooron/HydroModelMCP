using JSON3
using ModelContextProtocol

"""
Helper functions shared by MCP tool handlers.
"""

function create_error_response(error_message::String)
    return TextContent(text = "Error: " * error_message)
end

function create_error_response(e::Exception)
    err_msg = sprint(showerror, e, catch_backtrace())
    return TextContent(text = "Error: " * err_msg)
end

function create_json_response(payload)
    return TextContent(text = JSON3.write(payload))
end

function validate_required_params(params::Dict, required_keys::Vector{String})
    for key in required_keys
        if !haskey(params, key)
            return "Missing required parameter: $key"
        end
        value = params[key]
        if value === nothing || (value isa String && isempty(strip(value)))
            return "Parameter '$key' cannot be empty"
        end
    end
    return nothing
end

function validate_enum_param(
    params::Dict,
    param_name::String,
    valid_values::Vector{String},
    default::Union{String,Nothing} = nothing,
)
    if !haskey(params, param_name)
        return default === nothing ? "Missing required parameter: $param_name" : nothing
    end

    value = params[param_name]
    if !(value in valid_values)
        return "Invalid value for '$param_name': $value. Must be one of: $(join(valid_values, ", "))"
    end
    return nothing
end

function normalize_string_dict(value)
    if value isa Dict{String,Any}
        return copy(value)
    elseif value isa AbstractDict
        return Dict{String,Any}(string(k) => v for (k, v) in pairs(value))
    else
        throw(ArgumentError("Expected an object-like value, got $(typeof(value))"))
    end
end

workspace_root() = normpath(abspath(joinpath(@__DIR__, "..", "..")))

function _normalized_components(path::AbstractString)
    full_path = normpath(abspath(path))
    parts = collect(splitpath(full_path))
    if Sys.iswindows() && !isempty(parts)
        parts[1] = lowercase(parts[1])
    end
    return parts
end

function is_within_workspace(path::AbstractString; root::AbstractString = workspace_root())
    path_parts = _normalized_components(path)
    root_parts = _normalized_components(root)

    length(path_parts) >= length(root_parts) || return false
    return path_parts[1:length(root_parts)] == root_parts
end

function resolve_workspace_path(
    path::AbstractString;
    root::AbstractString = workspace_root(),
    must_exist::Bool = false,
)
    candidate = isabspath(path) ? normpath(path) : normpath(abspath(joinpath(root, path)))
    is_within_workspace(candidate; root = root) ||
        throw(ArgumentError("Path escapes workspace root: $path"))

    if must_exist && !ispath(candidate)
        throw(ArgumentError("Path not found: $candidate"))
    end

    return candidate
end

function optional_string(value, default::String)
    if value === nothing
        return default
    elseif value isa String
        return value
    end
    return string(value)
end

export create_error_response, create_json_response, validate_required_params,
       validate_enum_param, normalize_string_dict, workspace_root,
       is_within_workspace, resolve_workspace_path, optional_string
