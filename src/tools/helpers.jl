using ModelContextProtocol

"""
Helper functions for MCP tool handlers.
Provides consistent error handling and input validation.
"""

"""
    create_error_response(error_message::String) -> TextContent

Create a standardized error response for MCP tool handlers.
Returns a TextContent with the error message prefixed with "Error: ".
"""
function create_error_response(error_message::String)
    return TextContent(text = "Error: " * error_message)
end

"""
    create_error_response(e::Exception) -> TextContent

Create a standardized error response from an exception with full stack trace.
"""
function create_error_response(e::Exception)
    err_msg = sprint(showerror, e, catch_backtrace())
    return TextContent(text = "Error: " * err_msg)
end

"""
    validate_required_params(params::Dict, required_keys::Vector{String}) -> Union{Nothing, String}

Validate that all required parameters are present. Returns error message if validation fails, nothing otherwise.
"""
function validate_required_params(params::Dict, required_keys::Vector{String})
    for key in required_keys
        if !haskey(params, key)
            return "Missing required parameter: $key"
        end
        if params[key] === nothing || (params[key] isa String && isempty(params[key]))
            return "Parameter '$key' cannot be empty"
        end
    end
    return nothing
end

"""
    validate_enum_param(params::Dict, param_name::String, valid_values::Vector{String}, default::Union{String,Nothing}=nothing) -> Union{String, Nothing}

Validate that a parameter value is in the allowed enum values. Returns error message if validation fails, nothing otherwise.
If parameter is missing and default is provided, returns nothing (valid).
"""
function validate_enum_param(params::Dict, param_name::String, valid_values::Vector{String}, default::Union{String,Nothing}=nothing)
    if !haskey(params, param_name)
        return default === nothing ? "Missing required parameter: $param_name" : nothing
    end

    value = params[param_name]
    if !(value in valid_values)
        return "Invalid value for '$param_name': $value. Must be one of: $(join(valid_values, ", "))"
    end
    return nothing
end

export create_error_response, validate_required_params, validate_enum_param
